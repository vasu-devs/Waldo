"""
Graph Brain - LangGraph State Machine for Agentic RAG.

Implements a multi-node pipeline with:
- retrieve: Fetch docs from Qdrant
- grade_documents: Groq LLM relevance grading
- generate: LLM generation with context
- rewrite_query: Query transformation when all docs are irrelevant

Uses Groq for all LLM operations.
"""

import logging
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from groq import Groq

logger = logging.getLogger(__name__)


# =============================================================================
# GREETING PATTERNS
# =============================================================================

GREETING_PATTERNS: list[str] = [
    r"^\s*(hi|hey|hello|hola|sup|yo)\s*[!.,?]*\s*$",
    r"^\s*(good\s*(morning|afternoon|evening))\s*[!.,?]*\s*$",
    r"^\s*what'?s\s*up\s*[!.,?]*\s*$",
    r"^\s*how\s*are\s*you\s*[!.,?]*\s*$",
    r"^\s*help\s*[!.,?]*\s*$",
    r"^\s*(thanks|thank\s*you)\s*[!.,?]*\s*$",
]


# =============================================================================
# STATE DEFINITION
# =============================================================================

class Document(TypedDict):
    """A retrieved document from the vector store."""
    id: str
    shadow_text: str
    original_image_path: str | None
    element_type: str
    source_pdf: str
    page_number: int
    relevance_score: float


class GraphState(TypedDict):
    """State for the RAG graph.
    
    Attributes:
        query: The user's original query.
        documents: Retrieved documents from Qdrant.
        relevant_documents: Documents that passed grading.
        generation: The final generated response.
        retry_count: Number of query rewrites attempted.
        rewritten_query: The transformed query after rewrite.
    """
    query: str
    documents: list[Document]
    relevant_documents: list[Document]
    generation: str | None
    retry_count: int
    rewritten_query: str | None


# =============================================================================
# NODE IMPLEMENTATIONS
# =============================================================================

class GraphNodes:
    """
    Container for all graph node implementations.
    
    Uses Groq for fast LLM operations.
    """
    
    MAX_RETRIES = 2
    TOP_K = 10  # Retrieve more documents for better context coverage
    
    def __init__(
        self,
        groq_api_key: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "pdf_documents",
        model_name: str = "llama-3.3-70b-versatile",
    ) -> None:
        """
        Initialize graph nodes with required clients.
        
        Args:
            groq_api_key: Groq API key.
            qdrant_host: Qdrant server host.
            qdrant_port: Qdrant server port.
            collection_name: Qdrant collection name.
            model_name: Groq model to use.
        """
        # Initialize Groq client
        self.client = Groq(api_key=groq_api_key)
        self.model_name = model_name
        
        # Initialize VectorStore (handles Qdrant + embeddings)
        from backend.IngestScript.services.vector_store import VectorStore
        self.vector_store = VectorStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name,
        )
        logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
    
    # -------------------------------------------------------------------------
    # RETRIEVE NODE
    # -------------------------------------------------------------------------
    
    def retrieve(self, state: GraphState) -> GraphState:
        """
        Fetch documents from Qdrant based on the query using semantic search.
        
        Uses HYBRID RETRIEVAL: Semantic search + always include figures.
        
        Args:
            state: Current graph state with query.
            
        Returns:
            Updated state with retrieved documents.
        """
        query = state.get("rewritten_query") or state["query"]
        logger.info(f"[RETRIEVE] Query: '{query}'")
        
        try:
            # HYBRID RETRIEVAL: Get semantic results + all figures
            # 1. Semantic search for top-K results
            results = self.vector_store.search(query=query, limit=self.TOP_K)
            
            documents: list[Document] = []
            seen_ids = set()
            
            for doc in results:
                doc_id = str(doc.get("id", ""))
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                
                document: Document = {
                    "id": doc_id,
                    "shadow_text": doc.get("shadow_text", ""),
                    "original_image_path": doc.get("original_image_path"),
                    "element_type": doc.get("element_type", ""),
                    "source_pdf": doc.get("source_pdf", ""),
                    "page_number": doc.get("page_number", 0),
                    "relevance_score": doc.get("score", 0.0),
                }
                documents.append(document)
            
            logger.info(f"[RETRIEVE] Found {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"[RETRIEVE] Error: {e}")
            documents = []
        
        return {
            **state,
            "documents": documents,
            "relevant_documents": [],
        }


    
    # -------------------------------------------------------------------------
    # GRADE DOCUMENTS NODE (Hallucination Grader)
    # -------------------------------------------------------------------------
    
    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Grade each document for relevance using Groq LLM.
        
        Uses fast binary grading: 'yes' or 'no'.
        
        Args:
            state: Current graph state with documents.
            
        Returns:
            Updated state with relevant_documents filtered.
        """
        query = state.get("rewritten_query") or state["query"]
        documents = state["documents"]
        
        logger.info(f"[GRADE] Grading {len(documents)} documents...")
        
        relevant_documents: list[Document] = []
        
        # FAIL-SAFE: Generic queries auto-accept ALL documents (skip LLM)
        generic_keywords = [
            "summary", "summarize", "what is this", "about", "describe",
            "explain", "overview", "tell me", "what does", "content",
        ]
        query_lower = query.lower()
        is_generic_query = any(kw in query_lower for kw in generic_keywords)
        
        if is_generic_query:
            logger.info(f"[GRADE] Generic query detected: '{query}' -> auto-accepting ALL documents")
            for doc in documents:
                doc["relevance_score"] = 1.0
                relevant_documents.append(doc)
            return {
                **state,
                "relevant_documents": relevant_documents,
            }
        
        # INTELLIGENT VISUAL QUERY DETECTION using LLM
        # Ask the LLM if this query would benefit from visual content
        needs_visual = False
        try:
            intent_prompt = f"""Analyze this user query and determine if it would benefit from visual content (diagrams, figures, images, charts).

Query: "{query}"

Consider:
- Is the user asking about something that is better explained with a visual (anatomy, structure, process, location)?
- Does the query reference a specific figure, diagram, or visual element?
- Would seeing an image help answer the question more accurately?

Answer ONLY "visual" if visual content would significantly help, or "text" if text alone is sufficient."""
            
            intent_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            intent = intent_response.choices[0].message.content.strip().lower()
            needs_visual = "visual" in intent
            logger.info(f"[GRADE] Query intent analysis: '{query}' -> {'VISUAL' if needs_visual else 'TEXT'}")
        except Exception as e:
            logger.warning(f"[GRADE] Intent classification failed: {e}, defaulting to text-only")
        
        # Track if we've already accepted a figure for visual queries (limit to 1)
        figure_accepted = False
        
        for doc in documents:
            # Create specialized prompt based on document type
            if doc["element_type"] == "figure":
                # For figures, use strict grading
                prompt = f"""You are a STRICT figure relevance grader. Only accept figures that DIRECTLY answer the query.


User Query: {query}

Figure Description:
{doc['shadow_text'][:2000]}

STRICT RULES:
1. Answer 'yes' ONLY if this specific figure directly illustrates what the user is asking about.
2. The figure must contain the EXACT concepts, structures, or diagrams mentioned in the query.
3. Answer 'no' if the figure is only tangentially related or from a different section.
4. When in doubt, answer 'no' - only the most relevant figure should be shown.

Is this figure DIRECTLY relevant? Answer 'yes' or 'no'."""
            else:
                prompt = f"""You are a HIGH-RECALL relevance grader. Your goal is to NEVER miss relevant documents.

RULES:
1. If the document contains ANY keywords from the user question, answer 'yes'.
2. If the document is even REMOTELY related, answer 'yes'.
3. If you are unsure, answer 'yes'.
4. Only answer 'no' if the document is COMPLETELY unrelated.

User Query: {query}

Document Content:
{doc['shadow_text'][:2000]}

Is this document relevant? Answer ONLY 'yes' or 'no'."""

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                
                grade = response.choices[0].message.content.strip().lower()
                is_relevant = grade == "yes" or grade.startswith("yes")
                
                # For figures, only accept if we haven't accepted one yet
                if doc["element_type"] == "figure" and is_relevant:
                    if figure_accepted:
                        logger.info(
                            f"[GRADE] Doc {doc['id'][:8]}... "
                            f"({doc['element_type']}, p{doc['page_number']}): SKIPPED (already have a figure)"
                        )
                        continue
                    figure_accepted = True
                
                logger.info(
                    f"[GRADE] Doc {doc['id'][:8]}... "
                    f"({doc['element_type']}, p{doc['page_number']}): {grade}"
                )
                
                if is_relevant:
                    doc["relevance_score"] = 1.0
                    relevant_documents.append(doc)
                    
            except Exception as e:
                logger.error(f"[GRADE] Error grading doc {doc['id']}: {e}")
                # On error, include the document (fail-safe) but not figures
                if doc["element_type"] != "figure":
                    relevant_documents.append(doc)

        
        logger.info(f"[GRADE] {len(relevant_documents)}/{len(documents)} relevant")

        
        return {
            **state,
            "relevant_documents": relevant_documents,
        }
    
    # -------------------------------------------------------------------------
    # GENERATE NODE
    # -------------------------------------------------------------------------
    
    def generate(self, state: GraphState) -> GraphState:
        """
        Generate final answer using Groq with relevant docs.
        
        Args:
            state: Current graph state with relevant_documents.
            
        Returns:
            Updated state with generation.
        """
        query = state.get("rewritten_query") or state["query"]
        docs = state["relevant_documents"]
        
        logger.info(f"[GENERATE] Using {len(docs)} relevant documents")
        
        # OUT-OF-SCOPE CHECK: If no relevant documents, refuse to answer
        if not docs or len(docs) == 0:
            logger.info("[GENERATE] No relevant documents - refusing to answer (out of scope)")
            refusal = (
                "I don't have information about that in the uploaded document.\n\n"
                "I can only answer questions related to the content of the PDF you've uploaded. "
                "Please ask a question about the document, or upload a different PDF if you'd like to explore other topics."
            )
            return {
                **state,
                "generation": refusal,
            }
        
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"--- Document {i} ({doc['element_type']}, Page {doc['page_number']}) ---\n"
                f"{doc['shadow_text']}\n"
            )
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are a structured, data-driven AI assistant that ONLY answers from the provided document context.

CRITICAL RULE - NO HALLUCINATION:
* You can ONLY answer based on the document fragments provided below.
* If the question asks about something NOT in the documents, respond: "I don't have information about that in the uploaded document."
* NEVER make up information. NEVER use external knowledge. ONLY use what's in the context.
* If you're not 100% sure the answer is in the context, refuse to answer.

FORMATTING RULES:
1. Use Markdown `###` for section headers when organizing information.
2. Use bullet points `*` for lists, facts, details.
3. **Bold** key terms, names, numbers, and important concepts.
4. Never write a paragraph longer than 3 lines.
5. Do NOT say "Document 1 says...". Just state facts directly.

AGENTIC BEHAVIOR:
* At the end, ask ONE relevant follow-up question based on what you said.

Document Fragments:
{context}

User Question: {query}

REMEMBER: If the answer is NOT in the documents above, say "I don't have information about that in the uploaded document."

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=2048,
            )
            
            generation = response.choices[0].message.content
            logger.info(f"[GENERATE] Generated {len(generation)} chars")
            
        except Exception as e:
            logger.error(f"[GENERATE] Error: {e}")
            generation = f"Error generating response: {e}"
        
        return {
            **state,
            "generation": generation,
        }

    
    # -------------------------------------------------------------------------
    # REWRITE QUERY NODE
    # -------------------------------------------------------------------------
    
    def rewrite_query(self, state: GraphState) -> GraphState:
        """
        Rewrite the query to improve retrieval.
        
        Called when all documents are graded as irrelevant.
        
        Args:
            state: Current graph state.
            
        Returns:
            Updated state with rewritten_query and incremented retry_count.
        """
        original_query = state["query"]
        retry_count = state.get("retry_count", 0)
        
        logger.info(f"[REWRITE] Attempt {retry_count + 1}/{self.MAX_RETRIES}")
        
        prompt = f"""You are a query rewriter for a document retrieval system.

The original query did not retrieve relevant documents. Rewrite it to be:
1. More specific with key terms
2. Alternative phrasing that might match document content
3. Expanded with related concepts

Original Query: {original_query}

Rewritten Query (output ONLY the new query, nothing else):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=100,
            )
            
            rewritten = response.choices[0].message.content.strip()
            logger.info(f"[REWRITE] '{original_query}' -> '{rewritten}'")
            
        except Exception as e:
            logger.error(f"[REWRITE] Error: {e}")
            rewritten = original_query
        
        return {
            **state,
            "rewritten_query": rewritten,
            "retry_count": retry_count + 1,
        }
    
    # -------------------------------------------------------------------------
    # DIRECT RESPONSE NODE (for greetings)
    # -------------------------------------------------------------------------
    
    def direct_response(self, state: GraphState) -> GraphState:
        """
        Handle greetings and simple queries without RAG.
        
        Returns a friendly response prompting user to upload a document.
        
        Args:
            state: Current graph state.
            
        Returns:
            Updated state with generation (direct response).
        """
        query = state["query"]
        logger.info(f"[DIRECT_RESPONSE] Handling greeting: '{query}'")
        
        response = (
            "Hello! I am ready to help. Please ask me about your document."
        )
        
        return {
            **state,
            "generation": response,
            "documents": [],
            "relevant_documents": [],
        }


# =============================================================================
# CONDITIONAL EDGES
# =============================================================================

def _is_greeting(query: str) -> bool:
    """
    Check if query matches known greeting patterns.
    
    Args:
        query: User input query.
        
    Returns:
        True if query is a greeting, False otherwise.
    """
    query_lower = query.lower().strip()
    
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, query_lower, re.IGNORECASE):
            return True
    
    return False


def route_query(state: GraphState) -> str:
    """
    Route query to appropriate node based on classification.
    
    Routes greetings to 'direct_response' and questions to 'retrieve'.
    
    Args:
        state: Current graph state with query.
        
    Returns:
        'direct_response' for greetings.
        'retrieve' for RAG questions.
    """
    query = state["query"]
    
    if _is_greeting(query):
        logger.info(f"[ROUTER] Greeting detected: '{query}' -> direct_response")
        return "direct_response"
    else:
        logger.info(f"[ROUTER] RAG query: '{query}' -> retrieve")
        return "retrieve"


def should_rewrite_or_generate(state: GraphState) -> str:
    """
    Decide whether to rewrite query or proceed to generation.
    
    Returns:
        'rewrite_query' if all docs irrelevant and retries remaining.
        'generate' if any docs relevant or max retries reached.
    """
    relevant_docs = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)
    
    if len(relevant_docs) == 0 and retry_count < GraphNodes.MAX_RETRIES:
        logger.info("[ROUTER] No relevant docs -> rewrite_query")
        return "rewrite_query"
    else:
        if len(relevant_docs) == 0:
            logger.info("[ROUTER] No relevant docs but max retries -> generate")
        else:
            logger.info(f"[ROUTER] {len(relevant_docs)} relevant docs -> generate")
        return "generate"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_graph(
    groq_api_key: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "pdf_documents",
    model_name: str = "llama-3.3-70b-versatile",
) -> StateGraph:
    """
    Build the compiled LangGraph state machine.
    
    Args:
        groq_api_key: Groq API key.
        qdrant_host: Qdrant server host.
        qdrant_port: Qdrant server port.
        collection_name: Qdrant collection name.
        model_name: Groq model to use.
        
    Returns:
        Compiled StateGraph ready for invocation.
    """
    nodes = GraphNodes(
        groq_api_key=groq_api_key,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=collection_name,
        model_name=model_name,
    )
    
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("rewrite_query", nodes.rewrite_query)
    workflow.add_node("direct_response", nodes.direct_response)
    
    # Conditional entry point: route greeting vs RAG query
    workflow.add_conditional_edges(
        START,
        route_query,
        {
            "direct_response": "direct_response",
            "retrieve": "retrieve",
        },
    )
    
    # Direct response ends the graph
    workflow.add_edge("direct_response", END)
    workflow.add_edge("retrieve", "grade_documents")
    
    # Conditional edge: grade_documents -> rewrite_query OR generate
    workflow.add_conditional_edges(
        "grade_documents",
        should_rewrite_or_generate,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        },
    )
    
    # Rewrite loops back to retrieve
    workflow.add_edge("rewrite_query", "retrieve")
    
    # Generate ends the graph
    workflow.add_edge("generate", END)
    
    graph = workflow.compile()
    
    logger.info("Graph compiled successfully")
    return graph


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_rag_graph_from_settings():
    """
    Create the RAG graph using settings from environment.
    
    Returns:
        Compiled StateGraph.
    """
    # Assumes running from root
    from backend.IngestScript.config.settings import get_settings
    
    settings = get_settings()
    
    return build_graph(
        groq_api_key=settings.groq_api_key,
        qdrant_host=None, # Signal to use path
        qdrant_port=None,
        collection_name=settings.qdrant_collection_name,
        model_name=settings.groq_model,
    )


def build_graph_with_vector_store(
    groq_api_key: str,
    vector_store,
    model_name: str = "llama-3.3-70b-versatile",
):
    """
    Build the RAG graph using an EXISTING VectorStore instance.
    This ensures ingestion and RAG share the same in-memory database.
    
    Args:
        groq_api_key: Groq API key.
        vector_store: Pre-initialized VectorStore instance (shared).
        model_name: Groq model to use.
        
    Returns:
        Compiled StateGraph.
    """
    # Create nodes - we'll manually set the vector_store after creation
    nodes = object.__new__(GraphNodes)
    nodes.client = Groq(api_key=groq_api_key)
    nodes.model_name = model_name
    nodes.vector_store = vector_store  # USE SHARED INSTANCE - critical!
    logger.info(f"Using SHARED VectorStore instance")
    
    # Build graph
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("generate", nodes.generate)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("direct_response", nodes.direct_response)
    
    # Add edges - use standalone route functions
    graph.add_conditional_edges(
        START,
        route_query,  # standalone function
    )
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        should_rewrite_or_generate,  # standalone function
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("direct_response", END)
    
    graph = graph.compile()
    logger.info("Graph compiled successfully with SHARED VectorStore")
    return graph


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    print("=" * 60)
    print("Graph Brain - LangGraph RAG Pipeline (Groq)")
    print("=" * 60)
    
    try:
        graph = create_rag_graph_from_settings()
        
        print("\nGraph Structure:")
        print(graph.get_graph().draw_ascii())
        
        test_query = "What tables are in the document?"
        print(f"\nTest Query: {test_query}")
        
        initial_state: GraphState = {
            "query": test_query,
            "documents": [],
            "relevant_documents": [],
            "generation": None,
            "retry_count": 0,
            "rewritten_query": None,
        }
        
        result = graph.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Query: {result['query']}")
        print(f"Rewritten: {result.get('rewritten_query', 'N/A')}")
        print(f"Retries: {result['retry_count']}")
        print(f"Docs Retrieved: {len(result['documents'])}")
        print(f"Docs Relevant: {len(result['relevant_documents'])}")
        print(f"\nGeneration:\n{result['generation']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()