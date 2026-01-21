"""
Graph Brain - LangGraph State Machine for Agentic RAG.

Implements a multi-node pipeline with:
- retrieve: Fetch docs from Qdrant
- grade_documents: Gemini 2.0 relevance grading
- generate: Multimodal generation with text + images
- rewrite_query: Query transformation when all docs are irrelevant
"""

import logging
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class Document(TypedDict):
    """A retrieved document from the vector store."""
    id: str
    shadow_text: str
    original_image_path: str
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
    
    Uses Gemini 2.0 Flash for fast grading and generation.
    """
    
    MAX_RETRIES = 2
    TOP_K = 5
    
    def __init__(
        self,
        api_key: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "pdf_documents",
        model_name: str = "gemini-2.0-flash",
    ) -> None:
        """
        Initialize graph nodes with required clients.
        
        Args:
            api_key: Google AI API key.
            qdrant_host: Qdrant server host.
            qdrant_port: Qdrant server port.
            collection_name: Qdrant collection name.
            model_name: Gemini model to use.
        """
        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        # Initialize Qdrant
        from qdrant_client import QdrantClient
        try:
            self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
            logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}")
            logger.info("Using in-memory Qdrant for development")
            self.qdrant = QdrantClient(":memory:")
        
        self.collection_name = collection_name
    
    # -------------------------------------------------------------------------
    # RETRIEVE NODE
    # -------------------------------------------------------------------------
    
    def retrieve(self, state: GraphState) -> GraphState:
        """
        Fetch documents from Qdrant based on the query.
        
        Currently uses scroll (all docs) as a placeholder.
        In production, this would use semantic search with embeddings.
        
        Args:
            state: Current graph state with query.
            
        Returns:
            Updated state with retrieved documents.
        """
        query = state.get("rewritten_query") or state["query"]
        logger.info(f"[RETRIEVE] Query: '{query}'")
        
        # Scroll through documents (placeholder for semantic search)
        # TODO: Replace with embedding-based search
        try:
            result = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=self.TOP_K,
                with_payload=True,
                with_vectors=False,
            )
            
            documents: list[Document] = []
            for point in result[0]:
                doc: Document = {
                    "id": str(point.id),
                    "shadow_text": point.payload.get("shadow_text", ""),
                    "original_image_path": point.payload.get("original_image_path", ""),
                    "element_type": point.payload.get("element_type", ""),
                    "source_pdf": point.payload.get("source_pdf", ""),
                    "page_number": point.payload.get("page_number", 0),
                    "relevance_score": 0.0,
                }
                documents.append(doc)
            
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
    # GRADE DOCUMENTS NODE
    # -------------------------------------------------------------------------
    
    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Grade each document for relevance using Gemini 2.0 Flash.
        
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
        
        for doc in documents:
            # Build grading prompt
            prompt = f"""You are a relevance grader. Your task is to determine if a document is relevant to a user query.

User Query: {query}

Document Content:
{doc['shadow_text'][:2000]}

Is this document relevant to answering the user's query? 
Answer with ONLY 'yes' or 'no'. Nothing else."""

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=10,
                    ),
                )
                
                grade = response.text.strip().lower()
                is_relevant = grade == "yes"
                
                logger.info(
                    f"[GRADE] Doc {doc['id'][:8]}... "
                    f"({doc['element_type']}, p{doc['page_number']}): {grade}"
                )
                
                if is_relevant:
                    doc["relevance_score"] = 1.0
                    relevant_documents.append(doc)
                    
            except Exception as e:
                logger.error(f"[GRADE] Error grading doc {doc['id']}: {e}")
                # On error, include the document to be safe
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
        Generate final answer using Gemini 2.0 with relevant docs + images.
        
        Passes both text (shadow_text) and image paths for multimodal RAG.
        
        Args:
            state: Current graph state with relevant_documents.
            
        Returns:
            Updated state with generation.
        """
        query = state.get("rewritten_query") or state["query"]
        docs = state["relevant_documents"]
        
        logger.info(f"[GENERATE] Using {len(docs)} relevant documents")
        
        # Build context from documents
        context_parts = []
        image_parts = []
        
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"--- Document {i} ({doc['element_type']}, Page {doc['page_number']}) ---\n"
                f"{doc['shadow_text']}\n"
            )
            
            # Load image if available
            if doc["original_image_path"]:
                try:
                    from pathlib import Path
                    img_path = Path(doc["original_image_path"])
                    if img_path.exists():
                        with open(img_path, "rb") as f:
                            image_data = f.read()
                        
                        # Determine MIME type
                        suffix = img_path.suffix.lower()
                        mime_type = {
                            ".png": "image/png",
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".webp": "image/webp",
                        }.get(suffix, "image/png")
                        
                        image_parts.append(
                            types.Part.from_bytes(data=image_data, mime_type=mime_type)
                        )
                        logger.info(f"[GENERATE] Loaded image: {img_path.name}")
                except Exception as e:
                    logger.warning(f"[GENERATE] Could not load image: {e}")
        
        context = "\n".join(context_parts)
        
        # Build generation prompt
        prompt = f"""You are a helpful assistant answering questions based on provided documents.

Context Documents:
{context}

User Question: {query}

Instructions:
1. Answer based ONLY on the provided documents.
2. If the documents contain relevant tables or figures, reference them.
3. If you cannot answer from the documents, say so clearly.
4. Be concise but thorough.

Answer:"""

        try:
            # Build content with text + images
            content_parts = [prompt] + image_parts
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                ),
            )
            
            generation = response.text
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=100,
                ),
            )
            
            rewritten = response.text.strip()
            logger.info(f"[REWRITE] '{original_query}' -> '{rewritten}'")
            
        except Exception as e:
            logger.error(f"[REWRITE] Error: {e}")
            rewritten = original_query  # Fallback to original
        
        return {
            **state,
            "rewritten_query": rewritten,
            "retry_count": retry_count + 1,
        }


# =============================================================================
# CONDITIONAL EDGES
# =============================================================================

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
    api_key: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "pdf_documents",
    model_name: str = "gemini-2.0-flash",
) -> StateGraph:
    """
    Build the compiled LangGraph state machine.
    
    Args:
        api_key: Google AI API key.
        qdrant_host: Qdrant server host.
        qdrant_port: Qdrant server port.
        collection_name: Qdrant collection name.
        model_name: Gemini model to use.
        
    Returns:
        Compiled StateGraph ready for invocation.
    """
    # Initialize node implementations
    nodes = GraphNodes(
        api_key=api_key,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=collection_name,
        model_name=model_name,
    )
    
    # Create graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("rewrite_query", nodes.rewrite_query)
    
    # Add edges
    workflow.add_edge(START, "retrieve")
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
    
    # Compile
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
    import sys
    from pathlib import Path
    
    # Add parent to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.config.settings import get_settings
    
    settings = get_settings()
    
    return build_graph(
        api_key=settings.google_api_key,
        qdrant_host=settings.qdrant_host,
        qdrant_port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
        model_name=settings.gemini_model,
    )


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
    print("Graph Brain - LangGraph RAG Pipeline")
    print("=" * 60)
    
    try:
        graph = create_rag_graph_from_settings()
        
        # Print graph structure
        print("\nGraph Structure:")
        print(graph.get_graph().draw_ascii())
        
        # Test invocation
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
