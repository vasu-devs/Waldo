# 🪪 System Card: Multimodal Agentic RAG Pipeline

<table>
<tr>
<td><strong>Developer</strong></td>
<td>Vasudev Siddh (<a href="https://github.com/Vasu-DevS">@Vasu-DevS</a>)</td>
</tr>
<tr>
<td><strong>Date</strong></td>
<td>January 2026</td>
</tr>
<tr>
<td><strong>Purpose</strong></td>
<td>Technical Assessment for Software Developer Intern (SOS 42)</td>
</tr>
<tr>
<td><strong>Repository</strong></td>
<td><a href="https://github.com/Vasu-DevS/SOS42">github.com/Vasu-DevS/SOS42</a></td>
</tr>
</table>

---

## 1. System Overview

This is a **production-grade Agentic Retrieval-Augmented Generation (RAG) system** designed to ingest, understand, and query complex PDF documents containing text, data tables, and visual figures.

Unlike traditional text-only RAG pipelines, this system **treats visual data as first-class citizens**, using Vision Language Models (VLMs) to transcribe and index non-textual elements.

### What Makes This Different?

| Traditional RAG | This Multimodal RAG |
|-----------------|---------------------|
| Text extraction only | Text, tables, AND figures |
| Static retrieval | Agentic self-correction |
| Single-pass answers | Query rewriting with retries |
| No visual understanding | Gemini VLM transcription |
| Basic keyword search | Semantic + metadata hybrid search |

---

## 2. Core Capabilities

### 📄 Multimodal Ingestion
Automatically detects and segments **text**, **tables**, and **figures** using IBM's Docling library. Each element type is processed through specialized pipelines.

### 🖼️ Visual Transcription
Uses **Gemini 2.0 Flash** to generate semantic "shadow text" descriptions for charts and diagrams, making them **searchable via natural language queries**.

### 🧠 Agentic Orchestration
Implements a **LangGraph state machine** with adaptive routing, self-correction, and query rewriting capabilities. The system can "think" and retry when initial results are unsatisfactory.

### 🔍 Hybrid Retrieval
Combines **semantic vector search** (Qdrant with Sentence Transformers) with **metadata filtering** to retrieve precise page-level context with figure associations.

### 🛡️ Anti-Hallucination
Strict guardrails ensure the system **refuses to answer** when queries fall outside the document context, preventing fabricated responses.

---

## 3. Technical Architecture

| Component | Technology Selected | Rationale |
|-----------|---------------------|-----------|
| **Orchestrator** | LangGraph | Enables cyclic graphs for "retry" logic (rewriting queries) rather than linear chains |
| **Vision Model** | Gemini 2.0 Flash | High-speed, low-cost transcription of tables and figures into Markdown/Text |
| **Inference Engine** | Groq (Llama 3.3 70B) | Ultra-low latency critical for "Chat" experience; GPT-4 class reasoning |
| **Vector Database** | Qdrant | Efficient payload filtering (metadata) and high-dimensional vectors |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | Fast, high-quality 384-dim embeddings |
| **PDF Processing** | IBM Docling | State-of-the-art document structure extraction |
| **Backend** | FastAPI | Async processing with modern Python |
| **Frontend** | React + Vite + TailwindCSS | Responsive, modern chat interface |

---

## 4. Agentic Workflow

The system follows a **cognitive architecture** that doesn't simply retrieve and answer. It reasons about the quality of its retrieval and self-corrects.

### 4.1 High-Level State Machine

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                   │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    🔀 ROUTER NODE                                   │
│  Decides: Is this a RAG query or a greeting/chitchat?               │
│  • If greeting → Direct response                                    │
│  • If RAG query → Proceed to retrieval                              │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ RAG Query
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    📥 RETRIEVE NODE                                 │
│  Semantic search using Qdrant:                                      │
│  • Embed query with Sentence Transformers                           │
│  • Fetch top-K (10) most similar documents                          │
│  • Returns: text chunks + figure metadata                           │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ⚖️ GRADE NODE (Self-Reflection)                  │
│  LLM evaluates each retrieved document:                             │
│  • "Is this document relevant to the user's question?"              │
│  • Binary grading: 'yes' or 'no'                                    │
│  • Special handling for figures (visual queries)                    │
│  • Limits figures to 1 per response (most relevant)                 │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
    Has Relevant Docs?           No Relevant Docs
              │                       │
              ▼                       ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│    📝 GENERATE NODE     │   │    🔄 REWRITE NODE      │
│  Synthesize answer from │   │  Transform the query    │
│  relevant documents     │   │  to improve retrieval   │
│  • Grounded in context  │   │  • Max 2 retries        │
│  • Markdown formatting  │   │  • Then refuse if fail  │
│  • Inline figures       │   │                         │
└─────────────────────────┘   └───────────┬─────────────┘
              │                           │
              │                           └──→ Back to RETRIEVE
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    📤 RESPONSE TO USER                              │
│  Formatted answer with:                                             │
│  • Bullet points and headers                                        │
│  • Bold key terms                                                   │
│  • Inline figure (if relevant)                                      │
│  • Follow-up question suggestion                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Detailed Workflow Steps

#### Step 1: Query Routing
```python
# Router decides query type
if query matches ["hi", "hello", "hey", etc.]:
    return "DIRECT" → Skip retrieval, respond directly
else:
    return "RETRIEVE" → Proceed with RAG pipeline
```

#### Step 2: Document Retrieval
```python
# Semantic search with Qdrant
query_embedding = sentence_transformer.encode(query)
results = qdrant.search(
    collection="pdf_documents",
    query_vector=query_embedding,
    limit=10  # TOP_K
)
# Returns: shadow_text, element_type, page_number, image_path
```

#### Step 3: Document Grading
```python
# LLM grades each document
for doc in retrieved_documents:
    prompt = f"""Is this document relevant to: "{query}"?
    Document: {doc.shadow_text}
    Answer 'yes' or 'no'."""
    
    if llm_response == "yes":
        relevant_docs.append(doc)
        
    # Special: Limit figures to 1
    if doc.type == "figure" and figure_count >= 1:
        skip("already have a figure")
```

#### Step 4: Query Rewriting (If Needed)
```python
# If no relevant docs, try rewriting
if len(relevant_docs) == 0 and retry_count < 2:
    new_query = llm.rewrite(
        f"Rewrite this query for better retrieval: {query}"
    )
    return to_retrieve_node(new_query)
    
if retry_count >= 2:
    return "I don't have information about that in the uploaded document."
```

#### Step 5: Grounded Generation
```python
# Generate answer ONLY from context
prompt = f"""
CRITICAL: Only answer based on these documents.
If the answer is not in the documents, refuse.

Documents: {context}
Question: {query}

Answer:
"""
response = groq_llm.generate(prompt, temperature=0.3)
```

---

## 5. Ingestion Pipeline Workflow

### 5.1 PDF Processing Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    📄 PDF UPLOAD                                    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    🔧 DOCLING PARSER                                │
│  IBM's Document Understanding Library:                              │
│  • Structure detection (headers, paragraphs, lists)                 │
│  • Table extraction with cell boundaries                            │
│  • Figure detection with bounding boxes                             │
│  • OCR for scanned documents (via RapidOCR)                         │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │   TEXT   │    │  TABLES  │    │ FIGURES  │
   │  Chunks  │    │ (Images) │    │ (Images) │
   └────┬─────┘    └────┬─────┘    └────┬─────┘
        │               │               │
        ▼               ▼               ▼
   ┌──────────┐    ┌──────────────────────────┐
   │ Recursive │    │    GEMINI 2.0 FLASH     │
   │ Character │    │    VLM Transcription    │
   │ Splitting │    │  (35s delay/rate limit) │
   │ 1000/200  │    │                         │
   └────┬─────┘    └────────────┬─────────────┘
        │                       │
        │                       ▼
        │               ┌──────────────────────────┐
        │               │    SHADOW TEXT           │
        │               │  Rich semantic description│
        │               │  Figure 1 | Receptors... │
        │               └────────────┬─────────────┘
        │                            │
        └────────────┬───────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    🧲 EMBEDDING                                     │
│  Sentence Transformers (all-MiniLM-L6-v2)                           │
│  • 384 dimensions                                                   │
│  • Fast inference                                                   │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    💾 QDRANT STORAGE                                │
│  Vector + Payload:                                                  │
│  • shadow_text: Searchable text content                             │
│  • element_type: "text", "table", "figure"                          │
│  • page_number: Source page                                         │
│  • original_image_path: For figure rendering                        │
│  • source_pdf: Document identifier                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Figure Metadata Enrichment

Each figure is stored with **rich contextual metadata** for accurate retrieval:

```json
{
  "id": "uuid-12345",
  "shadow_text": "[Figure 1] Receptors in the human skin: Mechanoreceptors can be free receptors or encapsulated. Examples for free receptors are the hair receptors at the roots of hairs. Encapsulated receptors are the Pacinian corpuscles and the receptors in the glabrous (hairless) skin: Meissner corpuscles, Ruffini corpuscles and Merkel's disks. | Visual element from Page 1. Image file: figure_1_page_1.png",
  "element_type": "figure",
  "page_number": 1,
  "original_image_path": "E:/output/figure_1_page_1.png",
  "heading": "Figure 1"
}
```

**Shadow Text Sources:**
1. Figure number label (Figure 1, Figure 2...)
2. Docling caption extraction
3. Annotation text from PDF
4. Regex search for "Figure X:" in markdown
5. Gemini VLM detailed description (when quota available)

---

## 6. Safety & Hallucination Guardrails

To ensure reliability in an enterprise context, the system enforces **strict boundaries**:

### 6.1 Zero-Context Refusal
```python
if len(relevant_documents) == 0:
    return "I don't have information about that in the uploaded document."
```

### 6.2 Grounded Generation Prompt
```
CRITICAL RULE - NO HALLUCINATION:
* You can ONLY answer based on the document fragments provided below.
* If the question asks about something NOT in the documents, respond: 
  "I don't have information about that in the uploaded document."
* NEVER make up information. NEVER use external knowledge.
* If you're not 100% sure the answer is in the context, refuse to answer.
```

### 6.3 Low Temperature
- **Temperature: 0.3** (reduced from default 0.7)
- Minimizes creative fabrication
- Prioritizes factual, consistent responses

### 6.4 Figure Limiting
- Maximum **1 figure per response**
- Only the most semantically relevant figure is shown
- Prevents information overload

---

## 7. Performance Validation

### 7.1 Multimodal Success
✅ Successfully extracted and described complex figures:
- Medical diagrams (e.g., somatosensory system receptors)
- Technical charts and tables
- Figure-text associations preserved

### 7.2 Stress Test Results

| Test Case | Result | Notes |
|-----------|--------|-------|
| **Multi-hop Reasoning** | ✅ PASSED | Correctly linked disparate concepts across document sections |
| **Entity Extraction** | ✅ PASSED | Precise extraction of numerical data and proper nouns |
| **Hallucination Check** | ✅ PASSED | Correctly refused to invent information not present in the text |
| **Out-of-Scope Query** | ✅ PASSED | Refused to answer questions about unrelated topics |
| **Figure Retrieval** | ✅ PASSED | Returns correct figure when asked about diagram content |
| **Figure Numbering** | ✅ PASSED | "Show me Figure 1" returns the correct labeled figure |

### 7.3 Sample Interactions

**In-Scope Query:**
```
User: "Where are Meissner corpuscles located?"
System: "Meissner's corpuscles are located in the **glabrous (hairless) skin**, 
found in high density in the **digits** and around the **mouth**..."
[Figure 1 displayed inline]
```

**Out-of-Scope Query:**
```
User: "What is the capital of France?"
System: "I don't have information about that in the uploaded document.
I can only answer questions related to the content of the PDF you've uploaded."
```

---

## 8. Limitations & Future Work

### Current Limitations
- **Gemini Rate Limits**: Free tier requires 35s delays between figure transcriptions
- **Single Document**: Currently processes one PDF at a time
- **No Persistent Storage**: Vector DB resets on restart

### Future Improvements
- [ ] Multi-document support with source attribution
- [ ] Persistent Qdrant collections across sessions
- [ ] Streaming responses for long answers
- [ ] Fine-tuned embedding model for domain-specific documents
- [ ] Table-to-SQL conversion for structured queries

---

## 9. Running the System

### Prerequisites
```bash
# Required
Python 3.11+
Node.js 18+
Docker (for Qdrant)

# API Keys
GROQ_API_KEY=...
GOOGLE_API_KEY=...
```

### Quick Start
```bash
# 1. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 2. Start Backend
.\.venv\Scripts\python -m uvicorn backend.main:app --port 8000

# 3. Start Frontend
cd frontend && npm run dev

# 4. Open http://localhost:5173
```

---

<p align="center">
<strong>Built with 🔥 by Vasudev Siddh</strong><br>
<a href="https://github.com/Vasu-DevS">GitHub</a> • January 2026
</p>
