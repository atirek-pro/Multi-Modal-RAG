# Multi-Modal RAG

A local **Multi-Modal Retrieval-Augmented Generation (RAG)** pipeline for processing research PDFs containing text, tables, and images, generating multimodal summaries, indexing those summaries with FAISS, retrieving the original content, and generating answers with Gemini.

## Project Status

The project currently implements the following pipeline:

```text
PDF
 │
 ▼
Unstructured PDF Extraction
 │
 ├── Text
 ├── Tables
 └── Images
 │
 ▼
Gemini Summarization
 │
 ├── Text summaries
 ├── Table summaries
 └── Image summaries
 │
 ▼
FAISS Vector Store
 │
 ▼
MultiVectorRetriever
 │
 ▼
Original / Parent Documents
 │
 ▼
Gemini
 │
 ▼
Final Answer
```

---

# Project Structure

Current project structure:

```text
Multi-modal-rag/
│
├── content/
│   └── attention.pdf
│
├── data_preprocessing/
│   ├── data_extraction.py
│   ├── data_summarization.py
│   ├── load_data.py
│   ├── data_preprocessing_pipeline.py
│   │
│   └── loaded_data/
│       ├── faiss_index/
│       │   ├── index.faiss
│       │   └── index.pkl
│       │
│       └── parent_store.pkl
│
├── rag_pipeline.py
│
├── .env
└── README.md
```

---

# 1. PDF Data Extraction

The project uses **Unstructured** to partition and process PDF documents.

The current extraction pipeline uses:

- `strategy="hi_res"` for high-resolution PDF processing
- `infer_table_structure=True` for table extraction
- `extract_image_block_types=["Image"]` for image extraction
- Tesseract OCR for scanned/image-based text
- `chunking_strategy="by_title"` for text chunking

The extracted information is separated into:

```text
Text
Tables
Images
```

Images are extracted as Base64 data so they can later be passed to a multimodal model.

## Tesseract

Tesseract is explicitly configured because the executable was not available through the system PATH.

Current configuration:

```python
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

The extraction code verifies that this executable exists and configures `unstructured_pytesseract` to use it.

---

# 2. Multimodal Summarization

The extracted content is summarized using **Gemini 2.5 Flash**.

Different content types are handled separately.

## Text

Text chunks are summarized into concise semantic descriptions.

## Tables

Tables are converted to HTML using:

```python
table.metadata.text_as_html
```

The HTML representation is then summarized by Gemini.

## Images

Images are passed to Gemini as Base64 image data.

The image summarization prompt asks the model to describe information useful for RAG, including:

- diagrams
- graphs
- bar plots
- axes
- labels
- relationships between components
- important visual information

The result is an image summary that can be embedded and searched.

---

# 3. Embeddings

The project uses Google's embedding model:

```text
gemini-embedding-001
```

through:

```python
GoogleGenerativeAIEmbeddings
```

The embeddings are used for semantic retrieval.

During preprocessing:

```text
Summary
   │
   ▼
Embedding Model
   │
   ▼
Vector
   │
   ▼
FAISS
```

At query time, the user's question is automatically embedded by the same embedding model when the retriever is invoked.

For example:

```python
docs = retriever.invoke(question)
```

Internally:

```text
Question
   │
   ▼
embed_query()
   │
   ▼
Question Vector
   │
   ▼
FAISS similarity search
```

Therefore, the question does not need to be manually embedded in `rag_pipeline.py`.

---

# 4. FAISS Vector Store

The project uses **FAISS** instead of Chroma for vector storage.

FAISS stores the embeddings of the **child summary documents**.

Conceptually:

```text
Original Content
       │
       ▼
    Summary
       │
       ▼
   Embedding
       │
       ▼
     FAISS
```

The original content is stored separately.

This separation allows the system to search using concise semantic summaries while returning the original content to the LLM.

---

# 5. MultiVectorRetriever

The project uses:

```python
MultiVectorRetriever
```

The important concept is the relationship between:

```text
Child Summary
      │
      │ doc_id
      ▼
Parent / Original Document
```

For example:

```text
Summary A
doc_id = 123
      │
      ▼
Original Content A
```

When FAISS retrieves a summary, `MultiVectorRetriever` uses its `doc_id` to retrieve the corresponding parent document.

This means the system does not need to send the summary itself to Gemini as the only source of truth. Instead, the summary acts as the retrieval representation and the original content can be passed to the generation model.

---

# 6. Persistent Retrieval Data

The generated retrieval data is persisted under:

```text
data_preprocessing/loaded_data/
```

Current structure:

```text
loaded_data/
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
└── parent_store.pkl
```

## `index.faiss`

Contains the FAISS vector index containing embeddings of child summaries.

## `index.pkl`

Contains FAISS/LangChain index metadata and document mapping information.

## `parent_store.pkl`

Contains the original parent documents that are retrieved after a matching summary is found.

Because `InMemoryStore` itself is not persistent, its contents are serialized to `parent_store.pkl` and reconstructed when the RAG pipeline starts.

> `pickle` should only be loaded from trusted files because Python pickle deserialization can execute arbitrary code.

---

# 7. Preprocessing Pipeline

The preprocessing pipeline coordinates the entire ingestion process.

The execution flow is:

```text
data_preprocessing_pipeline.py
            │
            ▼
      extract_data()
            │
            ▼
     Text / Tables / Images
            │
            ▼
      summarize_data()
            │
            ▼
    Text / Table / Image
        Summaries
            │
            ▼
 build_and_save_retriever()
            │
            ▼
       FAISS + Parent Store
```

The preprocessing pipeline should be executed when:

- a new PDF is added
- an existing PDF changes
- the retrieval index needs to be rebuilt

Once the persisted index exists, the RAG application can load it directly without repeating preprocessing.

---

# 8. RAG Query Pipeline

`rag_pipeline.py` loads the persisted retriever and answers user questions.

Current flow:

```text
User Question
      │
      ▼
MultiVectorRetriever
      │
      ▼
Question Embedding
      │
      ▼
FAISS Similarity Search
      │
      ▼
Matching Summary Documents
      │
      ▼
doc_id
      │
      ▼
Parent Document Store
      │
      ▼
Original Text / Tables / Images
      │
      ▼
Multimodal Prompt
      │
      ▼
Gemini
      │
      ▼
Answer
```

The final generation model is configured separately from the embedding model.

Current generation model:

```text
gemini-3.1-pro-preview
```

The embedding model:

```text
gemini-embedding-001
```

These models have different responsibilities.

---

# 9. Multimodal Context

Retrieved parent documents are separated into:

```text
Text
Tables
Images
```

The RAG prompt can then provide:

- original text as text
- original table HTML as table context
- original images as multimodal image inputs

This allows Gemini to reason over multiple modalities when generating an answer.

---

# 10. Current Retrieval Design

The current implementation assigns document IDs to individual extracted items.

Conceptually, it currently behaves like:

```text
Text
 └── doc_id = A
      └── Text Summary

Table
 └── doc_id = B
      └── Table Summary

Image
 └── doc_id = C
      └── Image Summary
```

Therefore, retrieving the text summary associated with `A` retrieves the text parent associated with `A`, but does not automatically retrieve the table or image associated with `B` or `C`.

This is the main architectural improvement planned for the next iteration.

---

# Future Updates — Section-Level Multimodal Parent Grouping

The next major improvement will be to change the parent-child data model so that a **logical document section becomes the parent**, rather than treating every extracted modality as an independent parent.

## Desired Design

If one section contains:

```text
Section 5
│
├── Text
├── Table
└── Image
```

the entire section should receive **one shared `doc_id`**:

```text
doc_id = SECTION_5_123
```

Then each component can have its own retrieval summary:

```text
Text Summary
doc_id = SECTION_5_123

Table Summary
doc_id = SECTION_5_123

Image Summary
doc_id = SECTION_5_123
```

The FAISS index would therefore contain multiple child vectors pointing to the same parent.

```text
                    FAISS
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   Text Summary  Table Summary  Image Summary
        │             │             │
        └─────────────┼─────────────┘
                      │
              SAME doc_id
                      │
                      ▼
               SECTION_5_123
                      │
             ┌────────┼────────┐
             ▼        ▼        ▼
           Text     Table     Image
```

## Why This Design Is Better

This preserves the context of a multimodal section.

For example, an image may only make sense when combined with the text explaining it. Similarly, a table may have surrounding text that explains what its columns and values represent.

With section-level grouping:

```text
Question
   │
   ▼
FAISS
   │
   ▼
One or more matching summaries
   │
   ▼
Shared doc_id
   │
   ▼
Complete parent section
   │
   ├── Text
   ├── Table
   └── Image
   │
   ▼
Gemini
```

Therefore, even if only the image summary is the strongest match for a question, retrieval can return the complete parent section containing the image, its related text, and its related tables.

## Multiple Components, One Parent

The model should be flexible enough to support any combination of components.

### Text-only section

```text
Parent A
doc_id = A

└── Text
    └── Text Summary
```

### Text + table

```text
Parent B
doc_id = B

├── Text
│   └── Text Summary
│
└── Table
    └── Table Summary
```

### Text + table + image

```text
Parent C
doc_id = C

├── Text
│   └── Text Summary
│
├── Table
│   └── Table Summary
│
└── Image
    └── Image Summary
```

### Multiple images

```text
Parent D
doc_id = D

├── Text
│   └── Text Summary
│
├── Image 1
│   └── Image Summary
│
├── Image 2
│   └── Image Summary
│
└── Table
    └── Table Summary
```

All child summaries belonging to the same logical section point to the same parent `doc_id`.

## Important Retrieval Consideration

Multiple child summaries may point to the same parent.

For example:

```text
Text Summary  → doc_id = ABC
Image Summary → doc_id = ABC
Table Summary → doc_id = ABC
```

If all three are retrieved, the parent `ABC` should only be returned once.

The future retrieval logic should therefore:

```text
Retrieved child summaries
          │
          ▼
Extract doc_ids
          │
          ▼
Deduplicate doc_ids
          │
          ▼
Retrieve unique parent sections
```

This avoids sending duplicate parent content to the generation model.

---

# Overall Architecture

The target architecture is:

```text
                    PDF
                     │
                     ▼
              Unstructured
                     │
                     ▼
              Logical Sections
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      Text         Table        Image
        │            │            │
        ▼            ▼            ▼
   Text Summary  Table Summary Image Summary
        │            │            │
        └────────────┼────────────┘
                     │
              Shared parent ID
                     │
                     ▼
                   FAISS
                     │
                     ▼
               User Question
                     │
                     ▼
              Question Embedding
                     │
                     ▼
              Similarity Search
                     │
                     ▼
             Matching Summaries
                     │
                     ▼
              Shared parent IDs
                     │
                     ▼
             Complete Sections
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      Text         Table        Image
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
             Gemini Pro Model
                     │
                     ▼
                  Answer
```

---

# Environment

The project uses a Python virtual environment:

```text
.rag-env
```

Required environment variable:

```text
GOOGLE_API_KEY
```

The key is loaded from `.env`.

Example:

```text
GOOGLE_API_KEY=your_api_key_here
```

Do not commit `.env` to Git.

Recommended `.gitignore` entry:

```text
.env
.rag-env/
__pycache__/
```

---

# Running the Project

## Step 1 — Run preprocessing

From the project root:

```bat
python data_preprocessing\data_preprocessing_pipeline.py
```

This extracts the PDF, generates summaries, creates embeddings, and saves the FAISS index and parent store.

## Step 2 — Run RAG

After preprocessing completes:

```bat
python rag_pipeline.py
```

The RAG pipeline loads the existing FAISS index and parent store, embeds the question automatically through the retriever, retrieves relevant parent documents, and sends the resulting multimodal context to Gemini.

---

# Key Design Principle

The most important concept in this project is:

```text
SEARCH WITH SUMMARIES
        ↓
RETRIEVE ORIGINAL CONTENT
        ↓
GENERATE WITH ORIGINAL CONTENT
```

The summaries are optimized for **retrieval**.

The original text, tables, and images are preserved for **generation and reasoning**.

The planned section-level grouping extends this idea further:

```text
MULTIPLE MODALITY SUMMARIES
            ↓
      ONE PARENT ID
            ↓
 COMPLETE MULTIMODAL SECTION
            ↓
          GEMINI
```

This allows the RAG system to retrieve a complete piece of context instead of an isolated text, table, or image.
