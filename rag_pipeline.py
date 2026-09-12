import os

from dotenv import load_dotenv

from data_preprocessing.load_data import load_retriever

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()


if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY not found. "
        "Make sure it is defined in your .env file."
    )


# ============================================================
# Load Retriever
# ============================================================

print("=" * 60)
print("LOADING RETRIEVER")
print("=" * 60)

retriever = load_retriever()


# ============================================================
# Gemini Model
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    temperature=0.2
)


# ============================================================
# Parse Retrieved Documents
# ============================================================

def parse_docs(docs):
    """
    Separate retrieved parent documents into:

    - text
    - tables
    - images

    The parent documents contain the original content.
    """

    texts = []
    tables = []
    images = []

    for doc in docs:

        doc_type = doc.metadata.get(
            "type",
            "text"
        )

        if doc_type == "image":

            images.append(
                doc.page_content
            )

        elif doc_type == "table":

            tables.append(
                doc.page_content
            )

        else:

            texts.append(
                doc.page_content
            )

    return {
        "texts": texts,
        "tables": tables,
        "images": images
    }


# ============================================================
# Build Multimodal Prompt
# ============================================================

def build_prompt(kwargs):
    """
    Build the multimodal prompt sent to Gemini.

    Context can contain:
        - text
        - tables
        - images
    """

    docs_by_type = kwargs["context"]

    user_question = kwargs["question"]


    # ========================================================
    # Build Text Context
    # ========================================================

    context_parts = []


    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    if docs_by_type["texts"]:

        context_parts.append(
            "TEXT CONTEXT:\n"
            + "\n\n".join(
                docs_by_type["texts"]
            )
        )


    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    if docs_by_type["tables"]:

        context_parts.append(
            "TABLE CONTEXT:\n"
            + "\n\n".join(
                docs_by_type["tables"]
            )
        )


    context_text = "\n\n".join(
        context_parts
    )


    # ========================================================
    # System Instructions
    # ========================================================

    system_message = SystemMessage(
        content="""
You are a helpful RAG assistant.

Answer the user's question using only the
provided context.

The context may contain:
- text
- tables
- images

Use the images when they contain information
relevant to the question.

If the answer cannot be determined from the
provided context, say that the information is
not available in the provided context.

Do not invent or assume information that is
not present in the context.

Give a clear and concise answer.
"""
    )


    # ========================================================
    # User Text
    # ========================================================

    user_content = []


    prompt_text = f"""
CONTEXT:

{context_text}

QUESTION:

{user_question}
"""


    user_content.append(
        {
            "type": "text",
            "text": prompt_text
        }
    )


    # ========================================================
    # Add Images
    # ========================================================

    for image in docs_by_type["images"]:

        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"
                        + image
                    )
                }
            }
        )


    # ========================================================
    # Return Multimodal Messages
    # ========================================================

    return [
        system_message,
        HumanMessage(
            content=user_content
        )
    ]


# ============================================================
# Build RAG Chain
# ============================================================

def retrieve_context(question):

    """
    Retrieve parent documents using the MultiVectorRetriever.

    Internally this does:

        question
            ↓
        embedding model
            ↓
        question vector
            ↓
        FAISS similarity search
            ↓
        matching child summaries
            ↓
        parent documents
    """

    print("\nRetrieving relevant documents...")

    docs = retriever.invoke(
        question
    )

    print(
        f"Retrieved {len(docs)} parent documents."
    )

    return parse_docs(docs)


chain = (
    {
        "context": RunnableLambda(
            retrieve_context
        ),
        "question": RunnablePassthrough()
    }
    | RunnableLambda(build_prompt)
    | model
)


# ============================================================
# Ask Question
# ============================================================

question = "What is the attention mechanism?"


print("\n" + "=" * 60)
print("QUESTION")
print("=" * 60)

print(question)


print("\n" + "=" * 60)
print("GENERATING RESPONSE")
print("=" * 60)


response = chain.invoke(
    question
)


# ============================================================
# Print Response
# ============================================================

print("\n" + "=" * 60)
print("RESPONSE")
print("=" * 60)

print(response.content)