import os
import uuid
import pickle

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_classic.retrievers import MultiVectorRetriever
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Configuration
# ============================================================

# Directory containing this file:  
# Multi-modal-rag/data_preprocessing/ 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Actual persisted data: 
# # Multi-modal-rag/data_preprocessing/loaded_data/ 
LOADED_DATA_DIR = os.path.join( BASE_DIR, "loaded_data" ) 

FAISS_DIR = os.path.join( LOADED_DATA_DIR, "faiss_index" ) 

PARENT_STORE_FILE = os.path.join( LOADED_DATA_DIR, "parent_store.pkl" ) 

ID_KEY = "doc_id"

# ============================================================
# Embedding Model
# ============================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ============================================================
# Build and Save Retriever
# ============================================================

def build_and_save_retriever(
    texts,
    tables,
    images,
    text_summaries,
    table_summaries,
    image_summaries
):
    """
    Build a MultiVectorRetriever using:

    - FAISS for child-summary embeddings
    - InMemoryStore for original parent documents

    Then persist both to disk.

    Returns:
        MultiVectorRetriever
    """

    print("\nBuilding vector store...")


    # ========================================================
    # Create storage directory
    # ========================================================

    os.makedirs(
        LOADED_DATA_DIR,
        exist_ok=True
    )


    # ========================================================
    # Create Parent Store
    # ========================================================

    store = InMemoryStore()


    # ========================================================
    # Child Documents
    #
    # These are the summaries that will be embedded
    # and stored in FAISS.
    # ========================================================

    child_documents = []


    # ========================================================
    # Parent Documents
    #
    # These are the original extracted contents.
    # They will be stored in the parent store.
    # ========================================================

    parent_documents = []


    # ========================================================
    # Add Text
    # ========================================================

    print("Adding text documents...")

    if len(texts) != len(text_summaries):

        raise ValueError(
            f"Number of texts ({len(texts)}) does not match "
            f"number of text summaries ({len(text_summaries)})."
        )


    for text, summary in zip(
        texts,
        text_summaries
    ):

        doc_id = str(uuid.uuid4())


        # ----------------------------------------------------
        # Child document
        #
        # This SUMMARY gets embedded into FAISS.
        # ----------------------------------------------------

        child_documents.append(
            Document(
                page_content=summary,
                metadata={
                    ID_KEY: doc_id,
                    "type": "text"
                }
            )
        )


        # ----------------------------------------------------
        # Parent document
        #
        # This ORIGINAL text is stored in the parent store.
        # ----------------------------------------------------

        parent_documents.append(
            (
                doc_id,
                Document(
                    page_content=str(text),
                    metadata={
                        "type": "text"
                    }
                )
            )
        )


    # ========================================================
    # Add Tables
    # ========================================================

    print("Adding table documents...")

    if len(tables) != len(table_summaries):

        raise ValueError(
            f"Number of tables ({len(tables)}) does not match "
            f"number of table summaries ({len(table_summaries)})."
        )


    for table, summary in zip(
        tables,
        table_summaries
    ):

        doc_id = str(uuid.uuid4())


        # ----------------------------------------------------
        # Child document
        #
        # Table SUMMARY gets embedded into FAISS.
        # ----------------------------------------------------

        child_documents.append(
            Document(
                page_content=summary,
                metadata={
                    ID_KEY: doc_id,
                    "type": "table"
                }
            )
        )


        # ----------------------------------------------------
        # Parent document
        #
        # Store the original table HTML.
        # ----------------------------------------------------

        table_html = table.metadata.text_as_html


        parent_documents.append(
            (
                doc_id,
                Document(
                    page_content=table_html,
                    metadata={
                        "type": "table"
                    }
                )
            )
        )


    # ========================================================
    # Add Images
    # ========================================================

    print("Adding image documents...")

    if len(images) != len(image_summaries):

        raise ValueError(
            f"Number of images ({len(images)}) does not match "
            f"number of image summaries ({len(image_summaries)})."
        )


    for image, summary in zip(
        images,
        image_summaries
    ):

        doc_id = str(uuid.uuid4())


        # ----------------------------------------------------
        # Child document
        #
        # Image SUMMARY gets embedded into FAISS.
        # ----------------------------------------------------

        child_documents.append(
            Document(
                page_content=summary,
                metadata={
                    ID_KEY: doc_id,
                    "type": "image"
                }
            )
        )


        # ----------------------------------------------------
        # Parent document
        #
        # Store the ORIGINAL image as base64.
        # ----------------------------------------------------

        parent_documents.append(
            (
                doc_id,
                Document(
                    page_content=image,
                    metadata={
                        "type": "image",
                        "encoding": "base64"
                    }
                )
            )
        )


    # ========================================================
    # Validate
    # ========================================================

    if not child_documents:

        raise ValueError(
            "No child documents were created. "
            "Nothing can be added to FAISS."
        )


    if not parent_documents:

        raise ValueError(
            "No parent documents were created."
        )


    print(
        f"Total child documents: {len(child_documents)}"
    )

    print(
        f"Total parent documents: {len(parent_documents)}"
    )


    # ========================================================
    # Create FAISS Vector Store
    # ========================================================

    print("\nCreating FAISS index...")

    vectorstore = FAISS.from_documents(
        documents=child_documents,
        embedding=embedding_model
    )


    # ========================================================
    # Create MultiVectorRetriever
    # ========================================================

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=ID_KEY
    )


    # ========================================================
    # Add Parent Documents to InMemoryStore
    # ========================================================

    print("Adding parent documents to memory...")

    retriever.docstore.mset(
        parent_documents
    )


    # ========================================================
    # Save FAISS Index
    # ========================================================

    print("Saving FAISS index...")

    vectorstore.save_local(
        FAISS_DIR
    )


    # ========================================================
    # Save Parent Store
    # ========================================================

    print("Saving parent document store...")

    parent_store_data = dict(
        parent_documents
    )


    with open(
        PARENT_STORE_FILE,
        "wb"
    ) as f:

        pickle.dump(
            parent_store_data,
            f
        )


    # ========================================================
    # Finished
    # ========================================================

    print("\n" + "=" * 60)
    print("VECTOR STORE CREATED")
    print("=" * 60)

    print(
        f"FAISS index saved at:"
        f"\n{os.path.abspath(FAISS_DIR)}"
    )

    print(
        f"\nParent store saved at:"
        f"\n{os.path.abspath(PARENT_STORE_FILE)}"
    )


    return retriever


# ============================================================
# Load Existing Retriever
# ============================================================

def load_retriever():
    """
    Load an existing FAISS index and parent document store.

    Returns:
        MultiVectorRetriever
    """

    # ========================================================
    # Check FAISS
    # ========================================================

    if not os.path.exists(FAISS_DIR):

        raise FileNotFoundError(
            f"FAISS index not found at:\n"
            f"{os.path.abspath(FAISS_DIR)}\n\n"
            "Run build_and_save_retriever() first."
        )


    # ========================================================
    # Check Parent Store
    # ========================================================

    if not os.path.exists(PARENT_STORE_FILE):

        raise FileNotFoundError(
            f"Parent store not found at:\n"
            f"{os.path.abspath(PARENT_STORE_FILE)}\n\n"
            "Run build_and_save_retriever() first."
        )


    # ========================================================
    # Load FAISS
    # ========================================================

    print("Loading FAISS index...")

    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )


    # ========================================================
    # Create InMemoryStore
    # ========================================================

    store = InMemoryStore()


    # ========================================================
    # Load Parent Store
    # ========================================================

    print("Loading parent document store...")

    with open(
        PARENT_STORE_FILE,
        "rb"
    ) as f:

        parent_store_data = pickle.load(f)


    # ========================================================
    # Restore Parent Documents
    # ========================================================

    store.mset(
        list(
            parent_store_data.items()
        )
    )


    # ========================================================
    # Recreate MultiVectorRetriever
    # ========================================================

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=ID_KEY
    )


    # ========================================================
    # Finished
    # ========================================================

    print("\n" + "=" * 60)
    print("RETRIEVER LOADED")
    print("=" * 60)

    print(
        f"FAISS index:"
        f"\n{os.path.abspath(FAISS_DIR)}"
    )

    print(
        f"\nParent store:"
        f"\n{os.path.abspath(PARENT_STORE_FILE)}"
    )


    return retriever