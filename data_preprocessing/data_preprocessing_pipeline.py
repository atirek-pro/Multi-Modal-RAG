import os
from dotenv import load_dotenv
from data_extraction import extract_data
from data_summarization import summarize_data
from load_data import build_and_save_retriever

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY not found. "
        "Make sure it is defined in your .env file."
    )

# ============================================================
# Configuration
# ============================================================

OUTPUT_PATH = "./content/"
PDF_FILE = os.path.join(
    OUTPUT_PATH,
    "attention.pdf"
)


# ============================================================
# Step 1: Extract
# ============================================================

print("=" * 60)
print("STEP 1: DATA EXTRACTION")
print("=" * 60)

tables, texts, images = extract_data(PDF_FILE)


# ============================================================
# Step 2: Summarize
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: DATA SUMMARIZATION")
print("=" * 60)

text_summaries, table_summaries, image_summaries = summarize_data(
    texts,
    tables,
    images
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print("\nText summaries:")
print(text_summaries[0])

print("\nTable summaries:")
print(table_summaries[0])

print("\nNumber of images:")
print(image_summaries[0])

retriever = build_and_save_retriever(
    texts=texts,
    tables=tables,
    images=images,
    text_summaries=text_summaries,
    table_summaries=table_summaries,
    image_summaries=image_summaries
)