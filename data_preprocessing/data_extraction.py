import os
import base64

import unstructured_pytesseract
from unstructured.partition.pdf import partition_pdf


# ============================================================
# Tesseract Configuration
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if not os.path.exists(TESSERACT_PATH):
    raise FileNotFoundError(
        f"Tesseract executable not found at:\n{TESSERACT_PATH}"
    )

unstructured_pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# Extract Data
# ============================================================

def extract_data(file_path):

    print(f"Extracting data from: {file_path}")

    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000
    )

    print(f"Number of chunks: {len(chunks)}")

    # --------------------------------------------------------
    # Separate tables and text
    # --------------------------------------------------------

    tables = []
    texts = []

    for chunk in chunks:

        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)

    print(f"Number of tables: {len(tables)}")
    print(f"Number of text chunks: {len(texts)}")

    # --------------------------------------------------------
    # Extract images
    # --------------------------------------------------------

    images = []

    for chunk in chunks:

        if "CompositeElement" not in str(type(chunk)):
            continue

        chunk_els = chunk.metadata.orig_elements

        for el in chunk_els:

            if "Image" not in str(type(el)):
                continue

            image_base64 = getattr(
                el.metadata,
                "image_base64",
                None
            )

            if image_base64:
                images.append(image_base64)

    print(f"Number of images: {len(images)}")

    return tables, texts, images