# src/ingest.py
import os
import base64
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from src.config import PDF_PATH,IMAGE_DIR

def partition_document(file_path=PDF_PATH):
    print(f"Partitioning document...{file_path}")

    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )
    for i, element in enumerate(elements):
        if element.category == "Image":
            # Extract base64 data from metadata
            img_base64 = element.metadata.image_base64
            img_data = base64.b64decode(img_base64)
            
            # Save file
            file_name = f"image_{i}.png"
            path = os.path.join(IMAGE_DIR, file_name)
            with open(path, "wb") as f:
                f.write(img_data)
                
    print(f"✅ Saved {len([e for e in elements if e.category == 'Image'])} images to {IMAGE_DIR}")
    return elements

def create_chunks_by_title(elements):
    print("Creating chunks by title...")
    return chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )