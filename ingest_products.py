import os
from dotenv import load_dotenv
from product_ingestion import initialize_chroma

def ingest_data():
    """Ingest products and reviews into ChromaDB"""
    # Get paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    product_path = os.path.join(base_dir, 'products/products_all.json')
    reviews_path = os.path.join(base_dir, 'products/Amazon-review-summary.txt')
    
    print("Starting data ingestion...")
    
    # Initialize ChromaDB with products and reviews
    chroma_client, collection = initialize_chroma(
        product_path=product_path,
        reviews_path=reviews_path
    )
    
    print(f"Ingestion complete. Collection has {collection.count()} items.")
    return chroma_client, collection

if __name__ == "__main__":
    load_dotenv()
    ingest_data() 