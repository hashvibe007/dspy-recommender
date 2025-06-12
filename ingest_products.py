import os
from dotenv import load_dotenv
from qdrant import ingest_products

def ingest_data():
    """Ingest products and reviews into Qdrant"""
    # Get paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    product_path = os.path.join(base_dir, 'products/products_all.json')
    reviews_path = os.path.join(base_dir, 'products/Amazon-review-summary.txt')
    
    print("Starting data ingestion...")
    
    # Initialize Qdrant with products and reviews
    client = ingest_products(
        product_path=product_path,
        reviews_path=reviews_path
    )
    
    # Get collection info
    try:
        collection_info = client.get_collection("products")
        points_count = collection_info.points_count if hasattr(collection_info, 'points_count') else 'unknown'
        print(f"Ingestion complete. Collection has {points_count} items.")
    except Exception as e:
        print(f"Error getting collection info: {str(e)}")
    
    return client

if __name__ == "__main__":
    load_dotenv()
    ingest_data() 