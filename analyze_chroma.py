import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import json
from datetime import datetime
import numpy as np

def get_collection_info(collection_name="products", persist_dir="chroma_db"):
    """Safely get collection information"""
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        return client, collection
    except Exception as e:
        print(f"Error accessing collection: {str(e)}")
        return None, None

def safe_get_metadata_keys(metadata):
    """Safely get metadata keys"""
    if metadata is None:
        return []
    try:
        return list(metadata.keys()) if isinstance(metadata, dict) else []
    except:
        return []

def is_valid_array(arr):
    """Check if array is valid and non-empty"""
    return arr is not None and len(arr) > 0

def analyze_ingestion(collection_name="products", persist_dir="chroma_db"):
    """Analyze ChromaDB collection and save detailed information"""
    print("\n=== Starting ChromaDB Analysis ===")
    
    # First check if collection exists and has data
    client, collection = get_collection_info(collection_name, persist_dir)
    if not collection:
        print(f"Collection '{collection_name}' not found or empty. Please run ingest_products.py first.")
        return None
        
    try:
        count = collection.count()
        print(f"\nCollection '{collection_name}' contains {count} items")
        
        if count == 0:
            print("Collection is empty. Please run ingest_products.py first.")
            return None
            
        # Get a sample of data
        result = collection.get(
            limit=5,
            include=['documents', 'metadatas', 'embeddings']
        )
        
        print("\nCollection Overview:")
        print(f"- Number of items: {count}")
        
        # Safely handle metadata
        metadatas = result.get('metadatas', [])
        if is_valid_array(metadatas):
            first_valid_metadata = next((m for m in metadatas if m is not None), None)
            if first_valid_metadata:
                metadata_keys = safe_get_metadata_keys(first_valid_metadata)
                print(f"- Metadata fields: {metadata_keys}")
            else:
                print("- No valid metadata found in sample")
        else:
            print("- No metadata available")
            
        # Safely handle embeddings
        embeddings = result.get('embeddings', [])
        if is_valid_array(embeddings):
            first_valid_embedding = next((e for e in embeddings if e is not None and len(e) > 0), None)
            if first_valid_embedding is not None:
                if isinstance(first_valid_embedding, np.ndarray):
                    print(f"- Embedding dimension: {first_valid_embedding.shape[0]}")
                else:
                    print(f"- Embedding dimension: {len(first_valid_embedding)}")
            else:
                print("- No valid embeddings found in sample")
        else:
            print("- No embeddings available")
            
        # Sample document structure
        documents = result.get('documents', [])
        if is_valid_array(documents):
            print("\nSample Document Structure:")
            sample_doc = next((doc for doc in documents if doc), "")
            if sample_doc:
                sections = [line.split(':')[0].strip() for line in sample_doc.split('\n') if ':' in line]
                if sections:
                    print("Document sections:")
                    for section in sections[:5]:
                        print(f"  - {section}")
                else:
                    print("  No sections found in sample document")
            else:
                print("  No valid sample document found")
        else:
            print("\nNo documents available in collection")
                
        # Extract and save detailed data
        print("\nExtracting detailed collection data...")
        extracted_data = {
            "collection_name": collection_name,
            "total_items": count,
            "sample_items": []
        }
        
        # Process sample items
        for i in range(min(5, len(documents))):
            embedding_dim = 0
            if i < len(embeddings) and embeddings[i] is not None:
                if isinstance(embeddings[i], np.ndarray):
                    embedding_dim = embeddings[i].shape[0]
                else:
                    embedding_dim = len(embeddings[i])
                    
            item = {
                "document_preview": (documents[i][:200] + "..." 
                                   if documents[i] and len(documents[i]) > 200 
                                   else documents[i]) if i < len(documents) else None,
                "metadata": metadatas[i] if i < len(metadatas) and metadatas[i] else {},
                "embedding_dimension": embedding_dim
            }
            extracted_data["sample_items"].append(item)
            
        # Save to file
        output_dir = "chroma_exports"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"chroma_collection_{collection_name}_{timestamp}.json")
        
        # Convert numpy arrays to lists for JSON serialization
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        
        with open(output_file, 'w') as f:
            json.dump(extracted_data, f, cls=NumpyEncoder, indent=2)
            
        print(f"\nDetailed analysis saved to: {output_file}")
        return extracted_data
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    load_dotenv()
    analyze_ingestion() 