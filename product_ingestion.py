import os
import ujson
import csv
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import json
from datetime import datetime

def flatten_specifications(specs):
    flat_parts = []
    for section, entries in specs.items():
        flat_parts.append(f"--- {section} ---")
        for key, val in entries.items():
            flat_parts.append(f"{key}: {val}")
    return "\n".join(flat_parts)

def load_amazon_reviews(reviews_path):
    """Load Amazon reviews from CSV/TXT file"""
    reviews_dict = {}
    with open(reviews_path, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            material_id, asin, url, reviews = line.strip().split('|')
            reviews_dict[material_id] = {
                'asin': asin,
                'url': url,
                'reviews': reviews
            }
    return reviews_dict

def load_and_process_products(product_path, reviews_path=None, max_characters=2000):
    """Load and transform JSON into flat docs with optional Amazon reviews"""
    # Load Amazon reviews if path provided
    reviews_dict = load_amazon_reviews(reviews_path) if reviews_path else {}
    
    with open(product_path) as f:
        data = ujson.load(f)["products"]

    corpus = []
    metadata = []
    
    for product_id, product in tqdm(data.items()):
        # Basic product information
        lines = [
            f"Model: {product.get('model_name', '')}",
            f"Category: {product.get('category', '')}"
        ]

        if "basic_info" in product:
            for key, val in product["basic_info"].items():
                lines.append(f"{key.capitalize()}: {val}")

        if "specifications" in product:
            lines.append(flatten_specifications(product["specifications"]))
            
        # Add Amazon reviews if available
        material_id = product.get('material_id', '')  # Assuming this is the field name
        if material_id in reviews_dict:
            lines.append("--- Amazon Reviews ---")
            lines.append(f"Customer Feedback: {reviews_dict[material_id]['reviews']}")
            lines.append(f"Amazon URL: {reviews_dict[material_id]['url']}")

        corpus_text = "\n".join(lines).strip()[:max_characters]
        corpus.append(corpus_text)
        
        # Add metadata
        metadata.append({
            "product_id": product_id,
            "material_id": material_id,
            "category": product.get('category', ''),
            "model_name": product.get('model_name', ''),
            "has_reviews": material_id in reviews_dict
        })

    print(f"Loaded {len(corpus)} products into corpus.")
    return corpus, metadata

def get_or_create_chroma_client(persist_dir="chroma_db"):
    """Get or create ChromaDB client"""
    return chromadb.PersistentClient(path=persist_dir)

def get_or_create_collection(chroma_client, collection_name="products"):
    """Get existing collection or create a new one"""
    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )

def initialize_chroma(product_path, reviews_path=None, collection_name="products", persist_dir="chroma_db"):
    """Initialize ChromaDB with product corpus and reviews only if needed"""
    chroma_client = get_or_create_chroma_client(persist_dir)
    collection = get_or_create_collection(chroma_client, collection_name)
    
    # Check if collection is empty
    if collection.count() == 0:
        print("Collection is empty. Loading products and reviews...")
        corpus, metadata = load_and_process_products(product_path, reviews_path)
        
        print("Adding products and reviews to ChromaDB...")
        collection.add(
            documents=corpus,
            metadatas=metadata,
            ids=[str(i) for i in range(len(corpus))]
        )
        print("Products and reviews added successfully!")
    else:
        print(f"Using existing collection with {collection.count()} products")

    return chroma_client, collection

def get_product_retriever(chroma_client, collection_name="products", persist_dir="chroma_db", k=2):
    """Get ChromaDB retriever model"""
    from dspy.retrieve.chromadb_rm import ChromadbRM
    
    retriever_model = ChromadbRM(
        client=chroma_client,
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        k=k
    )
    
    return retriever_model

def extract_chroma_collection(collection_name="products", persist_dir="chroma_db", output_dir="chroma_exports"):
    """
    Extract and save ChromaDB collection data to understand the ingestion format
    Returns the collection data and saves it to a JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get ChromaDB client and collection
    chroma_client = get_or_create_chroma_client(persist_dir)
    collection = get_or_create_collection(chroma_client, collection_name)
    
    # Get all data from collection
    collection_data = collection.get(
        include=['documents', 'metadatas', 'embeddings']
    )
    
    # Create a structured format for better understanding
    extracted_data = {
        "collection_name": collection_name,
        "total_items": collection.count(),
        "extraction_time": datetime.now().isoformat(),
        "sample_items": []
    }
    
    # Process each item
    for i in range(min(5, len(collection_data['ids']))):  # Save first 5 items as samples
        item = {
            "id": collection_data['ids'][i],
            "document": collection_data['documents'][i],
            "metadata": collection_data['metadatas'][i],
            "embedding_info": {
                "dimension": len(collection_data['embeddings'][i]),
                "embedding_sample": collection_data['embeddings'][i][:5],  # First 5 dimensions as sample
                "embedding_stats": {
                    "min": min(collection_data['embeddings'][i]),
                    "max": max(collection_data['embeddings'][i]),
                    "avg": sum(collection_data['embeddings'][i]) / len(collection_data['embeddings'][i])
                }
            }
        }
        extracted_data["sample_items"].append(item)
    
    # Add collection statistics
    extracted_data["statistics"] = {
        "document_avg_length": sum(len(doc) for doc in collection_data['documents']) / len(collection_data['documents']),
        "metadata_keys": list(collection_data['metadatas'][0].keys()) if collection_data['metadatas'] else [],
        "embedding_dimension": len(collection_data['embeddings'][0]) if collection_data['embeddings'] else 0
    }
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"chroma_collection_{collection_name}_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(extracted_data, f, indent=2)
    
    print(f"Collection data extracted and saved to: {output_file}")
    return extracted_data

def analyze_chroma_collection(collection_name="products", persist_dir="chroma_db"):
    """
    Analyze the ChromaDB collection and print useful information
    """
    chroma_client = get_or_create_chroma_client(persist_dir)
    collection = get_or_create_collection(chroma_client, collection_name)
    
    # Get all data
    collection_data = collection.get(
        include=['documents', 'metadatas']
    )
    
    # Print analysis
    print("\n=== ChromaDB Collection Analysis ===")
    print(f"Collection Name: {collection_name}")
    print(f"Total Items: {collection.count()}")
    
    # Analyze metadata
    if collection_data['metadatas']:
        metadata_keys = set()
        has_reviews_count = 0
        categories = set()
        
        for metadata in collection_data['metadatas']:
            metadata_keys.update(metadata.keys())
            if metadata.get('has_reviews'):
                has_reviews_count += 1
            if 'category' in metadata:
                categories.add(metadata['category'])
        
        print("\nMetadata Analysis:")
        print(f"Available Fields: {', '.join(metadata_keys)}")
        print(f"Products with Reviews: {has_reviews_count}")
        print(f"Product Categories: {', '.join(categories)}")
    
    # Analyze documents
    if collection_data['documents']:
        doc_lengths = [len(doc) for doc in collection_data['documents']]
        avg_length = sum(doc_lengths) / len(doc_lengths)
        
        print("\nDocument Analysis:")
        print(f"Average Document Length: {avg_length:.2f} characters")
        print(f"Shortest Document: {min(doc_lengths)} characters")
        print(f"Longest Document: {max(doc_lengths)} characters")
        
        # Sample content sections
        sample_doc = collection_data['documents'][0]
        sections = [line.split(':')[0] for line in sample_doc.split('\n') if ':' in line]
        print("\nTypical Document Sections:")
        print('\n'.join(f"- {section}" for section in sections[:5]))
    
    print("\n=== Analysis Complete ===") 