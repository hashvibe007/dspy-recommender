import dspy
from dspy_qdrant import QdrantRM
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os

import os
import ujson
import csv
from tqdm import tqdm

import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

login(token=os.getenv("HUGGINGFACE_TOKEN"))


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
        material_id = product.get('material', '')  # Assuming this is the field name
        if material_id in reviews_dict:
            lines.append("--- Amazon Reviews ---")
            lines.append(f"Customer Feedback: {reviews_dict[material_id]['AI-reviews']}")
            lines.append(f"Amazon URL: {reviews_dict[material_id]['Product-url']}")

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
    print(corpus)
    print(metadata)
    return corpus, metadata

def initialize_qdrant(collection_name="products"):
    """Initialize Qdrant client and create collection if needed"""
    client = QdrantClient()
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_exists = any(col.name == collection_name for col in collections)
    
    if not collection_exists:
        # Create collection with proper configuration
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=384,  # Dimension for all-MiniLM-L6-v2
                distance=models.Distance.COSINE
            )
        )
        print(f"Created new collection: {collection_name}")
    else:
        print(f"Using existing collection: {collection_name}")
    
    return client

def ingest_products(product_path, reviews_path=None, collection_name="products"):
    """Ingest products into Qdrant"""
    # Initialize Qdrant
    client = initialize_qdrant(collection_name)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load and process products
    corpus, metadata = load_and_process_products(product_path, reviews_path)
    
    print("Generating embeddings and ingesting products...")
    batch_size = 100
    
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch_corpus = corpus[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]
        
        # Generate embeddings
        embeddings = encoder.encode(batch_corpus)
        
        # Prepare points for ingestion
        points = [
            models.PointStruct(
                id=idx + i,
                vector=embedding.tolist(),
                payload={
                    "text": text,
                    **meta
                }
            )
            for idx, (text, embedding, meta) in enumerate(zip(batch_corpus, embeddings, batch_metadata))
        ]
        
        # Upsert batch
        client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    print(f"Successfully ingested {len(corpus)} products into Qdrant")
    return client

def get_qdrant_retriever(client, collection_name="products", k=3):
    """Get Qdrant retriever model"""
    logger.info(f"Initializing Qdrant retriever with k={k}")
    
    # Check collection status
    try:
        collection_info = client.get_collection(collection_name)
        logger.info(f"Collection info: {collection_info}")
        
        # Test search to verify retrieval
        test_results = client.search(
            collection_name=collection_name,
            query_vector=[0.1] * 384,  # Test vector
            limit=1,
            with_payload=True,
            with_vectors=True
        )
        logger.info(f"Test search results: {json.dumps(test_results[0].payload, indent=2) if test_results else 'No results'}")
        
        # Get a few random points to verify data
        scroll_results = client.scroll(
            collection_name=collection_name,
            limit=3,
            with_payload=True,
            with_vectors=True
        )
        logger.info(f"Sample point payload: {json.dumps(scroll_results[0][0].payload, indent=2)}")
        
        # Log collection statistics
        logger.info(f"Collection points count: {collection_info.points_count}")
        logger.info(f"Collection vector size: {collection_info.config.params.vectors.size}")
        logger.info(f"Collection distance metric: {collection_info.config.params.vectors.distance}")
        
        # Test direct search
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_query = "washing machine with AI features"
        query_vector = model.encode(test_query).tolist()
        
        direct_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True
        )
        logger.info("Direct search results:")
        for i, result in enumerate(direct_results):
            logger.info(f"Result {i+1}: {json.dumps(result.payload, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}", exc_info=True)
        raise
    
    # Initialize retriever
    try:
        logger.info("Creating QdrantRM instance...")
        retriever = QdrantRM(
            collection_name,
            client,
            k=k
        )
        logger.info("Qdrant retriever initialized successfully")
        
        # Test retriever
        logger.info("Testing retriever with sample query...")
        test_query = "test query"
        test_result = retriever(test_query)
        logger.info(f"Test retriever result type: {type(test_result)}")
        logger.info(f"Test retriever result: {test_result}")
        
        return retriever
    except Exception as e:
        logger.error(f"Error initializing retriever: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    load_dotenv()
    
    # Ingest products
    # product_path = "products/products_all.json"
    # reviews_path = "products/Amazon-review-summary.txt"
    
    # client = ingest_products(product_path, reviews_path)
    client = initialize_qdrant()
    # Initialize retriever
    retriever = get_qdrant_retriever(client)
    
    
    # Configure dspy
    lm = dspy.LM(model='openai/gpt-4o-mini')
    dspy.configure(lm=lm, rm=retriever)



