import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import dspy
from product_ingestion import get_product_retriever, get_or_create_chroma_client, get_or_create_collection
from rag_module import RAG
import logging
import requests

# Load environment variables
load_dotenv()
api_key_openai = os.getenv("OPENAI_API_KEY")
api_key_gemini = os.getenv("GEMINI_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# Global variables
rag = None

def initialize_rag_system():
    """Initialize the RAG system"""
    global rag
    
    # Get ChromaDB client and collection
    chroma_client = get_or_create_chroma_client()
    collection = get_or_create_collection(chroma_client)
    
    if collection.count() == 0:
        raise RuntimeError("ChromaDB collection is empty. Please run ingest_products.py first.")
    
    # Get retriever
    retriever = get_product_retriever(chroma_client)
    
    # Configure dspy
    lm = dspy.LM(model='openai/gpt-4o-mini')
    dspy.configure(lm=lm, rm=retriever)
    
    # Initialize RAG
    rag = RAG()
    return rag

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global rag
    rag = initialize_rag_system()

@app.post("/recommend")
async def recommend(request: Request):
    """Endpoint for product recommendations"""
    try:
        data = await request.json()
        logging.info(f"Received request: {data}")
        result = rag(question=data["question"], customer_id=data["customer_id"])

        # Fetch ChromaDB client and collection
        chroma_client = get_or_create_chroma_client()
        collection = get_or_create_collection(chroma_client)

        # Overwrite amazon_url in recommendations with value from metadata (prefer material_id if present)
        for rec in result.recommendations:
            # Try to fetch by material_id first if available
            meta = None
            if hasattr(rec, 'material_id') and rec.material_id:
                meta = collection.get(where={"material_id": rec.material_id}, include=["metadatas"])
                if meta and meta["metadatas"] and meta["metadatas"][0].get("amazon_url"):
                    rec.amazon_url = meta["metadatas"][0]["amazon_url"]
                    # Also update product_id if needed
                    if meta["metadatas"][0].get("product_id"):
                        rec.product_id = meta["metadatas"][0]["product_id"]
                    # Fetch current price from external API
                    try:
                        price_resp = requests.post(
                            'https://agentapi.ifbanalytics.com/fg-mrp',
                            headers={'Content-Type': 'application/json'},
                            json={"model_id": rec.product_id}
                        )
                        if price_resp.status_code == 200:
                            price_data = price_resp.json()
                            price_str = price_data.get("MRP", "").replace(",", "").replace(".00", "")
                            try:
                                rec.price = float(price_str)
                            except Exception:
                                rec.price = price_str
                        else:
                            rec.price = None
                    except Exception as ex:
                        rec.price = None
                    continue  # Found by material_id, skip product_id lookup
            # Fallback to product_id
            meta = collection.get(where={"product_id": rec.product_id}, include=["metadatas"])
            if meta and meta["metadatas"] and meta["metadatas"][0].get("amazon_url"):
                rec.amazon_url = meta["metadatas"][0]["amazon_url"]
            # Fetch current price from external API
            try:
                price_resp = requests.post(
                    'https://agentapi.ifbanalytics.com/fg-mrp',
                    headers={'Content-Type': 'application/json'},
                    json={"model_id": rec.product_id}
                )
                if price_resp.status_code == 200:
                    price_data = price_resp.json()
                    price_str = price_data.get("MRP", "").replace(",", "").replace(".00", "")
                    try:
                        rec.price = float(price_str)
                    except Exception:
                        rec.price = price_str
                else:
                    rec.price = None
            except Exception as ex:
                rec.price = None
        response = {
            "persona_name": result.persona_name,
            "description": result.description,
            "key_characteristics": result.key_characteristics,
            "purchase_patterns": {
                "frequency": result.frequency,
                "preferred_categories": result.preferred_categories,
                "price_sensitivity": result.price_sensitivity
            },
            "service_history": {
                "frequency": result.frequency,
                "maintenance_type": result.maintenance_type,
                "common_issues": result.common_issues
            },
            "preferences": {
                "price_range": result.price_range,
                "features": result.features
            },
            "recommendations": result.recommendations
        }
        logging.info(f"Response for customer_id={data['customer_id']}: {response}")
        # Log DSPy LLM history for debugging hallucinations
        try:
            history_str = str(dspy.inspect_history(n=5))
            logging.info(f"DSPy Inspect History (last 5):\n{history_str}")
        except Exception as ex:
            logging.error(f"Error logging DSPy inspect_history: {ex}")
        return response
    except Exception as e:
        logging.error(f"Error for request {data if 'data' in locals() else ''}: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=1124)
    # For testing
    rag = initialize_rag_system()
    result = rag(question="suggest a washing machine under 30000 with AI features", customer_id="0023784201")