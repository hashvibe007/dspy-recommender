import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import dspy
from product_ingestion import get_product_retriever, get_or_create_chroma_client, get_or_create_collection
from rag_module import RAG

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
        result = rag(question=data["question"], customer_id=data["customer_id"])
        return {
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
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=1124)
    # For testing
    rag = initialize_rag_system()
    result = rag(question="suggest a washing machine under 30000 with AI features", customer_id="0023784201")