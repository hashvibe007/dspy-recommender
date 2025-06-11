import dspy
import dspy.predict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import ujson
import os 
from dotenv import load_dotenv
import time
from dspy.retrieve.chromadb_rm import ChromadbRM
import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel
import uvicorn
from customer_context import get_customer_history_context
from typing import Literal, List
from functools import lru_cache

load_dotenv()

api_key_openai = os.getenv("OPENAI_API_KEY")

api_key_gemini = os.getenv("GEMINI_API_KEY")


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from tqdm import tqdm

directory = os.path.dirname(os.path.abspath(__file__))
att_list = []

product_path = directory + '/products/products_all.json'

# json_data = json.load(open(product_path))

# print(json_data)

max_characters = 2000
topk_docs_to_retrieve = 2

def flatten_specifications(specs):
    flat_parts = []
    for section, entries in specs.items():
        flat_parts.append(f"--- {section} ---")
        for key, val in entries.items():
            flat_parts.append(f"{key}: {val}")
    return "\n".join(flat_parts)

# 🧩 Load and transform JSON into flat docs
with open(product_path) as f:
    data = ujson.load(f)["products"]

corpus = []
for product_id, product in tqdm(data.items()):
    lines = [f"Model: {product.get('model_name', '')}",
             f"Category: {product.get('category', '')}"]

    if "basic_info" in product:
        for key, val in product["basic_info"].items():
            lines.append(f"{key.capitalize()}: {val}")

    if "specifications" in product:
        lines.append(flatten_specifications(product["specifications"]))

    corpus_text = "\n".join(lines).strip()[:max_characters]
    # print(corpus_text)
    corpus.append(corpus_text)

print(f"Loaded {len(corpus)} products into corpus.")

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="products",embedding_function=embedding_functions.DefaultEmbeddingFunction())

# ad corpus to chroma
collection.add(
    documents=corpus,
    ids=[str(i) for i in range(len(corpus))]
)

retriever_model = ChromadbRM(
    client=chroma_client,
    collection_name="products",
    persist_directory="chroma_db",
    embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    k=topk_docs_to_retrieve
)

lm = dspy.LM(model='openai/gpt-4o-mini')

dspy.configure(lm=lm,rm=retriever_model)

class ProductRecommendation(dspy.Signature):
    product_id: str = dspy.InputField(desc="it is the product id of the product e.g 8903287021718 it's generally 13 digits")
    model_name: str = dspy.InputField(desc="it is actual model name of the product e.g TL - RGS 7 kg Aqua")
    category: Literal["Washing Machine","Refrigerator","Air Conditioner","Dishwasher","Microwave"] = dspy.InputField(desc="Category")
    recommendation_type: Literal["Up-sell","Cross-sell"] = dspy.InputField(desc="Recommendation type")
    price: float = dspy.InputField(desc="Price of the product")
    match_score: float = dspy.InputField(desc="Probability score of the recommendation between 0 and 1")
    reasons: str = dspy.InputField(desc="Reasons for the recommendation in a single sentence")
    key_features: list[str] = dspy.InputField(desc="Key features of the products in a list")
    
class recommend_products(dspy.Signature):
    question: str = dspy.InputField(desc="Question to recommend products")
    context: str = dspy.InputField(desc="IFB Products features and details")
    customer_details: str = dspy.InputField(desc="Customer details")
    
    persona_name: Literal["Loyalist Platinum","Value Conscious Gold","Cautious Bronze","Disengaged Iron"] = dspy.OutputField(desc="Customer persona based on the customer details")
    description: str = dspy.OutputField(desc="Description of the recommended approach considering customer details and products")
    key_characteristics: list[str] = dspy.OutputField(desc="Key characteristics of the recommended products")
    frequency: str = dspy.OutputField(desc="Frequency of the customer buying patterns")
    preferred_categories: list[Literal["Washing Machine","Refrigerator","Air Conditioner","Dishwasher","Microwave"]] = dspy.OutputField(desc="Preferred categories of the customer can be multiple")
    price_sensitivity: Literal["high","medium","low"] = dspy.OutputField(desc="Price sensitivity of the customer")
    maintenance_type: Literal["proactive","reactive"] = dspy.OutputField(desc="Maintenance type of the customer")
    common_issues: list[str] = dspy.OutputField(desc="Common issues of the customer")
    price_range: Literal["budget","mid-range","premium"] = dspy.OutputField(desc="Price range of the customer")
    features: list[str] = dspy.OutputField(desc="Features of the recommended products")
    recommendations: List[ProductRecommendation] = dspy.OutputField(desc="Top 3 product recommendations based on customer persona and context")

class RAG(dspy.Module):
    def __init__(self,num_of_passages=2):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_of_passages)
        self.recommend = dspy.ChainOfThought(recommend_products)

    def forward(self, question, customer_id):
        context = self.retrieve(question).passages
        customer_details = get_customer_details(customer_id)
        # print("Customer Details Type:", type(customer_details))
        # print("Customer Details:", customer_details)
        # owned_categories = extract_owned_categories(customer_details)
        # print("Owned Categories:", owned_categories)
        
        prediction = self.recommend(
            context=context, 
            question=question, 
            customer_details=customer_details
        )
        
        
        
        
        # Calculate recommendations
        # recommendations = []
        # for product in prediction.recommendations:
        #     score, reasons = calculate_match_score(
        #         product_features=product['features'],
        #         customer_preferences={
        #             'features': prediction.features,
        #             'price_range': prediction.price_range,
        #             'preferred_categories': prediction.preferred_categories
        #         },
        #         price=product['price'],
        #         category=product['category']
        #     )
            
        #     rec_type = determine_rec_type(product['category'], owned_categories)
            
        #     recommendations.append({
        #         'product_id': product['id'],
        #         'model_name': product['model_name'],
        #         'category': product['category'],
        #         'recommendation_type': rec_type,
        #         'price': product['price'],
        #         'match_score': score,
        #         'reasons': reasons,
        #         'key_features': product['features']
        #     })
        
        # Sort by match score and get top 3
        # recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        # top_recommendations = recommendations[:3]
        
        return dspy.Prediction(
            persona_name=prediction.persona_name,
            description=prediction.description,
            key_characteristics=prediction.key_characteristics,
            frequency=prediction.frequency,
            preferred_categories=prediction.preferred_categories,
            price_sensitivity=prediction.price_sensitivity,
            maintenance_type=prediction.maintenance_type,
            common_issues=prediction.common_issues,
            price_range=prediction.price_range,
            features=prediction.features,
            recommendations=prediction.recommendations
        )

  

rag = RAG()

def get_customer_details(customer_id):
    return get_customer_history_context(customer_id)

@lru_cache(maxsize=1000)
@app.post("/recommend")
async def recommend(request: Request):
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
    # uvicorn.run(app, host="0.0.0.0", port=1124)
    result = rag(question="suggest a washing machine under 30000 with AI features", customer_id="0023784201")
    print(result)