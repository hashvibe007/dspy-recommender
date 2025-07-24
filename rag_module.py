import dspy
from signatures import RecommendProducts
from customer_context import get_customer_history_context
import logging

class RAG(dspy.Module):
    def __init__(self, num_of_passages=2):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_of_passages)
        self.recommend = dspy.ChainOfThought(RecommendProducts)
        

    def forward(self, question, customer_id):
        logging.info(f"RAG: Calling retriever for question: {question}")
        context_obj = self.retrieve(question)
        context = context_obj.passages
        logging.info(f"RAG: Retriever returned {len(context)} passages")
        logging.info(context)
        customer_details = get_customer_history_context(customer_id)
        logging.info(f"RAG: Customer details fetched for customer_id={customer_id}")
        logging.info(customer_details)
        
        # Check for cache (if DSPy supports it)
        if hasattr(context_obj, 'from_cache') and context_obj.from_cache:
            logging.info("RAG: Retriever result was fetched from cache.")
        
        logging.info("RAG: Calling recommend chain of thought...")
        prediction = self.recommend(
            context=context, 
            question=question, 
            customer_details=customer_details
        )
        logging.info(f"RAG: Recommend chain returned prediction: {prediction}")
        logging.info(dspy.inspect_history(n=5))
        
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