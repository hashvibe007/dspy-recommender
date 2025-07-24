import dspy
from signatures import RecommendProducts, summariseCustomerHistory,EnhanceQuestion
from customer_context import get_customer_history_context
import logging

class RAG(dspy.Module):
    def __init__(self, num_of_passages=2):
        super().__init__()
        
        self.retrieve = dspy.Retrieve(k=num_of_passages)
        self.recommend = dspy.ChainOfThought(RecommendProducts)
        self.enhanced_question = dspy.ChainOfThought(EnhanceQuestion)
        self.customer_summary = dspy.ChainOfThought(summariseCustomerHistory)
        
        

    def forward(self, question, customer_id):
        logging.info(f"RAG: Calling retriever for question: {question}")
        enhanced_question = self.enhanced_question(question = question)
        logging.info(f"RAG: Enhanced question: {enhanced_question.enhanced_question}")
        new_question = enhanced_question.enhanced_question
        context_obj = self.retrieve(new_question)
        context = context_obj.passages
        
        customer_details = get_customer_history_context(customer_id)
        customer_summary = self.customer_summary(customer_message = customer_details)
        logging.info(f"RAG: Customer details fetched for customer_id={customer_id}")
        logging.info(customer_summary.summary)
        
        
        logging.info("RAG: Calling recommend chain of thought...")
        prediction = self.recommend(
            context=context, 
            question=new_question, 
            customer_details=customer_summary
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