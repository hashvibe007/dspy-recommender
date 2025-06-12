import dspy
from signatures import RecommendProducts
from customer_context import get_customer_history_context

class RAG(dspy.Module):
    def __init__(self, num_of_passages=2):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_of_passages)
        self.recommend = dspy.ChainOfThought(RecommendProducts)

    def forward(self, question, customer_id):
        context = self.retrieve(question).passages
        customer_details = get_customer_history_context(customer_id)
        
        prediction = self.recommend(
            context=context, 
            question=question, 
            customer_details=customer_details
        )
        
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