import dspy
from typing import Literal, List

class ProductRecommendation(dspy.Signature):
    product_id: str = dspy.InputField(desc="it is the key of the product e.g 8903287021718")
    model_name: str = dspy.InputField(desc="it is actual model name of the product e.g TL - RGS 7 kg Aqua")
    category: Literal["Washing Machine","Refrigerator","Air Conditioner","Dishwasher","Microwave"] = dspy.InputField(desc="Category")
    recommendation_type: Literal["Up-sell","Cross-sell"] = dspy.InputField(desc="Recommendation type")
    price: float = dspy.InputField(desc="Price of the product")
    match_score: float = dspy.InputField(desc="Probability score of the recommendation between 0 and 1")
    reasons: str = dspy.InputField(desc="Reasons for the recommendation in a single sentence")
    key_features: list[str] = dspy.InputField(desc="Key features of the products in a list")
    amazon_review_summary: str = dspy.InputField(desc="Amazon review summary of the product if available")

class RecommendProducts(dspy.Signature):
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