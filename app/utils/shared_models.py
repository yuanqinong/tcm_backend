from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class RecommendedProduct(BaseModel):
    name: str = Field(description="The name of the recommended product")
    explanation: str = Field(description="A brief explanation of why this product might appeal to the customer")
    price: float = Field(description="The price of the product")
    category: str = Field(description="The category of the product")

class Recommendations(BaseModel):
    recommendations: List[RecommendedProduct] = Field(description="List of recommended products")

output_parser = PydanticOutputParser(pydantic_object=Recommendations)