from langchain.prompts import PromptTemplate
from app.utils.shared_models import output_parser

prompt_template = """
You are an advanced product recommendation system. Your task is to analyze a customer's purchase history and a full product list to suggest potential products they might be interested in. The customer's purchase history and the full product list.

Customer Purchase History:
{purchase_history}

Full Product List:
{product_list}

Based on this information, please follow these steps to generate personalized recommendations:

1. Analyze the customer's purchase history:
   - Identify patterns in product categories, prices, and purchase dates.
   - Note any trends or preferences in the customer's buying behavior.

2. Compare with the full product list:
   - Identify items similar to those the customer has bought before.
   - Look for complementary products that pair well with previous purchases.
   - Consider products in the same or related categories.

3. Generate recommendations:
   - Suggest 3-5 products that align with the customer's preferences and buying patterns.
   - Ensure a mix of recommendations: some similar to past purchases and some introducing new items they might like.
   - Consider the price range of previous purchases when making recommendations.

4. For each recommendation, provide:
   - name (name of the recommended product)
   - explanation (brief explanation of why this product might appeal to the customer)
   - price (price of the recommended product)
   - category (category of the recommended product)

Your task is to analyze this information and provide personalized product recommendations as described above.
"""

# Create the prompt
prompt_template = PromptTemplate(
    template=prompt_template,
    input_variables=["purchase_history", "product_list"],
)