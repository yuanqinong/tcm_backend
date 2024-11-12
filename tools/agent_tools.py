from langchain_core.tools import tool
from typing import Callable
from uuid import UUID
from tools.recommendation import get_all_products, get_customer_purchases, get_recommendations
from tools.vector_embeddings import VectorEmbeddingsProcessor
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.utils.prompt.AgentPrompt import SYSTEM
import json
from pydantic import BaseModel
from app.utils.logger import logger
import os

@tool
def rag_tool(query: str) -> str:
    """Answer a query or question using the knowledge base."""
    logger.info("Invoking rag_tool")
    vector_embeddings_processor = VectorEmbeddingsProcessor()
    rag_chain = vector_embeddings_processor.load_vectorstore_and_retriever()
    response = rag_chain.invoke(query)
    logger.info(f"RAG Response: {response}")
    return {"answer": response}

@tool
async def get_customer_recommendations():
    """Generate personalized recommendations for a customer with customer purchases history."""
    logger.info("Invoking get_recommendations tool")
    #Dummy customer uuid due to chatbot not get the uuid (should be integrate the Smooch ID)
    customer_uuid = UUID("84eff5a1-aabf-4781-b811-ea2848e402b5")
    purchase_history = get_customer_purchases(customer_uuid)
    product_list = get_all_products()
    recommendations = await get_recommendations(purchase_history, product_list)
    logger.debug(f"Recommendations: {recommendations}")
    return recommendations

tools = [rag_tool, get_customer_recommendations]
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
model = ChatOllama(model="llama3.1",temperature=0, base_url=OLLAMA_BASE_URL)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

async def invoke_agent(query: str):
    try:
        logger.info(f"Invoking agent with query: {query}")
        
        response = await agent_executor.ainvoke({
            "input": query,
        })
        return response
    except Exception as e:
        logger.error(f"Error in invoke_agent: {str(e)}")
        raise ValueError(f"Something went wrong. Please try again later.")
