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


@tool
def rag_tool(query: str) -> str:
    """Answer a query or question using the knowledge base."""
    vector_embeddings_processor = VectorEmbeddingsProcessor()
    rag_chain = vector_embeddings_processor.load_vectorstore_and_retriever()
    return rag_chain.invoke(query)

@tool
async def get_recommendations():
    """Get recommendations for a customer with customer purchases history."""
    customer_uuid = UUID("8a829bf4-d7e0-4ecd-afa6-feaf20c69ae5")
    purchase_history = get_all_products()
    product_list = get_customer_purchases(customer_uuid)
    recommendations = await get_recommendations(purchase_history, product_list)
    return recommendations

tools = [rag_tool, get_recommendations]
model = ChatOllama(model="llama3.1",temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def invoke_agent(query: str):
    try:
        # Format tools descriptions
        tool_descriptions = [f"{tool.name}: {tool.description}" for tool in tools]
        
        # Get tool names
        tool_names = [tool.name for tool in tools]
        
        response = agent_executor.invoke(
            {
                "input": query,
                "tools": "\n".join(tool_descriptions),
                "tool_names": ", ".join(tool_names),
            }
        )
        return response
    except Exception as e:
        logger.error(f"Error in invoke_agent: {str(e)}")
        raise ValueError(f"Something went wrong. Please try again later.")
