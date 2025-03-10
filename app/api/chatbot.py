from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from tools.vector_embeddings import VectorEmbeddingsProcessor
from app.utils.logger import logger
from tools.agent_tools import invoke_agent
router = APIRouter()

class ChatRequest(BaseModel):
    query: str

"""
@router.post("/chat", tags=["chatbot"])
async def chat(chat_request: ChatRequest):
    vector_embeddings = VectorEmbeddingsProcessor()
    chain = vector_embeddings.load_vectorstore_and_retriever()
    
    try:
        response = chain.invoke(chat_request.query)
        print(response)
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error in sync_knowledge_base: {str(e)}")
        raise ValueError(f"Something went wrong. Please try again later.")
"""

@router.post("/chat", tags=["chatbot"])
async def chat(chat_request: ChatRequest, session_id: str):
    try:    
        response = await invoke_agent(chat_request.query, session_id)
        return response
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise ValueError(f"Something went wrong. Please try again later.")