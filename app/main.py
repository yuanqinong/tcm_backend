#uvicorn app.main:app --reload
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api import dashboard, chatbot

tags_metadata = [
    {
        "name": "dashboard",
        "description": "Operations with TCM dashboard",
    },
    {
        "name": "chatbot",
        "description": "Operations with chatbot",
    },
]

app = FastAPI(openapi_tags=tags_metadata)

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

@app.get("/")
async def default():
    return {"message": "Welcome to the TCM Knowledge Base"}

# Include the dashboard router
app.include_router(dashboard.router, prefix="/api/dashboard")

# Include the chatbot router
app.include_router(chatbot.router, prefix="/api/chatbot")

# Other app configurations and routers...
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)