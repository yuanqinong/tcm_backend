#uvicorn app.main:app --reload
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from app.api import dashboard, chatbot, loginPageAdmin, recommendation

tags_metadata = [
    {
        "name": "dashboard",
        "description": "Operations with TCM dashboard",
    },
    {
        "name": "chatbot",
        "description": "Operations with chatbot",
    },
    {
        "name": "login/signup",
        "description": "Operations with login/signup",
    },
    {
        "name": "recommendation",
        "description": "Operations with recommendation",
    }
]

app = FastAPI(openapi_tags=tags_metadata)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="TCM API",
        version="1.0.0",
        description="API for TCM",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "Bearer Auth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    openapi_schema["security"] = [{"Bearer Auth": []}]
    
    # Apply security to all routes
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            openapi_schema["paths"][path][method]["security"] = [{"Bearer Auth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

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

app.include_router(loginPageAdmin.router, prefix="/api/loginPage")

app.include_router(recommendation.router, prefix="/api/recommendation")