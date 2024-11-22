from fastapi import FastAPI, Depends, HTTPException, status, APIRouter,Security, Header, Response, Cookie
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from fastapi.security import HTTPBearer
from app.utils import logger  # Assuming you have a logger utility
from typing import Optional
import uuid
from sqlalchemy.dialects.postgresql import UUID

router = APIRouter()
security = HTTPBearer()
load_dotenv()

# Database setup
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")
SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User model
class User(Base):
    __tablename__ = "admins"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

Base.metadata.create_all(bind=engine)

# Pydantic models
class UserCreate(BaseModel):
    username: str
    password: str

class LoginData(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Authentication settings
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
COOKIE_MAX_AGE = ACCESS_TOKEN_EXPIRE_MINUTES * 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    tz = timezone(timedelta(hours=8))
    expire = datetime.now(tz) + timedelta(minutes=int(ACCESS_TOKEN_EXPIRE_MINUTES))
    logger.info(f"Expire time (Malaysia/UTC+8): {expire}")
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    access_token: str = Cookie(None, alias="access_token")
) -> User:
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Remove 'Bearer ' prefix if present
    if access_token.startswith("Bearer "):
        access_token = access_token[7:]
    
    try:
        payload = jwt.decode(
            access_token, 
            SECRET_KEY, 
            algorithms=[ALGORITHM],
        )
        username: str = payload.get("username")
        user_id: str = payload.get("user_id")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
        
    # Get user from database
    db = SessionLocal()
    try:
        user = get_user(db, username=username)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    finally:
        db.close()

# API endpoints
@router.post("/signup/admins", response_model=Token, tags=["login/signup"])
async def signup(user: UserCreate, response: Response):
    db = SessionLocal()
    try:
        db_user = get_user(db, username=user.username)
        if db_user:
            raise HTTPException(status_code=400, detail="Username already registered")
        
        hashed_password = get_password_hash(user.password)
        new_user = User(
            id=uuid.uuid4(),
            username=user.username,
            hashed_password=hashed_password
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        access_token = create_access_token(data={"username": user.username, "user_id": str(new_user.id)})

        response.set_cookie(
            key="access_token",
            value=f"{access_token}",
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=COOKIE_MAX_AGE
        )
        
        return {"access_token": access_token, "token_type": "bearer"}
    finally:
        db.close()

@router.post("/login/admins", response_model=Token, tags=["login/signup"])
async def login(login_data: LoginData, response: Response):
    db = SessionLocal()
    try:
        # Authenticate user
        user = authenticate_user(db, login_data.username, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Create access token
        access_token = create_access_token(
            data={"username": user.username, "user_id": str(user.id)}
        )
        
        # Set cookie
        response.set_cookie(
            key="access_token",
            value=f"{access_token}",
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=COOKIE_MAX_AGE  # 1 hour in seconds
        )
        
        # Return token in response body as before
        return {"access_token": access_token, "token_type": "Bearer"}
        
    finally:
        db.close()

@router.post("/logout", tags=["login/signup"])
async def logout(response: Response):
    response.delete_cookie(
        key="access_token",
        httponly=True,
        secure=True,
        samesite="lax"
    )
    return {"message": "Successfully logged out"}

# The protected route can stay as it is
@router.get("/auth/verify", tags=["protected"])
async def protected_route(current_user: User = Depends(get_current_user)):
    return {"message": "You have access to this protected route", "username": current_user.username, "user_id": str(current_user.id)}
