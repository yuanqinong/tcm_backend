import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import gridfs

MONGO_URL = os.getenv("MONGO_URL")
db_name = os.getenv("DB_NAME")

client = AsyncIOMotorClient(MONGO_URL)
db = client[db_name]

# We need a synchronous client for GridFS
sync_client = MongoClient(MONGO_URL)
sync_db = sync_client[db_name]
fs = gridfs.GridFS(sync_db)
