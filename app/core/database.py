import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv
load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DOC_DB_NAME = os.getenv("DOC_DB_NAME")
WEB_DB_NAME = os.getenv("WEB_DB_NAME")
WEB_COLLECTION_NAME = os.getenv("WEB_COLLECTION_NAME")
SYNC_STATUS_DB_NAME = os.getenv("SYNC_STATUS_DB_NAME", "sync_status_db")
SYNC_STATUS_COLLECTION_NAME = os.getenv("SYNC_STATUS_COLLECTION_NAME", "sync_status")

client = AsyncIOMotorClient(MONGO_URL)
doc_db = client[DOC_DB_NAME]
link_db = client[WEB_DB_NAME]

# Sync status for vector DB
sync_status_db = client[SYNC_STATUS_DB_NAME]
sync_status_collection = sync_status_db[SYNC_STATUS_COLLECTION_NAME]

# We need a synchronous client for GridFS
sync_client = MongoClient(MONGO_URL)
sync_doc_db = sync_client[DOC_DB_NAME]
fs = gridfs.GridFS(sync_doc_db)

# Collection for links
links_collection = link_db[WEB_COLLECTION_NAME]