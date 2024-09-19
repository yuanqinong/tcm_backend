# mongodb_langchain_loader.py

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from bson import ObjectId
import gridfs
from typing import List, Dict, Any
import logging
from datetime import datetime, UTC

class MongoDBLangChainLoader:
    def __init__(self, mongo_url: str, db_name: str):
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.async_client = None
        self.sync_client = None
        self.db = None
        self.fs = None
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        try:
            self.async_client = AsyncIOMotorClient(self.mongo_url)
            self.db = self.async_client[self.db_name]
            self.sync_client = MongoClient(self.mongo_url)
            sync_db = self.sync_client[self.db_name]
            self.fs = gridfs.GridFS(sync_db)
            self.logger.info(f"Connected to MongoDB: {self.db_name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    async def close(self):
        if self.async_client:
            self.async_client.close()
        if self.sync_client:
            self.sync_client.close()
        self.logger.info("Closed MongoDB connections")

    async def load_unprocessed_documents(self, collection_name: str) -> List[Dict[str, Any]]:
        try:
            collection = self.db[collection_name]
            query = {
                "$or": [
                    {"processed": {"$exists": False}},
                    {"processed": False},
                    {"last_updated": {"$gt": "$last_processed"}}
                ]
            }
            documents = await collection.find(query).to_list(length=None)
            return [self._serialize_document(doc) for doc in documents]
        except Exception as e:
            self.logger.error(f"Error loading unprocessed documents from {collection_name}: {str(e)}")
            raise

    async def mark_document_as_processed(self, collection_name: str, doc_id: str):
        try:
            collection = self.db[collection_name]
            await collection.update_one(
                {"_id": ObjectId(doc_id)},
                {
                    "$set": {
                        "processed": True,
                        "last_processed": datetime.now(UTC)
                    }
                }
            )
        except Exception as e:
            self.logger.error(f"Error marking document {doc_id} as processed in {collection_name}: {str(e)}")
            raise

    def _serialize_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        if doc is None:
            return None
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                doc[key] = str(value)
            elif isinstance(value, datetime):
                doc[key] = value.isoformat()
        return doc

# Usage example in the next code block