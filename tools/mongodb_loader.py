from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from bson import ObjectId
import gridfs
from typing import List, Dict, Any
from app.utils import logger
from datetime import datetime
from langchain.schema import Document
import os

class MongoDBLangChainLoader:
    def __init__(self, mongo_url: str, db_name: str):
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.async_client = None
        self.sync_client = None
        self.db = None
        self.fs = None
        self.logger = logger

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

    async def get_count_unprocessed_documents(self) -> int:
        try:
            query = {"Synced": False}
            count = await self.db.fs.files.count_documents(query)
            return count
        except Exception as e:
            self.logger.error(f"Error getting count of unprocessed documents: {str(e)}")
            raise

    async def load_unprocessed_documents(self) -> List[Document]:
        try:
            query = {"Synced": False}
            files = self.db.fs.files.find(query)
            documents = []
            temp_dir = "./temp_sync"
            os.makedirs(temp_dir, exist_ok=True)
            
            async for file in files:
                grid_out = self.fs.get(file['_id'])
                temp_file_path = os.path.join(temp_dir, file['filename'])
                
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(grid_out.read())
                
                doc = {
                        "file_id": str(file['_id']),
                        "filename": file['filename'],
                        "content_type": file['contentType'],
                        "synced": file['Synced'],
                        "upload_date": file['uploadDate'],
                        "local_path": temp_file_path
                    }
                
                documents.append(doc)
            
            self.logger.info(f"Downloaded {len(documents)} unprocessed files to {temp_dir}")
            return documents
        
        except Exception as e:
            self.logger.error(f"Error loading unprocessed documents from gridfs: {str(e)}")
            raise

    async def mark_documents_as_synced(self, file_ids: List[str]):
        try:
            result = await self.db.fs.files.update_many(
                {"_id": {"$in": [ObjectId(file_id) for file_id in file_ids]}},
                {"$set": {"Synced": True}}
            )
            
            if result.modified_count > 0:
                self.logger.info(f"Marked {result.modified_count} document(s) as synced")
                if result.modified_count < len(file_ids):
                    self.logger.warning(f"{len(file_ids) - result.modified_count} document(s) were not found or already synced")
            else:
                self.logger.warning(f"No documents were marked as synced. All {len(file_ids)} document(s) were not found or already synced")
            
            return result.modified_count

        except Exception as e:
            self.logger.error(f"Error marking documents as synced: {str(e)}")
            raise

# Usage example in the next code block