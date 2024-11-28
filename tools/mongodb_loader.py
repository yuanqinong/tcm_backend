from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from bson import ObjectId
import gridfs
from typing import List, Dict, Any
from app.utils import logger
from datetime import datetime
from langchain.schema import Document
import os
from dotenv import load_dotenv
load_dotenv()   

class MongoDBLangChainLoader:
    def __init__(self, mongo_url: str, db_name: str, temp_sync_docs_path: str = r"./temp_sync_docs"):
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.async_client = None
        self.sync_client = None
        self.db = None
        self.fs = None
        self.logger = logger
        self.temp_sync_docs_path = temp_sync_docs_path

        os.makedirs(self.temp_sync_docs_path, exist_ok=True)

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
            raise ValueError(f"Something went wrong. Please try again later.")

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
            raise ValueError(f"Something went wrong. Please try again later.")

    async def load_unprocessed_documents(self) -> List[Document]:
        try:
            query = {"Synced": False}
            files = self.db.fs.files.find(query)
            documents = []
            
            async for file in files:
                grid_out = self.fs.get(file['_id'])
                temp_file_path = os.path.join(self.temp_sync_docs_path, file['filename'])
                
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
            
            self.logger.info(f"Downloaded {len(documents)} unprocessed files to {self.temp_sync_docs_path}")
            return documents
        
        except Exception as e:
            self.logger.error(f"Error loading unprocessed documents from gridfs: {str(e)}")
            raise ValueError(f"Something went wrong. Please try again later.")

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
            raise ValueError(f"Something went wrong. Please try again later.")
    
    async def load_unprocessed_urls(self) -> List[Dict[str, Any]]:
        WEB_COLLECTION_NAME = os.getenv("WEB_COLLECTION_NAME")
        try:
            query = {"Synced": False}
            projection = {"url": 1, "_id": 1}  # Retrieve both 'url' and '_id' fields
            cursor = self.db[WEB_COLLECTION_NAME].find(query, projection)
            url_data = [
                {"url": doc['url'], "id": str(doc['_id'])}
                async for doc in cursor
                if 'url' in doc and '_id' in doc
            ]
            
            self.logger.info(f"Retrieved {len(url_data)} unprocessed URLs with their IDs")
            return url_data
        except Exception as e:
            self.logger.error(f"Error loading unprocessed URLs: {str(e)}")
            raise ValueError(f"Something went wrong. Please try again later.")

    async def mark_urls_as_synced(self, url_ids: List[str]):
        WEB_COLLECTION_NAME = os.getenv("WEB_COLLECTION_NAME")
        try:
            result = await self.db[WEB_COLLECTION_NAME].update_many(
                {"_id": {"$in": [ObjectId(url_id) for url_id in url_ids]}},
                {"$set": {"Synced": True}}
            )
            
            if result.modified_count > 0:
                self.logger.info(f"Marked {result.modified_count} URL(s) as synced")
                if result.modified_count < len(url_ids):
                    self.logger.warning(f"{len(url_ids) - result.modified_count} URL(s) were not found or already synced")
            else:
                self.logger.warning(f"No URLs were marked as synced. All {len(url_ids)} URL(s) were not found or already synced")
            
            return result.modified_count

        except Exception as e:
            self.logger.error(f"Error marking URLs as synced: {str(e)}")
            raise ValueError(f"Something went wrong. Please try again later.")