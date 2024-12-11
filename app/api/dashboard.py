from fastapi import APIRouter, File, UploadFile, HTTPException, Depends,Header, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from typing import List, Optional
from bson import ObjectId
from datetime import datetime, timedelta, timezone
from fastapi.responses import StreamingResponse
import gridfs
import zipfile
import io
from bson.errors import InvalidId
from app.utils import logger
from tools.vector_embeddings import VectorEmbeddingsProcessor
from tools.mongodb_loader import MongoDBLangChainLoader
from app.core.database import doc_db, fs, MONGO_URL, DOC_DB_NAME, WEB_DB_NAME, links_collection, sync_status_collection
from app.api.loginPageAdmin import get_current_user,User
from pydantic import BaseModel

# Define UTC+8 offset
utc_offset = timedelta(hours=8)

router = APIRouter()
security = HTTPBearer()

class SyncRequest(BaseModel):
    enable_ocr: bool = False

@router.post("/upload_files", tags=["dashboard"])
async def upload_files(files: List[UploadFile] = File(...), current_user: User = Depends(get_current_user)):
    uploaded_files = []
    try:
        for file in files:
            # Use UTC time and add 8 hours to get UTC+8
            date_time = datetime.now(timezone.utc) + utc_offset
            contents = await file.read()
            
            file_id = fs.put(
                contents, 
                filename=file.filename, 
                uploadDate=date_time,
                content_type=file.content_type,
                Synced=False
                )
            uploaded_files.append({"filename": file.filename, "file_id": str(file_id)})
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        raise ValueError(f"Something went wrong. Please try again later.")
    return {"message": f"{len(uploaded_files)} file(s) uploaded successfully", "files": uploaded_files}

@router.post("/upload_links", tags=["dashboard"])
async def upload_links(links: List[str],current_user: User = Depends(get_current_user)):
    try:
        uploaded_links = []
        for link in links:
            # Use UTC time and add 8 hours to get UTC+8
            date_time = datetime.now(timezone.utc) + utc_offset
            link_doc = {
                "url": link,
                "uploadDate": date_time,
                "Synced": False
            }
            result = await links_collection.insert_one(link_doc)
            uploaded_links.append({
                "url": link,
                "link_id": str(result.inserted_id)
            })
        
        logger.info(f"Uploaded {len(uploaded_links)} links to TCM_Link database")
        return {
            "message": f"{len(uploaded_links)} link(s) uploaded successfully",
            "links": uploaded_links
        }
    except Exception as e:
        logger.error(f"Error uploading links: {str(e)}")
        raise ValueError(f"Something went wrong. Please try again later.")

@router.post("/sync_knowledge_base", tags=["dashboard"])
async def sync_knowledge_base(
    sync_options: SyncRequest,
    current_user: User = Depends(get_current_user)
):
    logger.info(f"Starting knowledge base sync with OCR {'enabled' if sync_options.enable_ocr else 'disabled'}")
    docs_mongo_loader = MongoDBLangChainLoader(MONGO_URL, DOC_DB_NAME)
    await docs_mongo_loader.connect()
    documents = await docs_mongo_loader.load_unprocessed_documents()
    urls_mongo_loader = MongoDBLangChainLoader(MONGO_URL, WEB_DB_NAME)  
    await urls_mongo_loader.connect()
    links = await urls_mongo_loader.load_unprocessed_urls()
    if not documents and not links:
        return {"message": "All documents and links are synced"}
    if documents:
        logger.info(f"Total {len(documents)} unprocessed files found and processing...")
        try:    
            vector_embeddings_processor = VectorEmbeddingsProcessor()
            await vector_embeddings_processor.index_doc_to_vector(documents, enable_ocr=sync_options.enable_ocr,username=current_user.username)
        except Exception as e:
            logger.error(f"Error in sync_knowledge_base: {str(e)}")
            raise ValueError(f"Something went wrong. Please try again later.")
    if links:
        logger.info(f"Total {len(links)} unprocessed links found and processing...")
        try:
            vector_embeddings_processor = VectorEmbeddingsProcessor()
            await vector_embeddings_processor.index_url_to_vector(links)
            
        except Exception as e:
            logger.error(f"Error in sync_knowledge_base: {str(e)}")
            raise ValueError(f"Something went wrong. Please try again later.")
        
    return {"message": "Sync completed successfully"}
@router.post("/download_files", tags=["dashboard"])
async def download_files(file_ids: List[str],current_user: User = Depends(get_current_user)):
    logger.info(f"Attempting to download files with IDs: {file_ids}")
    try:
        # Validate all file IDs
        for file_id in file_ids:
            if not ObjectId.is_valid(file_id):
                logger.warning(f"Invalid ObjectId: {file_id}")
                raise ValueError(f"Something went wrong. Please try again later.")

        if len(file_ids) == 1:
            # Single file download
            file_id = file_ids[0]
            file = await doc_db.fs.files.find_one({"_id": ObjectId(file_id)})
            if not file:
                logger.warning(f"File metadata not found for ID: {file_id}")
                raise ValueError(f"Something went wrong. Please try again later.")

            try:
                grid_out = fs.get(ObjectId(file_id))
            except gridfs.errors.NoFile:
                logger.error(f"File content not found in GridFS for ID: {file_id}")
                raise ValueError(f"Something went wrong. Please try again later.")

            logger.info(f"Returning single file: {file['filename']}")
            return StreamingResponse(
                grid_out,
                media_type=file['contentType'],
                headers={"Content-Disposition": f"attachment; filename={file['filename']}"}
            )

        else:
            # Multiple files download
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_id in file_ids:
                    file = await doc_db.fs.files.find_one({"_id": ObjectId(file_id)})
                    if not file:
                        logger.warning(f"File metadata not found for ID: {file_id}")
                        continue  # Skip this file and continue with others

                    try:
                        grid_out = fs.get(ObjectId(file_id))
                    except gridfs.errors.NoFile:
                        logger.error(f"File content not found in GridFS for ID: {file_id}")
                        continue  # Skip this file and continue with others
                    zip_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Add file to ZIP
                    zip_file.writestr(file['filename'], grid_out.read())
                    logger.info(f"Added file to ZIP: {zip_filename}")

            # Prepare the ZIP file for streaming
            zip_buffer.seek(0)
            
            logger.info("Returning ZIP file with multiple files")
            return StreamingResponse(
                iter([zip_buffer.getvalue()]),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={zip_filename}.zip"}
            )

    except InvalidId as e:
        logger.error(f"Invalid ObjectId format: {str(e)}")
        raise ValueError(f"Something went wrong. Please try again later.")
    except Exception as e:
        logger.error(f"Unexpected error downloading files: {str(e)}")
        raise ValueError(f"Something went wrong. Please try again later.")

@router.post("/get_url_with_id", tags=["dashboard"])
async def get_url_with_id(ids: List[str],current_user: User = Depends(get_current_user)):
    print(ids)
    url_with_id = []
    for id in ids:
        link_doc = await links_collection.find_one({"_id": ObjectId(id)})
        print(link_doc)
        if link_doc:
            url_with_id.append(link_doc["url"])
        
    return url_with_id

@router.get("/files", tags=["dashboard"])
async def list_files(current_user: User = Depends(get_current_user)):
    logger.info(f"Current user: {current_user.username}")
    try:
        files = await doc_db.fs.files.find().to_list(length=None)
        return [{"filename": file["filename"], "id": str(file["_id"]), "upload_date": file["uploadDate"], "content_type": file["contentType"], "Synced": file["Synced"]} for file in files]
    except Exception as e:
        logger.error(f"Error retrieving files: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving files")
    
@router.get("/links", tags=["dashboard"])
async def get_links(current_user: User = Depends(get_current_user)):
    try:
        links = await links_collection.find().to_list(length=None)
        formatted_links = []
        for link in links:
            formatted_links.append({
                "id": str(link["_id"]),
                "url": link["url"],
                "upload_date": link["uploadDate"],
                "Synced": link["Synced"]
            })
        
        logger.info(f"Retrieved {len(formatted_links)} links from TCM_Link database")
        return formatted_links
        
    except Exception as e:
        logger.error(f"Error retrieving links: {str(e)}")
        raise ValueError(f"Something went wrong. Please try again later.")
    
@router.get("/get_count_unprocessed_files", tags=["dashboard"])
async def get_count_unprocessed_files(current_user: User = Depends(get_current_user)):
    try:
        mongo_loader = MongoDBLangChainLoader(MONGO_URL, DOC_DB_NAME)
        await mongo_loader.connect()
        count = await mongo_loader.get_count_unprocessed_documents()
        await mongo_loader.close()
        return {"count": count}
    except Exception as e:
        logger.error(f"Error retrieving count of unprocessed files: {str(e)}")
        raise ValueError(f"Something went wrong. Please try again later.")

@router.delete("/delete_file", tags=["dashboard"])
async def delete_files(file_ids: List[str],current_user: User = Depends(get_current_user)):
    deleted_files = []
    errors = []

    for file_id in file_ids:
        try:
            # Check if the file exists
            file = await doc_db.fs.files.find_one({"_id": ObjectId(file_id)})
            if not file:
                errors.append(f"File {file_id} not found")
                continue

            # Delete the file from GridFS
            fs.delete(ObjectId(file_id))

            # Delete the file metadata from fs.files collection
            await doc_db.fs.files.delete_one({"_id": ObjectId(file_id)})

            # Delete any chunks associated with the file
            await doc_db.fs.chunks.delete_many({"files_id": ObjectId(file_id)})

            deleted_files.append(file_id)
        except Exception as e:
            errors.append(f"Error deleting file {file_id}: {str(e)}")

    return {
        "message": f"{len(deleted_files)} file(s) deleted successfully",
        "deleted_files": deleted_files,
        "errors": errors
    }

@router.delete("/delete_embeddings", tags=["dashboard"])
async def delete_embeddings(ids: List[str],current_user: User = Depends(get_current_user)):
    try:
        vector_embeddings_processor = VectorEmbeddingsProcessor()
        result = await vector_embeddings_processor.delete_embeddings(ids)
        return result
    except Exception as e:
        logger.error(f"Error deleting embeddings: {str(e)}")
        raise ValueError(f"Something went wrong. Please try again later.")

@router.delete("/delete_links", tags=["dashboard"])
async def delete_links(link_ids: List[str],current_user: User = Depends(get_current_user)):
    deleted_links = []
    errors = []

    for link_id in link_ids:
        try:
            # Check if the link exists
            link = await links_collection.find_one({"_id": ObjectId(link_id)})
            if not link:
                errors.append(f"Link {link_id} not found")
                continue

            # Delete the link from the links collection 
            await links_collection.delete_one({"_id": ObjectId(link_id)})

            deleted_links.append(link_id)
        except Exception as e:
            errors.append(f"Error deleting link {link_id}: {str(e)}")

    return {    
        "message": f"{len(deleted_links)} link(s) deleted successfully",
        "deleted_links": deleted_links,
        "errors": errors
    }

@router.get("/active_syncs", tags=["dashboard"])
async def get_active_syncs(current_user: User = Depends(get_current_user)):
    """Get all currently active sync sessions"""
    try:
        # Find sessions that are in progress (created or processing)
        cursor = sync_status_collection.find({
            "status": {"$in": ["created", "processing"]}
        }).sort("start_time", -1)
        
        active_sessions = await cursor.to_list(length=None)
        
        # Convert ObjectId to string for JSON serialization
        for session in active_sessions:
            session["_id"] = str(session["_id"])
            # Remove current_file field as it's redundant
            if "current_file" in session:
                del session["current_file"]
            
        return active_sessions
    except Exception as e:
        logger.error(f"Error retrieving active syncs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving active syncs")
    
@router.get("/latest_sync_status", tags=["dashboard"])
async def get_latest_sync_status(current_user: User = Depends(get_current_user)):
    """Get the latest sync session across all statuses (created, processing, completed)
    
    Returns the single most recent session based on start_time
    """
    try:
        # Query to get the latest document across all relevant statuses
        cursor = sync_status_collection.find(
            {"status": {"$in": ["created", "processing", "completed"]}}
        ).sort("start_time", -1).limit(1)
        
        latest_session = await cursor.to_list(length=1)
        
        if not latest_session:
            return None
            
        # Convert ObjectId to string for JSON serialization
        latest_session[0]["_id"] = str(latest_session[0]["_id"])
        return latest_session[0]
        
    except Exception as e:
        logger.error(f"Error retrieving latest sync status: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Error retrieving latest sync status"
        )