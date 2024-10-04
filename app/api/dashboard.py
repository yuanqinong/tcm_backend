from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from bson import ObjectId
from datetime import datetime
from fastapi.responses import StreamingResponse
import gridfs
import zipfile
import io
from bson.errors import InvalidId
from app.utils.logger import logger
from tools.vector_embeddings import VectorEmbeddingsProcessor
from tools.mongodb_loader import MongoDBLangChainLoader
from app.core.database import db, fs, MONGO_URL, db_name


router = APIRouter()

@router.post("/upload", tags=["dashboard"])
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_files = []
    for file in files:
        contents = await file.read()
        
        file_id = fs.put(
            contents, 
            filename=file.filename, 
            uploadDate=datetime.now(),
            content_type=file.content_type,
            Synced=False
        )
        uploaded_files.append({"filename": file.filename, "file_id": str(file_id)})
    return {"message": f"{len(uploaded_files)} file(s) uploaded successfully", "files": uploaded_files}

@router.post("/sync_knowledge_base", tags=["dashboard"])
async def sync_knowledge_base():
    mongo_loader = MongoDBLangChainLoader(MONGO_URL, db_name)
    await mongo_loader.connect()
    documents = await mongo_loader.load_unprocessed_documents()
    document_ids = [doc["file_id"] for doc in documents]
    if not documents:
        return {"message": "All documents are synced"}
    logger.info(f"Total {len(documents)} unprocessed files found and processing...")
    try:    
        vector_embeddings_processor = VectorEmbeddingsProcessor()
         # Use the thread pool for CPU-bound task
        #loop = asyncio.get_event_loop()
        #await loop.run_in_executor(thread_pool, vector_embeddings_processor.index_doc_to_vector)
        await vector_embeddings_processor.index_doc_to_vector(documents)
        await mongo_loader.mark_documents_as_synced(document_ids)
        await mongo_loader.close()
        return {"message": f"{len(documents)} unprocessed files Synced"}
    except Exception as e:
        logger.error(f"Error in sync_knowledge_base: {str(e)}")
        return {"error": str(e)}

@router.post("/download_files", tags=["dashboard"])
async def download_files(file_ids: List[str]):
    logger.info(f"Attempting to download files with IDs: {file_ids}")
    try:
        # Validate all file IDs
        for file_id in file_ids:
            if not ObjectId.is_valid(file_id):
                logger.warning(f"Invalid ObjectId: {file_id}")
                raise HTTPException(status_code=400, detail=f"Invalid file ID format: {file_id}")

        if len(file_ids) == 1:
            # Single file download
            file_id = file_ids[0]
            file = await db.fs.files.find_one({"_id": ObjectId(file_id)})
            if not file:
                logger.warning(f"File metadata not found for ID: {file_id}")
                raise HTTPException(status_code=404, detail="File not found")

            try:
                grid_out = fs.get(ObjectId(file_id))
            except gridfs.errors.NoFile:
                logger.error(f"File content not found in GridFS for ID: {file_id}")
                raise HTTPException(status_code=404, detail="File content not found")

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
                    file = await db.fs.files.find_one({"_id": ObjectId(file_id)})
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
                    zip_file.writestr(zip_filename, grid_out.read())
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
        raise HTTPException(status_code=400, detail="Invalid file ID format")
    except Exception as e:
        logger.error(f"Unexpected error downloading files: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
    
@router.get("/files", tags=["dashboard"])
async def list_files():
    files = await db.fs.files.find().to_list(length=None)
    return [{"filename": file["filename"], "file_id": str(file["_id"]), "upload_date": file["uploadDate"], "content_type": file["contentType"], "Synced": file["Synced"]} for file in files]

@router.get("/get_count_unprocessed_files", tags=["dashboard"])
async def get_count_unprocessed_files():
    mongo_loader = MongoDBLangChainLoader(MONGO_URL, db_name)
    await mongo_loader.connect()
    count = await mongo_loader.get_count_unprocessed_documents()
    await mongo_loader.close()
    return {"count": count}

@router.delete("/delete_file", tags=["dashboard"])
async def delete_files(file_ids: List[str]):
    deleted_files = []
    errors = []

    for file_id in file_ids:
        try:
            # Check if the file exists
            file = await db.fs.files.find_one({"_id": ObjectId(file_id)})
            if not file:
                errors.append(f"File {file_id} not found")
                continue

            # Delete the file from GridFS
            fs.delete(ObjectId(file_id))

            # Delete the file metadata from fs.files collection
            await db.fs.files.delete_one({"_id": ObjectId(file_id)})

            # Delete any chunks associated with the file
            await db.fs.chunks.delete_many({"files_id": ObjectId(file_id)})

            deleted_files.append(file_id)
        except Exception as e:
            errors.append(f"Error deleting file {file_id}: {str(e)}")

    return {
        "message": f"{len(deleted_files)} file(s) deleted successfully",
        "deleted_files": deleted_files,
        "errors": errors
    }

@router.delete("/delete_embeddings", tags=["dashboard"])
async def delete_embeddings(file_ids: List[str]):
    vector_embeddings_processor = VectorEmbeddingsProcessor()
    result = await vector_embeddings_processor.delete_embeddings(file_ids)
    return result



