import asyncio
import bs4
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_postgres import PGVector
from tools.mongodb_loader import MongoDBLangChainLoader
from langchain_postgres.vectorstores import PGVector
from app.utils.prompt.MultiQueryRetrieverPrompt import prompt as MultiQueryRetrieverPrompt
from app.utils.prompt.ChatPrompt import prompt as ChatPrompt
from app.core.database import MONGO_URL, DOC_DB_NAME, WEB_DB_NAME
from tools.ocr_service import OCRService
import psycopg
from typing import List, Dict, Any
from app.utils import logger
from dotenv import load_dotenv
from pathlib import Path
import os
import fitz
from datetime import datetime

load_dotenv()
connection_params = {
    "user": os.getenv("PGVECTOR_USER"),
    "password": os.getenv("PGVECTOR_PASSWORD"),
    "host": os.getenv("PGVECTOR_HOST"),
    "port": os.getenv("PGVECTOR_PORT"),
    "database": os.getenv("PGVECTOR_DB_NAME")
}
class VectorEmbeddingsProcessor:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", llm_model: str = "llama3.1", temp_images_path: str = r".\temp_images"):
        self.huggingface_embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.llm_model = llm_model
        self.temp_images_path = temp_images_path

        os.makedirs(self.temp_images_path, exist_ok=True)

        self.chain = None
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            'web': WebBaseLoader
        }
    
    def split_text(self, pages: List[Dict[str, Any]], chunk_size: int = 2000, chunk_overlap: int = 200):
        try:    
            logger.info("Splitting text into chunks")
            # Clean the text content of each page
            for page in pages:
                if hasattr(page, 'page_content'):
                    page.page_content = self.clean_text(page.page_content)
                
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]  # Define custom separators
            )
            chunks = splitter.split_documents(pages)
            logger.info(f"Total {len(chunks)} chunks created")
            return chunks
        except Exception as e:
            logger.error(f"Error in split_text: {str(e)}")
            raise ValueError(f"Something went wrong. Please try again later.")

    def store_to_vector(self, chunks: List[Dict[str, Any]]):
        connection_params = {
            "user": os.getenv("PGVECTOR_USER"),
            "password": os.getenv("PGVECTOR_PASSWORD"),
            "host": os.getenv("PGVECTOR_HOST"),
            "port": os.getenv("PGVECTOR_PORT"),
            "database": os.getenv("PGVECTOR_DB_NAME")
        }
        try:
            # Use asyncio.to_thread to run the potentially blocking PGVector operation in a separate thread
            connection_string = f"postgresql+psycopg://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}"
            vector_store = PGVector.from_documents(
                documents=chunks,
                embedding=self.huggingface_embeddings,
                collection_name=os.getenv("PGVECTOR_COLLECTION_NAME"),
                connection=connection_string,
                use_jsonb=True,
            )
            logger.info(f"Total {len(chunks)} chunks stored in vector database")
            return vector_store
        except Exception as e:
            logger.error(f"Error in store_to_vector: {str(e)}")
            raise ValueError(f"Something went wrong. Please try again later.")

    async def index_url_to_vector(self, links):
        url_mongo_loader = None
        try:
            pages = []
            processed_url_ids = []
            url_mongo_loader = MongoDBLangChainLoader(MONGO_URL, WEB_DB_NAME)
            await url_mongo_loader.connect()
            loader_class = self.loaders['web']
            for url in links:
                object_id = url['id']
                link = url['url']
                loader = loader_class([link])
                try:    
                    loaded_pages = await asyncio.to_thread(loader.load)
                    for page in loaded_pages:
                        page.metadata['object_id'] = object_id
                        pages.append(page)
                    processed_url_ids.append(object_id)
                except Exception as e:
                    logger.error(f"Error loading URL {link}: {str(e)}")
                    raise ValueError(f"Something went wrong. Please try again later.")
            if not pages:
                    raise ValueError(f"Something went wrong. Please try again later.")
            
            chunks = self.split_text(pages)
            try:
                if chunks:
                    self.store_to_vector(chunks)
                    logger.info(f"Successfully stored {len(chunks)} chunks in vector database")
                    
                if processed_url_ids:
                    updated_count = await url_mongo_loader.mark_urls_as_synced(processed_url_ids)
                    logger.info(f"Marked {updated_count} out of {len(processed_url_ids)} processed documents as synced")
            except Exception as e:
                logger.error(f"Error storing chunks to vector database: {str(e)}")

                raise ValueError(f"Something went wrong. Please try again later.")
        except Exception as e:
            logger.error(f"Error in index_url_to_vector: {str(e)}")
            raise ValueError(f"Something went wrong. Please try again later.")
        
        return {"message": f"{len(links)} links processed"}
    
    async def index_doc_to_vector(self, documents, enable_ocr: bool):
        mongo_loader = None
        modified_paths = {}  # Track any path changes during processing

        try:
            pages = []
            mongo_loader = MongoDBLangChainLoader(MONGO_URL, DOC_DB_NAME)
            await mongo_loader.connect()
            
            processed_file_ids = []
            for doc in documents:
                file_path = doc['local_path']
                object_id = doc['file_id']
                original_filename = os.path.basename(file_path)
                original_extension = os.path.splitext(file_path)[1].lower()
                
                logger.info(f"Processing document: {file_path} with object_id: {object_id}")
                file_extension = original_extension

                if file_extension not in self.loaders:
                    logger.warning(f"Unsupported file type: {file_extension}. Skipping {file_path}")
                    raise ValueError(f"Unsupported file type: {file_extension}. Skipping {file_path}")
                
                # Only perform OCR if enabled
                if enable_ocr:
                    ocr_service = OCRService()
                    if file_extension == '.pdf':
                        ocr_service.process_pdf_with_ocr(file_path, self.temp_images_path)
                    elif file_extension == '.docx':
                        new_file_path = ocr_service.process_docx_with_ocr(file_path, self.temp_images_path)
                        modified_paths[file_path] = new_file_path
                        file_path = new_file_path
                        file_extension = '.pdf'
                    
                loader_class = self.loaders[file_extension]
                logger.info(f"Loading file for chunking: {file_path}")
                loader = loader_class(file_path)
                try:
                    async for page in loader.alazy_load():
                        # Preserve original file information in metadata
                        page.metadata['original_filename'] = original_filename
                        page.metadata['original_extension'] = original_extension
                        page.metadata['object_id'] = object_id
                        # Optionally modify the source to show original path
                        if original_extension == '.docx':
                            page.metadata['source'] = page.metadata['source'].replace('.pdf', '.docx')
                        pages.append(page)
                    processed_file_ids.append(object_id)
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {str(e)}")
                    raise ValueError(f"Something went wrong. Please try again later.")

            if not pages:
                raise ValueError(f"Something went wrong. Please try again later.")

            chunks = self.split_text(pages)
            try:
                if chunks:
                    self.store_to_vector(chunks)
                    logger.info(f"Successfully stored {len(chunks)} chunks in vector database")
                    
                # Mark documents as synced only after successful storage
                if processed_file_ids:
                    updated_count = await mongo_loader.mark_documents_as_synced(processed_file_ids)
                    logger.info(f"Marked {updated_count} out of {len(processed_file_ids)} processed documents as synced")
            except Exception as e:
                logger.error(f"Error storing chunks to vector database: {str(e)}")
                raise ValueError(f"Something went wrong. Please try again later.")

        except Exception as e:
            logger.error(f"Error in index_doc_to_vector: {str(e)}")
            raise ValueError(f"Something went wrong. Please try again later.")

        finally:
            # Close MongoDB connection if it was opened
            if mongo_loader:
                await mongo_loader.close()
            
            # Delete files in temp_sync folder
            for doc in documents:
                original_path = doc['local_path']
                # Use the modified path if it exists, otherwise use the original path
                file_path = modified_paths.get(original_path, original_path)
                
                try:
                    if os.path.exists(file_path):
                        logger.info(f"Cleaning up file: {file_path}")
                        await asyncio.to_thread(os.remove, file_path)
                        logger.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")

                # Clean up temp images
                image_path = Path(self.temp_images_path)
                try:
                    logger.info(f"Cleaning up temporary directory: {image_path}")
                    for img in image_path.glob("*"):
                        try:
                            await asyncio.to_thread(os.remove, str(img))
                            logger.debug(f"Removed file: {img}")
                        except Exception as e:
                            logger.error(f"Error removing file {img}: {str(e)}")                         
                except Exception as e:
                    logger.error(f"Error in cleanup_temp_image_path: {str(e)}")
            logger.info("Cleaned up temporary files in temp_sync folder")

        return {"message": f"{len(documents)} files processed"}
 
    async def delete_embeddings(self, object_ids: List[str]):
        try:
            connection_params = {
            "user": os.getenv("PGVECTOR_USER"),
            "password": os.getenv("PGVECTOR_PASSWORD"),
            "host": os.getenv("PGVECTOR_HOST"),
            "port": os.getenv("PGVECTOR_PORT"),
            "database": os.getenv("PGVECTOR_DB_NAME")
            }
            connection_string = f"postgresql://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}"
            async with await psycopg.AsyncConnection.connect(connection_string) as aconn:
                async with aconn.cursor() as acur:
                    # Prepare the SQL query
                    query = """
                    DELETE FROM langchain_pg_embedding
                    WHERE cmetadata->>'object_id' = ANY(%s)
                    """

                    # Execute the query
                    await acur.execute(query, (object_ids,))

                    # Get the number of deleted rows
                    deleted_count = acur.rowcount

                # The transaction is automatically committed when the context is exited

            return {"message": f"Successfully deleted {deleted_count} embeddings"}

        except Exception as e:
            logger.error(f"Error deleting embeddings: {str(e)}")
            return {"error": str(e)}  
        
    def load_vectorstore_and_retriever(self):
        logger.info("Starting load_vectorstore_and_retriever")
        try:
            connection_params = {
                "user": os.getenv("PGVECTOR_USER"),
                "password": os.getenv("PGVECTOR_PASSWORD"),
                "host": os.getenv("PGVECTOR_HOST"),
                "port": os.getenv("PGVECTOR_PORT"),
                "database": os.getenv("PGVECTOR_DB_NAME")
            }
            connection_string = f"postgresql+psycopg://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}"
            logger.debug(f"Connection string created: {connection_string.replace(connection_params['password'], '******')}")

            logger.info("Initializing PGVector")
            vector_store = PGVector(
                embeddings=self.huggingface_embeddings,
                collection_name=os.getenv("PGVECTOR_COLLECTION_NAME"),
                connection=connection_string,
                use_jsonb=True,
            )
            logger.info("PGVector initialized successfully")

            logger.info("Setting up QUERY_PROMPT")
            QUERY_PROMPT = MultiQueryRetrieverPrompt
            logger.info(f"Initializing ChatOllama with model: {self.llm_model}")
            OLLAMA_HOST = os.getenv("OLLAMA_HOST")
            OLLAMA_PORT = os.getenv("OLLAMA_PORT")
            OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
    
            llm = ChatOllama(model=self.llm_model, temperature=0.3, base_url=OLLAMA_BASE_URL)

            logger.info("Setting up MultiQueryRetriever")
            retriever = MultiQueryRetriever.from_llm(
                vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
                llm,
                prompt=QUERY_PROMPT,
            )
            logger.info("MultiQueryRetriever set up successfully")

            def format_docs(docs):
                format_docs = "\n\n".join([d.page_content for d in docs])
                return format_docs

            logger.info("Initializing StrOutputParser")
            parser = StrOutputParser()

            logger.info("Setting up answer_prompt")
            answer_prompt = ChatPrompt

            logger.info("Setting up the chain")
            self.chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | answer_prompt
                | llm
                | parser
            )
            logger.info("Chain setup completed successfully")
            return self.chain

        except Exception as e:
            logger.error(f"Error in load_vectorstore_and_retriever: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean text by removing excessive newlines, spaces and Spire.Doc warnings."""
        
        # Replace multiple newlines with a single newline
        text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        # Replace multiple spaces with a single space
        text = ' '.join(text.split())
        text = text.replace("Evaluation Warning: The document was created with Spire.Doc for Python.", "")
        return text