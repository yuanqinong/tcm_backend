import asyncio
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
import psycopg
from typing import List, Dict, Any
from app.utils.logger import logger
from dotenv import load_dotenv
import os
load_dotenv()
connection_params = {
    "user": os.getenv("PGVECTOR_USER"),
    "password": os.getenv("PGVECTOR_PASSWORD"),
    "host": os.getenv("PGVECTOR_HOST"),
    "port": os.getenv("PGVECTOR_PORT"),
    "database": os.getenv("PGVECTOR_DB_NAME")
}
class VectorEmbeddingsProcessor:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", llm_model: str = "llama3.1", temp_sync_path: str = "./temp_sync"):
        self.huggingface_embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.llm_model = llm_model
        self.temp_sync_path = temp_sync_path
        self.chain = None
    
    def split_text(self, pages: List[Dict[str, Any]], chunk_size: int = 1500, chunk_overlap: int = 100):
        logger.info("Splitting text into chunks")
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(pages)
        logger.info(f"Total {len(chunks)} chunks created")
        return chunks

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
            print(connection_string)
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
            raise  # Re-raise the exception to be handled by the calling function
    
    async def index_doc_to_vector(self, documents):
        try:
            pages = []
            for doc in documents:
                file_path = doc['local_path']
                object_id = doc['file_id']  # This is the ObjectId from MongoDB
                logger.info(f"Loading PDF: {file_path}")
                loader = PyPDFLoader(file_path)
                async for page in loader.alazy_load():
                    # Add the object_id to each page's metadata
                    page.metadata['object_id'] = object_id
                    pages.append(page)
            chunks = self.split_text(pages)
            self.store_to_vector(chunks)
            logger.info(f"Total {len(documents)} PDF files Synced")

            # Delete files in temp_sync folder
            for doc in documents:
                file_path = doc['local_path']
                try:
                    await asyncio.to_thread(os.remove, file_path)
                    logger.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")

            logger.info("Cleaned up temporary files in temp_sync folder")
            
        except Exception as e:
            logger.error(f"Error in index_doc_to_vector: {str(e)}")

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
        logger.info("Loading vector store and retriever")
        connection_params = {
            "user": os.getenv("PGVECTOR_USER"),
            "password": os.getenv("PGVECTOR_PASSWORD"),
            "host": os.getenv("PGVECTOR_HOST"),
            "port": os.getenv("PGVECTOR_PORT"),
            "database": os.getenv("PGVECTOR_DB_NAME")
        }
        connection_string = f"postgresql+psycopg://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}"
        vector_store = PGVector(
        embedding=self.huggingface_embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION_NAME"),
        connection=connection_string,
        use_jsonb=True,
        )
        QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}"""
        )
        llm = ChatOllama(model=self.llm_model, temperature=0)
        retriever = MultiQueryRetriever.from_llm(
            vector_store.as_retriever(),
            llm,
        )
        def format_docs(docs):  
            format_docs = "\n\n".join([d.page_content for d in docs])
            return format_docs
        parser = StrOutputParser()
        prompt = QUERY_PROMPT
        try:
            self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | parser
            )
            print("Chain setup completed successfully")
        except Exception as e:
            print(f"Error in setup_conversational_chain: {str(e)}")

    """
       async def index_pdf_and_create_chain(self):
        try:
            pdf_files = [f for f in os.listdir(self.temp_sync_path) if f.endswith('.pdf')]
            if not pdf_files:
                logger.warning("No PDF files found in the directory.")
            else:
                pages = []
                for pdf_file in pdf_files:
                    file_path = os.path.join(self.temp_sync_path, pdf_file)
                    logger.info(f"Loading PDF: {file_path}")
                    loader = PyPDFLoader(file_path)
                async for page in loader.alazy_load():
                    pages.append(page)

            chunks = await self.split_text(pages)
            _, retriever = self.create_vectorstore_and_retriever(chunks)
            self.setup_conversational_chain(retriever)
            logger.info(f"Total {len(pdf_files)} indexed and chain created")
            return self.chain
        except Exception as e:
            logger.error(f"Error in index_pdf_and_create_chain: {str(e)}")


    def chat_with_pdf(self, query: str) -> str:
        if not self.chain:
            raise ValueError("Chain not initialized. Call index_pdf_and_create_chain first.")
        handler = StdOutCallbackHandler()
        return self.chain.invoke({"question": query}, {"callbacks": [handler]}) 

            def create_vectorstore_and_retriever(self, chunks: List[Dict[str, Any]]):
        logger.info("Creating vector store and retriever")
        connection_params = {
            "user": os.getenv("PGVECTOR_USER"),
            "password": os.getenv("PGVECTOR_PASSWORD"),
            "host": os.getenv("PGVECTOR_HOST"),
            "port": os.getenv("PGVECTOR_PORT"),
            "database": os.getenv("PGVECTOR_DB_NAME")
        }
        QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template'''You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}'''
        )
        connection_string = f"postgresql+psycopg://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}"
        vector_store = PGVector.from_documents(
        documents=chunks,
        embeddings=self.huggingface_embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION_NAME"),
        connection=connection_string,
        use_jsonb=True,
        )

        llm = ChatOllama(model=self.llm_model, temperature=0)
        retriever = MultiQueryRetriever.from_llm(
            vector_store.as_retriever(),
            llm,
            prompt=QUERY_PROMPT
        )
        return vector_store, retriever

    def setup_conversational_chain(self, retriever):
        llm = ChatOllama(model=self.llm_model, temperature=50)
        parser = StrOutputParser()
        
        template = '''Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        '''

        prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):  
            format_docs = "\n\n".join([d.page_content for d in docs])
            return format_docs
        
        try:
            self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | parser
            )
            print("Chain setup completed successfully")
        except Exception as e:
            print(f"Error in setup_conversational_chain: {str(e)}")
    """ 

