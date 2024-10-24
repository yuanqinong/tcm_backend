import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from app.utils.prompt.RecommendationPrompt import prompt
from sqlalchemy.exc import SQLAlchemyError
from typing import List
from langchain.schema import StrOutputParser
import os
from langchain_ollama import ChatOllama
from app.utils.logger import logger
from app.utils.shared_models import output_parser
from sqlalchemy.orm import class_mapper
import json
from pydantic import ValidationError

load_dotenv()

DATABASE_URL = os.getenv("USERDB_CONNECTION")

# Create the engine
engine = sa.create_engine(DATABASE_URL)

# Create a base class for declarative models
Base = declarative_base()

# Define your models
class Customer(Base):
    __tablename__ = 'customers'
    uuid = sa.Column(sa.UUID, primary_key=True)
    username = sa.Column(sa.String, unique=True, nullable=False)

class Product(Base):
    __tablename__ = 'products'
    id = sa.Column(sa.Integer, primary_key=True)
    category = sa.Column(sa.String, nullable=False)
    name = sa.Column(sa.String, nullable=False)
    price = sa.Column(sa.Numeric(10, 2), nullable=False)

class PurchaseHistory(Base):
    __tablename__ = 'purchase_history'
    id = sa.Column(sa.Integer, primary_key=True)
    customer_uuid = sa.Column(sa.UUID, sa.ForeignKey('customers.uuid'), nullable=False)
    product_category = sa.Column(sa.String, nullable=False)
    product_name = sa.Column(sa.String, nullable=False)
    product_quantity = sa.Column(sa.Integer, nullable=False)
    price = sa.Column(sa.Numeric(10, 2), nullable=False)
    purchase_date = sa.Column(sa.DateTime, nullable=False)
    
# Create a session
Session = sessionmaker(bind=engine)

def serialize(model):
    columns = [c.key for c in class_mapper(model.__class__).columns]
    return {c: getattr(model, c) for c in columns}
# Example queries
def get_all_products():
    try:
        session = Session()
        products = session.query(Product).all()
        
        # Serialize the products to JSON
        products_json = json.dumps([serialize(product) for product in products], default=str)
        
        return products_json
    except SQLAlchemyError as e:
        logger.error(f"Database error in get_all_products: {str(e)}")
        return json.dumps([])
    except Exception as e:
        logger.error(f"Unexpected error in get_all_products: {str(e)}")
        return json.dumps([])
    finally:
        session.close()

def get_customer_purchases(uuid):
    try:
        session = Session()
        purchases = session.query(PurchaseHistory).filter(PurchaseHistory.customer_uuid == uuid).all()
        
        # Serialize the purchases to JSON
        purchases_json = json.dumps([serialize(purchase) for purchase in purchases], default=str)
        
        return purchases_json
    except SQLAlchemyError as e:
        logger.error(f"Database error in get_customer_purchases for UUID {uuid}: {str(e)}")
        return json.dumps([])
    except Exception as e:
        logger.error(f"Unexpected error in get_customer_purchases for UUID {uuid}: {str(e)}")
        return json.dumps([])
    finally:
        session.close()


def get_products_by_category(category):
    try:
        session = Session()
        products = session.query(Product).filter(Product.category == category).all()
        return products
    except SQLAlchemyError as e:
        logger.error(f"Database error in get_products_by_category for category {category}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in get_products_by_category for category {category}: {str(e)}")
        return []
    finally:
        session.close()

async def get_recommendations(purchase_history, product_list):
    try:
        logger.info("Invoking get_recommendations function")
        # Initialize the language model
        llm = ChatOllama(model="llama3.1", temperature=0)
        # Create the LLMChain
       
        chain = prompt | llm | StrOutputParser()

        # Prepare the input
        input_data = {
            "purchase_history": purchase_history,
            "product_list": product_list
        }
        
        # Run the chain
        result = await chain.ainvoke(input_data)
        
        # Log raw result for debugging
        logger.debug(f"Raw LLM output: {result}")
        
        try:
            recommendations = output_parser.parse(result)
            return recommendations
        except ValidationError as ve:
            logger.error(f"Pydantic validation error: {str(ve)}")
            # Attempt to parse the result as JSON for more information
            try:
                parsed_result = json.loads(result)
                logger.error(f"Parsed invalid result: {json.dumps(parsed_result, indent=2)}")
            except json.JSONDecodeError:
                logger.error("Result is not valid JSON")
            return None
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        return None
