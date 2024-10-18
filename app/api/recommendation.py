from fastapi import APIRouter, Request, Query
from uuid import UUID
from tools.recommendation import get_all_products, get_customer_purchases, get_recommendations
from app.utils.logger import logger

router = APIRouter()

@router.get("/recommendations", tags=["recommendation"])
async def get_recommendations_endpoint(
    request: Request, 
    customer_uuid: UUID = Query(..., description="Customer UUID")
):
    try:
        purchase_history = get_all_products()
        product_list = get_customer_purchases(customer_uuid)
        recommendations = await get_recommendations(purchase_history, product_list)

        return recommendations
    except Exception as e:
        logger.error(f"Error in get_recommendations_endpoint: {str(e)}")
        return {"error": "An error occurred while processing your request."}
