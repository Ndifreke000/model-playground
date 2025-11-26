"""
Vercel serverless function handler for the ML API
This converts the FastAPI app to work with Vercel's serverless platform
"""
from api.ml_service.app import app
from mangum import Mangum

# Mangum adapter converts ASGI (FastAPI) to AWS Lambda/Vercel format
handler = Mangum(app, lifespan="off")
