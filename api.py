import os
import json
from fastapi import FastAPI, UploadFile, HTTPException, Request, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import uvicorn
from contextlib import asynccontextmanager
import base64
from typing import Optional

from polygon_inference import PolygonInference
from utils import log

# API Key configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Get API key from environment variable
API_KEY = os.getenv("API_KEY", "")

async def verify_api_key(
    header_key: Optional[str] = Depends(api_key_header),
    query_key: Optional[str] = Query(None, alias="api_key")
) -> Optional[str]:
    """Verify the API key from either header or query parameter.
    
    If API authentication is not enabled (no API key configured),
    this function will always return None.
    
    Args:
        header_key: API key from X-API-Key header
        query_key: API key from api_key query parameter
        
    Returns:
        The verified API key or None if authentication is disabled
        
    Raises:
        HTTPException: 401 if API key is missing (when required)
        HTTPException: 403 if API key is invalid (when required)
    """
    if not API_KEY:
        return None
        
    api_key = header_key or query_key
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is missing"
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the predictor on startup."""
    experiment_path = os.getenv("EXPERIMENT_PATH")
    if not experiment_path:
        raise ValueError("EXPERIMENT_PATH environment variable must be set")
    init_predictor(experiment_path)
    yield

app = FastAPI(
    title="Polygon Inference API",
    description="API for inferring polygons in images using a trained model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global predictor instance
predictor = None

def init_predictor(experiment_path: str):
    """Initialize the global predictor instance."""
    global predictor
    if predictor is None:
        predictor = PolygonInference(experiment_path)

@app.post("/invocations")
async def invoke(request: Request, file: UploadFile = None, api_key: Optional[str] = Depends(verify_api_key)):
    """Main inference endpoint for processing images.
    
    The endpoint accepts image data in three different ways:
    1. As a file upload using multipart/form-data (via the file parameter)
    2. As a base64-encoded image in a JSON payload with an 'image' field
    3. As raw image data in the request body
    
    Authentication can be provided in two ways:
    1. Via the X-API-Key header
    2. Via the api_key query parameter
    
    Args:
        request: The request containing the image data
        file: Optional uploaded file (multipart/form-data)
        api_key: Optional API key for authentication (required only if API key is configured)
        
    Returns:
        JSON response containing the inferred polygons
        
    Raises:
        HTTPException: 400 if no image data is found in the request
        HTTPException: 500 if there is an error processing the image
        HTTPException: 401 if API key is missing (when API key is configured)
        HTTPException: 403 if API key is invalid (when API key is configured)
    """
    try:
        if file:
            # Handle file upload
            image_data = await file.read()
        else:
            # Read request body
            body = await request.body()
            
            # Parse the request body
            try:
                data = json.loads(body)
                if 'image' in data:
                    # Handle base64 encoded image
                    image_data = base64.b64decode(data['image'])
                else:
                    raise HTTPException(status_code=400, detail="No image data found in request")
            except json.JSONDecodeError:
                # Handle raw image data
                image_data = body
        
        # Get inferences
        polygons = predictor.infer(image_data)
        
        # Prepare response
        response = {
            "polygons": polygons,
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        log(f"Error processing image: {str(e)}", "ERROR")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping(api_key: Optional[str] = Depends(verify_api_key)):
    """Health check endpoint to verify service status."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the API
    uvicorn.run(app, host="0.0.0.0", port=8080) 