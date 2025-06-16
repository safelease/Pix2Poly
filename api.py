import os
import json
from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import base64

from polygon_inference import PolygonInference
from utils import log

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
async def invoke(request: Request, file: UploadFile = None):
    """Main inference endpoint for processing images.
    
    The endpoint accepts image data in three different ways:
    1. As a file upload using multipart/form-data (via the file parameter)
    2. As a base64-encoded image in a JSON payload with an 'image' field
    3. As raw image data in the request body
    
    Args:
        request: The request containing the image data
        file: Optional uploaded file (multipart/form-data)
        
    Returns:
        JSON response containing the inferred polygons
        
    Raises:
        HTTPException: 400 if no image data is found in the request
        HTTPException: 500 if there is an error processing the image
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
async def ping():
    """Health check endpoint to verify service status."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the API
    uvicorn.run(app, host="0.0.0.0", port=8080) 