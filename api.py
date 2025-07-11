import os
import json
import hashlib
import tempfile
from fastapi import FastAPI, UploadFile, HTTPException, Request, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import uvicorn
from contextlib import asynccontextmanager
import base64
from typing import Optional
import requests
import shutil
from pathlib import Path
from diskcache import Cache

from polygon_inference import PolygonInference
from utils import log

# API Key configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Get API key from environment variable
API_KEY = os.getenv("API_KEY")
EXPERIMENT_PATH = os.getenv("EXPERIMENT_PATH", "runs_share/Pix2Poly_inria_coco_224")
MODEL_URL = os.getenv("MODEL_URL", "https://github.com/safelease/Pix2Poly/releases/download/main/runs_share.zip")

# Cache configuration
CACHE_TTL = int(os.getenv("CACHE_TTL", 24 * 3600))  # 24 hours

# Global cache instance
cache = Cache(
    directory=os.path.join(tempfile.gettempdir(), "pix2poly_cache"),
    timeout=1,  # 1 second timeout for cache operations
    disk_min_file_size=0,  # Store all items on disk
    disk_pickle_protocol=4,  # Use protocol 4 for better compatibility
)

def get_cache_key(image_data: bytes) -> str:
    """Generate a cache key from image data.
    
    Args:
        image_data: Raw image data
        
    Returns:
        SHA-256 hash of the image data as a string
    """
    return hashlib.sha256(image_data).hexdigest()


async def verify_api_key(
    header_key: Optional[str] = Depends(api_key_header),
    query_key: Optional[str] = Query(None, alias="api_key"),
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
        raise HTTPException(status_code=401, detail="API key is missing")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


def download_model_files(model_url: str, target_dir: str) -> str:
    """Download model files to the target directory.

    Args:
        model_url: URL to download the model files from
        target_dir: Directory to save the model files to

    Returns:
        Path to the downloaded model directory

    Raises:
        ValueError: If download fails or model files are invalid
    """
    # Create target directory if it doesn't exist
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Check if model files already exist
    if target_path.exists() and any(target_path.iterdir()):
        log(f"Model files already exist in {target_dir}, skipping download", "INFO")
        return str(target_path)

    # Download the model files using requests
    zip_path = target_path / "runs_share.zip"
    
    log(f"Downloading model files from {model_url}", "INFO")
    response = requests.get(model_url, stream=True)
    response.raise_for_status()
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the zip file
    log(f"Extracting model files to {target_dir}", "INFO")
    shutil.unpack_archive(zip_path, target_path)

    # Remove the zip file
    zip_path.unlink()

    return str(target_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the predictor on startup."""
    # Download model files to a temporary directory
    model_dir = download_model_files(
        MODEL_URL,
        "/tmp/pix2poly_model",
    )

    # Initialize predictor with downloaded model
    init_predictor(os.path.join(model_dir, EXPERIMENT_PATH))
    yield


app = FastAPI(
    title="Polygon Inference API",
    description="API for inferring polygons in images using a trained model",
    version="1.0.0",
    lifespan=lifespan,
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
async def invoke(
    request: Request,
    file: UploadFile = None,
    api_key: Optional[str] = Depends(verify_api_key),
    merge_tolerance: Optional[float] = Query(None, description="Tolerance for point-in-polygon tests during validation (in pixels, allows points to be slightly outside)"),
    tile_overlap_ratio: Optional[float] = Query(None, description="Overlap ratio between tiles (0.0 = no overlap, 1.0 = complete overlap)"),
):
    """Main inference endpoint for processing images.

    The endpoint accepts image data in three different ways:
    1. As a file upload using multipart/form-data (via the file parameter)
    2. As a base64-encoded image in a JSON payload with an 'image' field
    3. As raw image data in the request body

    Authentication can be provided in two ways:
    1. Via the X-API-Key header
    2. Via the api_key query parameter

    Configuration parameters can be provided in two ways:
    1. Via query parameters (merge_tolerance, tile_overlap_ratio)
    2. Via the JSON payload fields (merge_tolerance, tile_overlap_ratio)

    Args:
        request: The request containing the image data
        file: Optional uploaded file (multipart/form-data)
        api_key: Optional API key for authentication (required only if API key is configured)
        merge_tolerance: Optional tolerance for point-in-polygon tests during validation (in pixels, allows points to be slightly outside)
        tile_overlap_ratio: Optional overlap ratio between tiles (0.0 = no overlap, 1.0 = complete overlap)

    Returns:
        JSON response containing the inferred polygons

    Raises:
        HTTPException: 400 if no image data is found in the request
        HTTPException: 500 if there is an error processing the image
        HTTPException: 401 if API key is missing (when API key is configured)
        HTTPException: 403 if API key is invalid (when API key is configured)
    """
    log(f"Invoking image analysis")

    # Initialize configuration parameters and validate ranges
    effective_merge_tolerance = merge_tolerance
    effective_tile_overlap_ratio = tile_overlap_ratio
    
    # Validate merge_tolerance (should be positive)
    if effective_merge_tolerance is not None and effective_merge_tolerance < 0:
        raise HTTPException(status_code=400, detail="merge_tolerance must be non-negative")
    
    # Validate tile_overlap_ratio (should be between 0.0 and 1.0)
    if effective_tile_overlap_ratio is not None and (effective_tile_overlap_ratio < 0.0 or effective_tile_overlap_ratio > 1.0):
        raise HTTPException(status_code=400, detail="tile_overlap_ratio must be between 0.0 and 1.0")

    if file:
        # Handle file upload
        image_data = await file.read()
    else:
        # Read request body
        body = await request.body()

        # Parse the request body
        try:
            data = json.loads(body)
            if "image" in data:
                # Handle base64 encoded image
                image_data = base64.b64decode(data["image"])
                
                # Extract configuration parameters from JSON (if query params not provided)
                if effective_merge_tolerance is None and "merge_tolerance" in data:
                    effective_merge_tolerance = float(data["merge_tolerance"])
                    if effective_merge_tolerance < 0:
                        raise HTTPException(status_code=400, detail="merge_tolerance must be non-negative")
                if effective_tile_overlap_ratio is None and "tile_overlap_ratio" in data:
                    effective_tile_overlap_ratio = float(data["tile_overlap_ratio"])
                    if effective_tile_overlap_ratio < 0.0 or effective_tile_overlap_ratio > 1.0:
                        raise HTTPException(status_code=400, detail="tile_overlap_ratio must be between 0.0 and 1.0")
            else:
                raise HTTPException(
                    status_code=400, detail="No image data found in request"
                )
        except json.JSONDecodeError:
            # Handle raw image data
            image_data = body

    # Generate cache key including configuration parameters
    cache_key_base = get_cache_key(image_data)
    cache_key = f"{cache_key_base}_{effective_merge_tolerance}_{effective_tile_overlap_ratio}"
    cached_result = cache.get(cache_key)
    
    if cached_result is not None:
        return JSONResponse(content=cached_result)

    # Get inferences
    polygons = predictor.infer(image_data, merge_tolerance=effective_merge_tolerance, tile_overlap_ratio=effective_tile_overlap_ratio)

    # Prepare response
    response = {
        "polygons": polygons,
    }

    # Store result in cache
    cache.set(cache_key, response, expire=CACHE_TTL)

    return JSONResponse(content=response)

@app.get("/ping")
async def ping(api_key: Optional[str] = Depends(verify_api_key)):
    """Health check endpoint to verify service status."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.get("/clear-cache")
async def clear_cache(api_key: Optional[str] = Depends(verify_api_key)):
    """Clear the cache endpoint to remove all cached results."""
    cache.clear()
    return {"status": "success", "message": "Cache cleared successfully"}


if __name__ == "__main__":
    # Run the API
    uvicorn.run(app, host="0.0.0.0", port=8080)
