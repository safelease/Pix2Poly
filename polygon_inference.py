# Standard library imports
import os
import tempfile
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

# Third-party imports
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shapely.geometry
import shapely.ops

# Local imports
from config import CFG
from tokenizer import Tokenizer
from utils import (
    seed_everything,
    test_generate,
    postprocess,
    permutations_to_polygons,
    log,
    calculate_slice_bboxes,
)
from models.model import (
    Encoder,
    Decoder,
    EncoderDecoder
)

# Cache configuration
ENABLE_CACHE = False  # Set to False to disable caching by default

class PolygonInference:
    def __init__(self, experiment_path: str, device: Optional[str] = None) -> None:
        """Initialize the polygon inference with a trained model.
        
        Args:
            experiment_path (str): Path to the experiment folder containing the model checkpoint
            device (str | None, optional): Device to run the model on. Defaults to CFG.DEVICE
        """
        self.device: str = device or CFG.DEVICE
        self.experiment_path: str = os.path.realpath(experiment_path)
        self.model: Optional[EncoderDecoder] = None
        self.tokenizer: Optional[Tokenizer] = None
        self._initialize_model()
        
        # Create persistent temporary directory for caching only if caching is enabled
        if ENABLE_CACHE:
            self.cache_dir: Path = Path(tempfile.gettempdir()) / "pix2poly_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        
    def _initialize_model(self) -> None:
        """Initialize the model and tokenizer.
        
        This method:
        1. Creates a new tokenizer instance
        2. Initializes the encoder-decoder model
        3. Loads the latest checkpoint from the experiment directory
        """
        self.tokenizer = Tokenizer(
            num_classes=1,
            num_bins=CFG.NUM_BINS,
            width=CFG.INPUT_WIDTH,
            height=CFG.INPUT_HEIGHT,
            max_len=CFG.MAX_LEN
        )
        CFG.PAD_IDX = self.tokenizer.PAD_code

        encoder = Encoder(model_name=CFG.MODEL_NAME, pretrained=True, out_dim=256)
        decoder = Decoder(
            cfg=CFG,
            vocab_size=self.tokenizer.vocab_size,
            encoder_len=CFG.NUM_PATCHES,
            dim=256,
            num_heads=8,
            num_layers=6
        )
        self.model = EncoderDecoder(cfg=CFG, encoder=encoder, decoder=decoder)
        self.model.to(self.device)
        self.model.eval()
        
        # Load latest checkpoint
        latest_checkpoint = self._find_latest_checkpoint()
        checkpoint_path = os.path.join(self.experiment_path, "logs", "checkpoints", latest_checkpoint)
        log(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])
        log("Checkpoint loaded successfully")

    def _find_latest_checkpoint(self) -> str:
        """Find the checkpoint with the highest epoch number.
        
        Returns:
            str: Filename of the latest checkpoint
        """
        checkpoint_dir = os.path.join(self.experiment_path, "logs", "checkpoints")
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_') and f.endswith('.pth')]
        latest_checkpoint = sorted(checkpoint_files)[-1]
        return latest_checkpoint

    def _get_tile_hash(self, tile: np.ndarray) -> str:
        """Generate a hash for the input tile.
        
        Args:
            tile (np.ndarray): Input image tile
            
        Returns:
            str: MD5 hash of the tile
        """
        return hashlib.md5(tile.tobytes()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached result if it exists.
        
        Args:
            cache_key (str): Key to look up in the cache
            
        Returns:
            dict | None: Cached result if found, None otherwise
        """
        if not ENABLE_CACHE or self.cache_dir is None:
            return None
            
        cache_path = self.cache_dir / f"{cache_key}.npy"
        if cache_path.exists():
            return np.load(cache_path, allow_pickle=True).item()
        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Save result to cache.
        
        Args:
            cache_key (str): Key to store in the cache
            result (dict): Result to cache
        """
        if not ENABLE_CACHE or self.cache_dir is None:
            return
            
        cache_path = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_path, result, allow_pickle=True)

    def _process_tiles_batch(self, tiles: List[np.ndarray]) -> List[Dict[str, List[np.ndarray]]]:
        """Process a single batch of tiles.
        
        Args:
            tiles (list[np.ndarray]): List of tile images to process
            
        Returns:
            list[dict]: List of results for each tile, where each result contains:
                - polygons: List of polygon coordinates
        """
        valid_transforms = A.Compose([
            A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])
        
        # Transform each tile individually and stack them
        transformed_tiles: List[torch.Tensor] = []
        for tile in tiles:
            transformed = valid_transforms(image=tile)
            transformed_tiles.append(transformed['image'])
        
        # Stack the transformed tiles into a batch
        batch_tensor = torch.stack(transformed_tiles).to(self.device)
        
        with torch.no_grad():
            batch_preds, batch_confs, perm_preds = test_generate(
                self.model, batch_tensor, self.tokenizer,
                max_len=CFG.generation_steps,
                top_k=0,
                top_p=1
            )
            
            vertex_coords, confs = postprocess(batch_preds, batch_confs, self.tokenizer)
            
            results: List[Dict[str, List[np.ndarray]]] = []
            for j in range(len(tiles)):
                if vertex_coords[j] is not None:
                    coord = torch.from_numpy(vertex_coords[j])
                else:
                    coord = torch.tensor([])
                
                padd = torch.ones((CFG.N_VERTICES - len(coord), 2)).fill_(CFG.PAD_IDX)
                coord = torch.cat([coord, padd], dim=0)
                
                batch_polygons = permutations_to_polygons(perm_preds[j:j+1], [coord], out='torch')
                
                valid_polygons: List[np.ndarray] = []
                for poly in batch_polygons[0]:
                    poly = poly[poly[:, 0] != CFG.PAD_IDX]
                    if len(poly) > 0:
                        valid_polygons.append(poly.cpu().numpy()[:, ::-1]) # Convert to [x,y] format
                
                result = {
                    'polygons': valid_polygons
                }
                
                # Cache the result
                cache_key = self._get_tile_hash(tiles[j])
                self._save_to_cache(cache_key, result)
                
                results.append(result)
        
        return results

    def _merge_polygons(self, tile_results: List[Dict[str, List[np.ndarray]]], positions: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Merge polygon predictions from multiple tiles into a single set of polygons.
        
        Args:
            tile_results (list[dict]): List of dictionaries containing 'polygons' for each tile
            positions (list[tuple[int, int, int, int]]): List of (x, y, x_end, y_end) tuples for each tile's position
            
        Returns:
            list[np.ndarray]: List of merged polygons in original image coordinates
        """
        all_polygons: List[shapely.geometry.Polygon] = []
        
        # Transform each polygon from tile coordinates to original image coordinates
        for tile_result, (x, y, x_end, y_end) in zip(tile_results, positions):
            tile_polygons = tile_result['polygons']
            height, width = CFG.INPUT_HEIGHT, CFG.INPUT_WIDTH
            
            # Transform each polygon from tile coordinates to original image coordinates
            for poly in tile_polygons:
                # Scale coordinates to match tile size
                scaled_poly = poly * np.array([width/CFG.INPUT_WIDTH, height/CFG.INPUT_HEIGHT])
                
                # Translate to original image position
                translated_poly = scaled_poly + np.array([x, y])
                
                # Convert to shapely polygon
                shapely_poly = shapely.geometry.Polygon(translated_poly)
                if shapely_poly.is_valid and shapely_poly.area > CFG.MIN_POLYGON_AREA:
                    all_polygons.append(shapely_poly)
        
        # Use shapely's unary_union to merge overlapping polygons
        merged = shapely.ops.unary_union(all_polygons)
        
        simplified_polygons: List[np.ndarray] = []

        # Merging tiles may have created redundant points in the middle of a line.  We can remove them.
        for poly in merged.geoms:
            simplified = poly.simplify(CFG.POLYGON_SIMPLIFICATION_TOLERANCE)
            if simplified.is_valid and not simplified.is_empty:
                simplified_polygons.append(np.array(simplified.exterior.coords))
        
        return simplified_polygons

    def infer(self, image_data: bytes) -> List[List[List[float]]]:
        """Infer polygons in an image.
        
        Args:
            image_data (bytes): Raw image data
            
        Returns:
            list[list[list[float]]]: List of polygons where each polygon is a list of [x,y] coordinates.
                Each coordinate is rounded to 2 decimal places.
                
        Raises:
            ValueError: If the image data is invalid, empty, or cannot be decoded
            RuntimeError: If there are issues with model prediction or polygon processing
        """
        if not image_data:
            raise ValueError("Empty image data provided")

        seed_everything(42)

        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image data")
        
        if image.size == 0:
            raise ValueError("Decoded image is empty")

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Split image into tiles
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("Invalid image dimensions")
            
        overlap_ratio = CFG.TILE_OVERLAP / CFG.TILE_SIZE
        
        bboxes = calculate_slice_bboxes(
            image_height=height,
            image_width=width,
            slice_height=CFG.TILE_SIZE,
            slice_width=CFG.TILE_SIZE,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio
        )
        
        tiles: List[np.ndarray] = []
        tiles_to_process: List[np.ndarray] = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            tile = image[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            tiles.append(tile)
            
        # Process tiles in batches
        all_results: List[Dict[str, List[np.ndarray]]] = []
        
        # First check cache for all tiles
        for tile in tiles:
            cache_key = self._get_tile_hash(tile)
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                all_results.append(cached_result)
            else:
                tiles_to_process.append(tile)
        
        # Process remaining tiles in batches
        for i in range(0, len(tiles_to_process), CFG.PREDICTION_BATCH_SIZE):
            batch_tiles = tiles_to_process[i:i + CFG.PREDICTION_BATCH_SIZE]
            batch_results = self._process_tiles_batch(batch_tiles)
            all_results.extend(batch_results)

        merged_polygons = self._merge_polygons(all_results, bboxes)
        
        # Convert to list format
        polygons_list = [poly.tolist() for poly in merged_polygons]
        # Round coordinates to two decimal places
        polygons_list = [[[round(x, 2), round(y, 2)] for x, y in polygon] for polygon in polygons_list]
        
        return polygons_list