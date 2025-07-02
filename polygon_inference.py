# Standard library imports
import os
import time
import hashlib
import pickle
from typing import List, Tuple, Dict, Optional, Any

# Third-party imports
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

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
from models.model import Encoder, Decoder, EncoderDecoder


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
        self.cache_dir: str = "/tmp/pix2poly_cache"
        self._ensure_cache_dir()
        self._initialize_model()

    def _ensure_cache_dir(self) -> None:
        """Ensure the cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_cache_key(self, tiles: List[np.ndarray]) -> str:
        """Generate a cache key based on the input tiles.
        
        Args:
            tiles (List[np.ndarray]): List of tile images
            
        Returns:
            str: Hash-based cache key
        """
        # Create a hash based on all tile data
        hasher = hashlib.sha256()
        for tile in tiles:
            hasher.update(tile.tobytes())
        return hasher.hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the full path for a cache file.
        
        Args:
            cache_key (str): The cache key
            
        Returns:
            str: Full path to the cache file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict[str, List[np.ndarray]]]]:
        """Load results from cache if they exist.
        
        Args:
            cache_key (str): The cache key to look for
            
        Returns:
            Optional[List[Dict[str, List[np.ndarray]]]]: Cached results if found, None otherwise
        """
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                log(f"Failed to load cache from {cache_path}: {e}")
                # Remove corrupted cache file
                try:
                    os.remove(cache_path)
                except:
                    pass
        return None

    def _save_to_cache(self, cache_key: str, results: List[Dict[str, List[np.ndarray]]]) -> None:
        """Save results to cache.
        
        Args:
            cache_key (str): The cache key
            results (List[Dict[str, List[np.ndarray]]]): Results to cache
        """
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            log(f"Failed to save cache to {cache_path}: {e}")

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
            max_len=CFG.MAX_LEN,
        )
        CFG.PAD_IDX = self.tokenizer.PAD_code

        encoder = Encoder(model_name=CFG.MODEL_NAME, pretrained=True, out_dim=256)
        decoder = Decoder(
            cfg=CFG,
            vocab_size=self.tokenizer.vocab_size,
            encoder_len=CFG.NUM_PATCHES,
            dim=256,
            num_heads=8,
            num_layers=6,
        )
        self.model = EncoderDecoder(cfg=CFG, encoder=encoder, decoder=decoder)
        self.model.to(self.device)
        self.model.eval()

        # Load latest checkpoint
        latest_checkpoint = self._find_single_checkpoint()
        checkpoint_path = os.path.join(
            self.experiment_path, "logs", "checkpoints", latest_checkpoint
        )
        log(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint["state_dict"])
        log("Checkpoint loaded successfully")

    def _find_single_checkpoint(self) -> str:
        """Find the single checkpoint file. Crashes if there is more than one checkpoint.

        Returns:
            str: Filename of the single checkpoint

        Raises:
            FileNotFoundError: If no checkpoint directory or files are found
            RuntimeError: If more than one checkpoint file is found
        """
        checkpoint_dir = os.path.join(self.experiment_path, "logs", "checkpoints")
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        checkpoint_files = [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("epoch_") and f.endswith(".pth")
        ]
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

        if len(checkpoint_files) > 1:
            raise RuntimeError(
                f"Multiple checkpoint files found in {checkpoint_dir}: {checkpoint_files}. Expected exactly one checkpoint."
            )

        return checkpoint_files[0]

    def _process_tiles_batch(
        self, tiles: List[np.ndarray]
    ) -> List[Dict[str, List[np.ndarray]]]:
        """Process a single batch of tiles.

        Args:
            tiles (list[np.ndarray]): List of tile images to process

        Returns:
            list[dict]: List of results for each tile, where each result contains:
                - polygons: List of polygon coordinates
        """
        # Generate cache key and try to load from cache
        cache_key = self._generate_cache_key(tiles)
        cached_results = self._load_from_cache(cache_key)
        if cached_results is not None:
            log(f"Cache hit for batch of {len(tiles)} tiles")
            return cached_results

        log(f"Cache miss for batch of {len(tiles)} tiles, processing...")
        
        valid_transforms = A.Compose(
            [
                A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        )

        # Transform each tile individually and stack them
        transformed_tiles: List[torch.Tensor] = []
        for tile in tiles:
            transformed = valid_transforms(image=tile)
            transformed_tiles.append(transformed["image"])

        # Stack the transformed tiles into a batch
        batch_tensor = torch.stack(transformed_tiles).to(self.device)

        with torch.no_grad():
            batch_preds, batch_confs, perm_preds = test_generate(
                self.model,
                batch_tensor,
                self.tokenizer,
                max_len=CFG.generation_steps,
                top_k=0,
                top_p=1,
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

                batch_polygons = permutations_to_polygons(
                    perm_preds[j : j + 1], [coord], out="torch"
                )

                valid_polygons: List[np.ndarray] = []
                for poly in batch_polygons[0]:
                    poly = poly[poly[:, 0] != CFG.PAD_IDX]
                    if len(poly) > 0:
                        valid_polygons.append(
                            poly.cpu().numpy()[:, ::-1]
                        )  # Convert to [x,y] format

                result = {"polygons": valid_polygons}

                results.append(result)

        # Save results to cache
        self._save_to_cache(cache_key, results)
        
        return results

    def _create_tile_visualization(
        self,
        tiles: List[np.ndarray],
        tile_results: List[Dict[str, List[np.ndarray]]],
        positions: List[Tuple[int, int, int, int]],
    ) -> None:
        """Create a tile visualization showing each tile with its detected polygons.

        Args:
            tiles (List[np.ndarray]): List of tile images
            tile_results (List[Dict[str, List[np.ndarray]]]): List of results for each tile
            positions (List[Tuple[int, int, int, int]]): List of (x, y, x_end, y_end) tuples for each tile's position
        """
        if not tiles:
            return

        # Calculate grid dimensions
        num_tiles = len(tiles)
        cols = math.ceil(math.sqrt(num_tiles))
        rows = math.ceil(num_tiles / cols)
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i in range(num_tiles):
            ax = axes[i]
            tile = tiles[i]
            tile_result = tile_results[i]
            
            # Convert RGB to display format
            display_tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR) if tile.shape[-1] == 3 else tile
            display_tile = cv2.cvtColor(display_tile, cv2.COLOR_BGR2RGB)
            
            ax.imshow(display_tile)
            ax.set_title(f'Tile {i+1}')
            ax.axis('off')
            
            # Draw polygons on this tile
            for poly in tile_result["polygons"]:
                if len(poly) > 2:
                    # Close the polygon for visualization
                    poly_closed = np.vstack([poly, poly[0]])
                    ax.plot(poly_closed[:, 0], poly_closed[:, 1], 'g-', linewidth=2)
                    
                    # Draw vertices
                    ax.scatter(poly[:, 0], poly[:, 1], c='red', s=20, zorder=5)

        # Hide unused subplots
        for i in range(num_tiles, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('tile-visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        log(f"Saved tile visualization with {num_tiles} tiles to tile-visualization.png")

    def _merge_polygons(
        self,
        tile_results: List[Dict[str, List[np.ndarray]]],
        positions: List[Tuple[int, int, int, int]],
        image_height: int,
        image_width: int,
    ) -> List[np.ndarray]:
        """Merge polygon predictions from multiple tiles using a bitmap approach.

        This method creates a bitmap where pixels inside any polygon are set to True,
        then vectorizes the bitmap back to polygons. This eliminates geometric artifacts
        from traditional polygon union operations.

        Args:
            tile_results (list[dict]): List of dictionaries containing 'polygons' for each tile
            positions (list[tuple[int, int, int, int]]): List of (x, y, x_end, y_end) tuples for each tile's position
            image_height (int): Height of the original image
            image_width (int): Width of the original image

        Returns:
            list[np.ndarray]: List of merged polygons in original image coordinates
        """
        # Scale factor for subpixel precision
        scale_factor = 16
        
        # Create bitmap at 8x resolution for subpixel precision
        bitmap = np.zeros((image_height * scale_factor, image_width * scale_factor), dtype=np.uint8)
        
        # Fill bitmap with polygon regions
        for tile_idx, (tile_result, (x, y, x_end, y_end)) in enumerate(zip(tile_results, positions)):
            tile_polygons = tile_result["polygons"]
            
            for poly_idx, poly in enumerate(tile_polygons):
                if len(poly) < 3:  # Skip invalid polygons
                    continue
                
                # Check if polygon is in a corner (should be removed)
                tile_width = x_end - x
                tile_height = y_end - y
                edge_tolerance = 8.0  # Consider points within 8 pixels of edge as "on edge"
                corner_tolerance = 2.0  # Consider points within 2 pixels of corner as "near corner"
                
                # Check which edges have points
                on_left_edge = poly[:, 0] <= edge_tolerance
                on_right_edge = poly[:, 0] >= tile_width - edge_tolerance
                on_top_edge = poly[:, 1] <= edge_tolerance
                on_bottom_edge = poly[:, 1] >= tile_height - edge_tolerance
                
                # Check if near any corner
                near_top_left = (poly[:, 0] <= corner_tolerance) & (poly[:, 1] <= corner_tolerance)
                near_top_right = (poly[:, 0] >= tile_width - corner_tolerance) & (poly[:, 1] <= corner_tolerance)
                near_bottom_left = (poly[:, 0] <= corner_tolerance) & (poly[:, 1] >= tile_height - corner_tolerance)
                near_bottom_right = (poly[:, 0] >= tile_width - corner_tolerance) & (poly[:, 1] >= tile_height - corner_tolerance)
                
                # Check for corner polygons (polygons that span two adjacent edges AND are near the corner)
                is_corner_polygon = (
                    (np.any(on_top_edge) and np.any(on_left_edge) and np.any(near_top_left)) or
                    (np.any(on_top_edge) and np.any(on_right_edge) and np.any(near_top_right)) or
                    (np.any(on_bottom_edge) and np.any(on_left_edge) and np.any(near_bottom_left)) or
                    (np.any(on_bottom_edge) and np.any(on_right_edge) and np.any(near_bottom_right))
                )
                
                if is_corner_polygon:
                   continue
                
                # Transform polygon from tile coordinates to image coordinates
                transformed_poly = poly + np.array([x, y])
                
                # Scale up coordinates for high-resolution bitmap
                scaled_poly = transformed_poly * scale_factor
                
                # Ensure coordinates are within scaled bitmap bounds
                scaled_poly[:, 0] = np.clip(scaled_poly[:, 0], 0, image_width * scale_factor - 1)
                scaled_poly[:, 1] = np.clip(scaled_poly[:, 1], 0, image_height * scale_factor - 1)
                
                # Convert to integer coordinates for rasterization
                poly_coords = scaled_poly.astype(np.int32)
                
                # Fill the polygon region in the bitmap
                cv2.fillPoly(bitmap, [poly_coords], 255)
        
        # Apply morphological closing to fill small gaps and smooth edges
        # Scale kernel size proportionally to the scaled bitmap
        kernel_size = max(1, min(3 * scale_factor, min(image_height, image_width) * scale_factor // 1000))  # Adaptive kernel size
        if kernel_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            bitmap = cv2.morphologyEx(bitmap, cv2.MORPH_CLOSE, kernel)
        
        # Save bitmap for debugging (optional)
        cv2.imwrite('debug_polygon_bitmap.png', bitmap)
        log("Saved debug bitmap to debug_polygon_bitmap.png")
        
        # Find contours in the bitmap
        contours, _ = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        merged_polygons: List[np.ndarray] = []
        
        for contour in contours:
            # Skip very small contours (area is scaled by scale_factor^2)
            area = cv2.contourArea(contour)
            if area < CFG.MIN_POLYGON_AREA * (scale_factor ** 2):
                continue
            
            # Simplify the contour to reduce jaggedness while preserving shape
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.01 * perimeter  # 1% of perimeter
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert from OpenCV format to our polygon format
            if len(simplified_contour) >= 3:  # Valid polygon needs at least 3 points
                # Reshape from (n, 1, 2) to (n, 2) and convert to float
                polygon_coords = simplified_contour.reshape(-1, 2).astype(np.float32)
                
                # Scale down coordinates back to original image coordinate system
                polygon_coords = polygon_coords / scale_factor
                
                merged_polygons.append(polygon_coords)
        
        log(f"Bitmap approach: {len(merged_polygons)} polygons extracted from bitmap")
        return merged_polygons

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

        overlap_ratio = 0.5

        bboxes = calculate_slice_bboxes(
            image_height=height,
            image_width=width,
            slice_height=CFG.TILE_SIZE,
            slice_width=CFG.TILE_SIZE,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
        )

        tiles: List[np.ndarray] = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            tile = image[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            tiles.append(tile)

        # Process tiles in batches
        all_results: List[Dict[str, List[np.ndarray]]] = []

        for i in range(0, len(tiles), CFG.PREDICTION_BATCH_SIZE):
            batch_start_time = time.time()
            batch_tiles = tiles[i : i + CFG.PREDICTION_BATCH_SIZE]
            batch_results = self._process_tiles_batch(batch_tiles)
            all_results.extend(batch_results)
            
            batch_time = time.time() - batch_start_time
            log(f"Processed batch of {len(batch_tiles)} tiles: {batch_time/len(batch_tiles):.3f}s per tile")

        # Create tile visualization
        self._create_tile_visualization(tiles, all_results, bboxes)

        merged_polygons = self._merge_polygons(all_results, bboxes, height, width)

        # Convert to list format
        polygons_list = [poly.tolist() for poly in merged_polygons]
        # Round coordinates to two decimal places
        polygons_list = [
            [[round(x, 2), round(y, 2)] for x, y in polygon]
            for polygon in polygons_list
        ]

        return polygons_list
