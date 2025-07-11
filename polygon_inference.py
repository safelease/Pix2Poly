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
from shapely.geometry import Polygon
from buildingregulariser import regularize_geodataframe
import geopandas as gpd

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
        self, tiles: List[np.ndarray], debug: bool = False
    ) -> List[Dict[str, List[np.ndarray]]]:
        """Process a single batch of tiles.

        Args:
            tiles (list[np.ndarray]): List of tile images to process

        Returns:
            list[dict]: List of results for each tile, where each result contains:
                - polygons: List of polygon coordinates
        """
        # Generate cache key and try to load from cache (only when debug=True)
        if debug:
            cache_key = self._generate_cache_key(tiles)
            cached_results = self._load_from_cache(cache_key)
            if cached_results is not None:
                log(f"Cache hit for batch of {len(tiles)} tiles")
                return cached_results

            log(f"Cache miss for batch of {len(tiles)} tiles, processing...")
        else:
            log(f"Processing batch of {len(tiles)} tiles (caching disabled)...")
            cache_key = None
        
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

        # Save results to cache (only when debug=True)
        if debug and cache_key is not None:
            self._save_to_cache(cache_key, results)
        
        return results

    def _create_tile_visualization(
        self,
        tiles: List[np.ndarray],
        tile_results: List[Dict[str, List[np.ndarray]]],
        positions: List[Tuple[int, int, int, int]],
    ) -> None:
        """Create a tile visualization showing each tile with its detected polygons and coordinate scales.

        Args:
            tiles (List[np.ndarray]): List of tile images
            tile_results (List[Dict[str, List[np.ndarray]]]): List of results for each tile
            positions (List[Tuple[int, int, int, int]]): List of (x, y, x_end, y_end) tuples for each tile's position
        """
        if not tiles:
            return

        # Calculate grid dimensions based on actual spatial arrangement
        # Extract unique x and y starting positions
        x_positions = sorted(set(pos[0] for pos in positions))
        y_positions = sorted(set(pos[1] for pos in positions))
        
        cols = len(x_positions)
        rows = len(y_positions)
        
        # Create mapping from (x, y) position to (row, col) index
        x_to_col = {x: i for i, x in enumerate(x_positions)}
        y_to_row = {y: i for i, y in enumerate(y_positions)}
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        
        # Handle different subplot layouts
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        # Initialize all subplots as empty
        for i in range(rows):
            for j in range(cols):
                axes[i][j].axis('off')

        # Place each tile in the correct position
        for i, (tile, tile_result, pos) in enumerate(zip(tiles, tile_results, positions)):
            x, y, x_end, y_end = pos
            
            # Get the grid position for this tile
            row = y_to_row[y]
            col = x_to_col[x]
            
            ax = axes[row][col]
            
            # Tiles are already in RGB format, no conversion needed for matplotlib
            ax.imshow(tile)
            ax.set_title(f'Tile {i}')
            
            # Enable axis and set up coordinate scales
            ax.axis('on')
            
            # Get tile dimensions
            tile_height, tile_width = tile.shape[:2]
            
            # Set up x-axis ticks and labels (global coordinates)
            x_range = x_end - x
            x_step = max(1, x_range // 8)  # Show ~8 ticks across width
            x_tick_positions = range(0, tile_width, max(1, tile_width // 8))
            x_global_coords = [x + pos * x_range // tile_width for pos in x_tick_positions]
            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels([str(coord) for coord in x_global_coords], fontsize=8)
            
            # Set up y-axis ticks and labels (global coordinates)
            y_range = y_end - y
            y_step = max(1, y_range // 8)  # Show ~8 ticks across height
            y_tick_positions = range(0, tile_height, max(1, tile_height // 8))
            y_global_coords = [y + pos * y_range // tile_height for pos in y_tick_positions]
            ax.set_yticks(y_tick_positions)
            ax.set_yticklabels([str(coord) for coord in y_global_coords], fontsize=8)
            
            # Set axis limits to match tile dimensions
            ax.set_xlim(0, tile_width)
            ax.set_ylim(tile_height, 0)  # Invert y-axis for image coordinates
            
            # Style the grid and ticks
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=8, length=3)
            
            # Draw polygons on this tile
            polygons = tile_result["polygons"]
            polygon_valid = tile_result["polygon_valid"]
            
            for poly_idx, (poly, is_valid) in enumerate(zip(polygons, polygon_valid)):
                if len(poly) > 2:
                    # Use green for valid polygons, red for invalid ones
                    color = 'g' if is_valid else 'r'
                    vertex_color = 'red' if is_valid else 'darkred'
                    
                    # Close the polygon for visualization
                    poly_closed = np.vstack([poly, poly[0]])
                    ax.plot(poly_closed[:, 0], poly_closed[:, 1], f'{color}-', linewidth=2)
                    
                    # Draw vertices
                    ax.scatter(poly[:, 0], poly[:, 1], c=vertex_color, s=20, zorder=5)
                    
                    # Calculate centroid and render polygon index
                    centroid_x = np.mean(poly[:, 0])
                    centroid_y = np.mean(poly[:, 1])
                    
                    # Use white text with black outline for visibility
                    text_color = 'white'
                    outline_color = 'black'
                    
                    # Add text with outline for better visibility
                    ax.text(centroid_x, centroid_y, str(poly_idx), 
                           fontsize=12, fontweight='bold', color=text_color,
                           ha='center', va='center', zorder=6,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=outline_color, alpha=0.7))

        plt.tight_layout()
        plt.savefig('tile-visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        log(f"Saved tile visualization to tile-visualization.png")

    def _validate_all_polygons(
        self, 
        tile_results: List[Dict[str, List[np.ndarray]]], 
        positions: List[Tuple[int, int, int, int]], 
        image_height: int, 
        image_width: int,
        merge_tolerance: float
    ) -> List[Dict[str, List[np.ndarray]]]:
        """Validate all polygons in the tile results and add validation attributes.
        
        This method implements a heuristic to validate polygons by checking if their boundary edges
        have points that are contained in polygons from other tiles.
        
        Args:
            tile_results (List[Dict[str, List[np.ndarray]]]): List of tile results containing polygons
            positions (List[Tuple[int, int, int, int]]): List of (x, y, x_end, y_end) tuples for each tile's position
            image_height (int): Height of the original image
            image_width (int): Width of the original image
            merge_tolerance (float): Tolerance for point-in-polygon tests during validation (in pixels)
            
        Returns:
            List[Dict[str, List[np.ndarray]]]: Updated tile results with validation attributes
        """
        log("Starting polygon validation process...")
        
        # Initialize polygon_valid list for each tile
        total_polygons = 0
        for tile_idx, tile_result in enumerate(tile_results):
            tile_result["polygon_valid"] = [True] * len(tile_result["polygons"])
            total_polygons += len(tile_result["polygons"])
            log(f"Tile {tile_idx}: {len(tile_result['polygons'])} polygons to validate")
        
        log(f"Total polygons to validate: {total_polygons}")
        
        # Remove overlapping polygons within each tile (before edge validation)
        
        def check_polygon_overlap(poly1, poly2):
            """Check if two polygons overlap using Shapely."""
            try:
                # Convert numpy arrays to Shapely polygons
                if len(poly1) < 3 or len(poly2) < 3:
                    return False
                
                shapely_poly1 = Polygon(poly1)
                shapely_poly2 = Polygon(poly2)
                
                # Check if polygons are valid
                if not shapely_poly1.is_valid or not shapely_poly2.is_valid:
                    return False
                
                # Check for intersection (but not just touching)
                return shapely_poly1.intersects(shapely_poly2) and not shapely_poly1.touches(shapely_poly2)
            except:
                return False

        def calculate_polygon_area(poly):
            """Calculate the area of a polygon."""
            try:
                if len(poly) < 3:
                    return 0
                shapely_poly = Polygon(poly)
                if not shapely_poly.is_valid:
                    return 0
                return shapely_poly.area
            except:
                return 0
        
        log("Phase 1: Removing overlapping polygons within each tile...")
        overlap_removed_count = 0
        
        for tile_idx, tile_result in enumerate(tile_results):
            polygons = tile_result["polygons"]
            polygon_valid = tile_result["polygon_valid"]
            
            if len(polygons) <= 1:
                log(f"Tile {tile_idx}: Skipping overlap check (≤1 polygon)")
                continue  # Skip tiles with 0 or 1 polygon
            
            # Keep iterating until no overlaps are found
            while True:
                # Get currently valid polygons with their indices
                valid_polygons = [(i, poly) for i, poly in enumerate(polygons) if polygon_valid[i]]
                
                if len(valid_polygons) <= 1:
                    break  # No overlaps possible with 0 or 1 valid polygons
                
                # Find all overlapping pairs
                overlapping_pairs = []
                for i in range(len(valid_polygons)):
                    for j in range(i + 1, len(valid_polygons)):
                        idx1, poly1 = valid_polygons[i]
                        idx2, poly2 = valid_polygons[j]
                        
                        if check_polygon_overlap(poly1, poly2):
                            overlapping_pairs.append((idx1, idx2))
                            log(f"Tile {tile_idx}: Overlap detected between polygon {idx1} and polygon {idx2}")
                
                if not overlapping_pairs:
                    log(f"Tile {tile_idx}: No overlaps found, overlap validation complete")
                    break  # No overlaps found
                
                # Find all polygons involved in overlaps
                overlapping_indices = set()
                for idx1, idx2 in overlapping_pairs:
                    overlapping_indices.add(idx1)
                    overlapping_indices.add(idx2)
                
                # Calculate areas for overlapping polygons
                polygon_areas = []
                for idx in overlapping_indices:
                    area = calculate_polygon_area(polygons[idx])
                    polygon_areas.append((idx, area))
                
                # Find the largest polygon
                largest_idx, largest_area = max(polygon_areas, key=lambda x: x[1])
                
                # Mark the largest polygon as invalid
                polygon_valid[largest_idx] = False
                overlap_removed_count += 1
                log(f"Tile {tile_idx}: Marking polygon {largest_idx} as INVALID (largest overlapping polygon, area={largest_area:.2f})")
                
                # Log areas of all overlapping polygons for context
                for idx, area in polygon_areas:
                    if idx != largest_idx:
                        log(f"Tile {tile_idx}: Polygon {idx} (area={area:.2f}) remains valid")
                
                # Continue to next iteration to check for remaining overlaps
        
        log(f"Phase 1 complete: {overlap_removed_count} polygons removed due to overlaps")
        
        # Now perform edge validation on remaining valid polygons
        log("Phase 2: Validating polygons with boundary edges...")
        edge_rejected_count = 0
        
        def is_edge_near_tile_boundary(p1, p2, tile_bounds, tolerance=2):
            """Check if an edge is colinear with the tile boundary within tolerance."""
            x_min, y_min, x_max, y_max = tile_bounds
            x1, y1 = p1
            x2, y2 = p2
            
            # Check if edge is roughly horizontal and colinear with top boundary
            if (abs(y1 - y_min) <= tolerance and abs(y2 - y_min) <= tolerance and
                abs(y1 - y2) <= tolerance):
                return True
            
            # Check if edge is roughly horizontal and colinear with bottom boundary  
            if (abs(y1 - y_max) <= tolerance and abs(y2 - y_max) <= tolerance and
                abs(y1 - y2) <= tolerance):
                return True
            
            # Check if edge is roughly vertical and colinear with left boundary
            if (abs(x1 - x_min) <= tolerance and abs(x2 - x_min) <= tolerance and
                abs(x1 - x2) <= tolerance):
                return True
            
            # Check if edge is roughly vertical and colinear with right boundary
            if (abs(x1 - x_max) <= tolerance and abs(x2 - x_max) <= tolerance and
                abs(x1 - x2) <= tolerance):
                return True
            
            return False
        
        def generate_edge_sample_points(p1, p2, num_points=10, margin_px=10):
            """Generate equally spaced points along an edge, leaving a fixed margin at each end.
            Always generates at least one point in the center of the line."""
            # Calculate edge length
            edge_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Always generate center point
            center_x = p1[0] + 0.5 * (p2[0] - p1[0])
            center_y = p1[1] + 0.5 * (p2[1] - p1[1])
            
            # If edge is too short to accommodate margins, return just the center point
            if edge_length <= 2 * margin_px:
                return [(center_x, center_y)]
            
            # Calculate t values for the start and end of the usable region
            t_start = margin_px / edge_length
            t_end = 1.0 - margin_px / edge_length
            
            points = []
            
            # If only one point requested, return center point
            if num_points == 1:
                return [(center_x, center_y)]
            
            # Generate points evenly spaced within the usable region
            for i in range(num_points):
                # Distribute points evenly within the usable region
                t_local = i / (num_points - 1)
                t = t_start + t_local * (t_end - t_start)
                
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                points.append((x, y))
            
            return points
        
        def point_in_polygon(point, polygon):
            """Check if a point is inside a polygon using OpenCV."""
            if len(polygon) < 3:
                return False
            # Convert polygon to the format expected by cv2.pointPolygonTest
            poly_points = polygon.astype(np.float32).reshape((-1, 1, 2))
            return cv2.pointPolygonTest(poly_points, point, True) >= -merge_tolerance
        
        # Process each tile
        for tile_idx, (tile_result, tile_pos) in enumerate(zip(tile_results, positions)):
            x, y, x_end, y_end = tile_pos
            tile_width = x_end - x
            tile_height = y_end - y
            tile_bounds = (0, 0, tile_width, tile_height)  # tile local coordinates
            
            polygons = tile_result["polygons"]
            polygon_valid = tile_result["polygon_valid"]
            
            valid_polygons_count = sum(polygon_valid)
            log(f"Tile {tile_idx}: Processing {valid_polygons_count} valid polygons for edge validation")
            
            # Check each polygon in this tile (only those still valid after overlap removal)
            for poly_idx, polygon in enumerate(polygons):
                # Skip polygons already rejected for overlap
                if not polygon_valid[poly_idx]:
                    log(f"Tile {tile_idx}, Polygon {poly_idx}: SKIPPED (already invalid from overlap check)")
                    continue
                
                if len(polygon) < 3:
                    polygon_valid[poly_idx] = False
                    log(f"Tile {tile_idx}, Polygon {poly_idx}: INVALID (< 3 vertices)")
                    continue
                
                # Find edges that are near tile boundaries
                boundary_edges = []
                for i in range(len(polygon) - 1):
                    p1 = polygon[i]
                    p2 = polygon[i + 1]
                    
                    if is_edge_near_tile_boundary(p1, p2, tile_bounds):
                        boundary_edges.append((p1, p2))
                
                # If no boundary edges, polygon is valid (not on tile boundary)
                if not boundary_edges:
                    log(f"Tile {tile_idx}, Polygon {poly_idx}: VALID (no boundary edges)")
                    continue
                
                log(f"Tile {tile_idx}, Polygon {poly_idx}: Found {len(boundary_edges)} boundary edges, validating...")
                
                # Check sample points along boundary edges
                polygon_is_valid = True
                
                for edge_idx, (p1, p2) in enumerate(boundary_edges):
                    sample_points = generate_edge_sample_points(p1, p2)
                    
                    # Determine if this edge is horizontal or vertical
                    is_horizontal_edge = abs(p1[1] - p2[1]) <= 2  # Edge is roughly horizontal
                    is_vertical_edge = abs(p1[0] - p2[0]) <= 2    # Edge is roughly vertical
                    
                    edge_type = "horizontal" if is_horizontal_edge else "vertical" if is_vertical_edge else "diagonal"
                    log(f"Tile {tile_idx}, Polygon {poly_idx}, Edge {edge_idx}: {edge_type} edge with {len(sample_points)} sample points")
                    
                    # Convert sample points to global image coordinates
                    global_sample_points = [(px + x, py + y) for px, py in sample_points]
                    
                    # Check if each sample point is contained in any polygon from other tiles
                    for point_idx, global_point in enumerate(global_sample_points):
                        point_found_in_other_polygon = False
                        match_info = None
                        
                        # Check all other tiles
                        for other_tile_result, other_tile_pos in zip(tile_results, positions):
                            if other_tile_result is tile_result:
                                continue
                            
                            other_x, other_y, other_x_end, other_y_end = other_tile_pos
                            
                            # Skip tiles in same row for horizontal edges
                            if is_horizontal_edge and other_y == y:
                                continue
                            
                            # Skip tiles in same column for vertical edges  
                            if is_vertical_edge and other_x == x:
                                continue
                            
                            # Convert global point to other tile's local coordinates
                            local_point = (global_point[0] - other_x, global_point[1] - other_y)
                            
                            # Check if point is inside any valid polygon in this other tile
                            for other_poly_idx, other_polygon in enumerate(other_tile_result["polygons"]):
                                # Only consider polygons that are still valid (not rejected for overlap)
                                if not other_tile_result["polygon_valid"][other_poly_idx]:
                                    continue
                                
                                if point_in_polygon(local_point, other_polygon):
                                    point_found_in_other_polygon = True
                                    # Store match location information for detailed logging
                                    match_tile_idx = None
                                    for search_tile_idx, (search_tile_result, search_tile_pos) in enumerate(zip(tile_results, positions)):
                                        if search_tile_result is other_tile_result:
                                            match_tile_idx = search_tile_idx
                                            break
                                    
                                    match_info = {
                                        'tile_idx': match_tile_idx,
                                        'polygon_idx': other_poly_idx,
                                        'tile_bounds': (other_x, other_y, other_x_end, other_y_end),
                                        'local_point': local_point,
                                        'global_point': global_point
                                    }
                                    break
                            
                            if point_found_in_other_polygon:
                                break
                    
                        # If any sample point is not found in other polygons, mark as invalid
                        if not point_found_in_other_polygon:
                            polygon_is_valid = False
                            log(f"Tile {tile_idx}, Polygon {poly_idx}: Sample point {point_idx} at {global_point} NOT found in other polygons")
                            break
                        else:
                            # Get the matched polygon points and convert to global coordinates
                            matched_polygon = tile_results[match_info['tile_idx']]["polygons"][match_info['polygon_idx']]
                            match_tile_bounds = match_info['tile_bounds']
                            # Convert from local tile coordinates to global coordinates
                            global_polygon_points = matched_polygon + np.array([match_tile_bounds[0], match_tile_bounds[1]])
                            # Format points as list of [x, y] coordinates for logging
                            polygon_points_str = '[' + ', '.join(f'[{x:.1f}, {y:.1f}]' for x, y in global_polygon_points) + ']'
                            
                            log(f"Tile {tile_idx}, Polygon {poly_idx}: Sample point {point_idx} at {global_point} found in "
                                f"Tile {match_info['tile_idx']}, Polygon {match_info['polygon_idx']} "
                                f"(polygon points: {polygon_points_str})")
                    
                    if not polygon_is_valid:
                        break
                
                # Update polygon validity
                polygon_valid[poly_idx] = polygon_is_valid
                
                if polygon_is_valid:
                    log(f"Tile {tile_idx}, Polygon {poly_idx}: VALID (all boundary edge sample points found in other polygons)")
                else:
                    edge_rejected_count += 1
                    log(f"Tile {tile_idx}, Polygon {poly_idx}: INVALID (boundary edge validation failed)")
        
        # Log final validation summary
        final_valid_count = sum(sum(tile_result["polygon_valid"]) for tile_result in tile_results)
        log(f"Phase 2 complete: {edge_rejected_count} polygons removed due to edge validation")
        log(f"Validation summary: {final_valid_count}/{total_polygons} polygons remain valid")
        
        return tile_results

    def _merge_polygons(
        self,
        tile_results: List[Dict[str, List[np.ndarray]]],
        positions: List[Tuple[int, int, int, int]],
        image_height: int,
        image_width: int,
        debug: bool = False,
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
            polygon_valid = tile_result["polygon_valid"]
            
            for poly_idx, (poly, is_valid) in enumerate(zip(tile_polygons, polygon_valid)):
                # Skip invalid polygons
                if not is_valid:
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
        
        kernel_size = 32
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        bitmap = cv2.morphologyEx(bitmap, cv2.MORPH_CLOSE, kernel)
        
        # Save bitmap for debugging (optional)
        if debug:
            cv2.imwrite('debug_polygon_bitmap.png', bitmap)
            log("Saved debug bitmap to debug_polygon_bitmap.png")
        
        # Find contours in the bitmap
        contours, _ = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Collect all valid contours into shapely polygons
        shapely_polygons = []
        
        for contour in contours:
            # Skip very small contours (area is scaled by scale_factor^2)
            area = cv2.contourArea(contour)
            if area < CFG.MIN_POLYGON_AREA * (scale_factor ** 2):
                continue
            
            # Convert contour to Shapely Polygon
            contour_points = contour.reshape(-1, 2).astype(np.float64)
            shapely_polygon = Polygon(contour_points)
            shapely_polygons.append(shapely_polygon)
        
        # Initialize merged_polygons to avoid NameError if no valid contours are found
        merged_polygons: List[np.ndarray] = []
        
        # Create single GeoDataFrame with all polygons and regularize them all at once
        if shapely_polygons:
            gdf = gpd.GeoDataFrame({'geometry': shapely_polygons})
            regularized_gdf = regularize_geodataframe(gdf, simplify_tolerance=20, parallel_threshold=100)
            
            # Process the regularized polygons
            for regularized_polygon in regularized_gdf.geometry:
                # Convert back to numpy array for OpenCV format
                coords = np.array(regularized_polygon.exterior.coords[:-1])  # Remove duplicate last point
                
                # Convert from OpenCV format to our polygon format
                if len(coords) >= 3:  # Valid polygon needs at least 3 points
                    # Scale down coordinates back to original image coordinate system
                    polygon_coords = coords.astype(np.float32) / scale_factor
                    
                    merged_polygons.append(polygon_coords)
        
        log(f"Bitmap approach: {len(merged_polygons)} polygons extracted from bitmap")
        return merged_polygons

    def infer(self, image_data: bytes, debug: bool = False, merge_tolerance: Optional[float] = None, tile_overlap_ratio: Optional[float] = None) -> List[List[List[float]]]:
        """Infer polygons in an image.

        Args:
            image_data (bytes): Raw image data
            debug (bool): Whether to save debug images (tile visualization and bitmap)
            merge_tolerance (Optional[float]): Tolerance for point-in-polygon tests during validation (in pixels, allows points to be slightly outside). If None, uses CFG.MERGE_TOLERANCE
            tile_overlap_ratio (Optional[float]): Overlap ratio between tiles (0.0 = no overlap, 1.0 = complete overlap). If None, uses CFG.TILE_OVERLAP_RATIO

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

        # Use provided parameters or fall back to config defaults
        effective_merge_tolerance = merge_tolerance if merge_tolerance is not None else CFG.MERGE_TOLERANCE
        effective_tile_overlap_ratio = tile_overlap_ratio if tile_overlap_ratio is not None else CFG.TILE_OVERLAP_RATIO

        overlap_ratio = effective_tile_overlap_ratio

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

        log(f"Total number of tiles to process: {len(tiles)}")

        # Process tiles in batches
        all_results: List[Dict[str, List[np.ndarray]]] = []

        for i in range(0, len(tiles), CFG.PREDICTION_BATCH_SIZE):
            batch_start_time = time.time()
            batch_tiles = tiles[i : i + CFG.PREDICTION_BATCH_SIZE]
            batch_results = self._process_tiles_batch(batch_tiles, debug)
            all_results.extend(batch_results)
            
            batch_time = time.time() - batch_start_time
            log(f"Processed batch of {len(batch_tiles)} tiles: {batch_time/len(batch_tiles):.3f}s per tile")

        # Validate all polygons and add validation attributes
        all_results = self._validate_all_polygons(all_results, bboxes, height, width, effective_merge_tolerance)

        # Create tile visualization
        if debug:
            self._create_tile_visualization(tiles, all_results, bboxes)

        merged_polygons = self._merge_polygons(all_results, bboxes, height, width, debug)

        # Convert to list format
        polygons_list = [poly.tolist() for poly in merged_polygons]
        # Round coordinates to two decimal places
        polygons_list = [
            [[round(x, 2), round(y, 2)] for x, y in polygon]
            for polygon in polygons_list
        ]

        return polygons_list
