# """
# Project YOLO 2D Masks to 3D Global Coordinates

# Transforms 2D mask polygons to 3D points using:
# - Upsampled depth maps (RGB resolution)
# - Camera poses from SfM
# - Camera intrinsics

# Usage:
#     python -m src.project_masks_to_3d \
#         --masks-dir data/yolo_masks \
#         --poses-json data/sfm/poses.json \
#         --depth-dir data/depth_upsampled \
#         --calib-rgb calib/rgb_camera_info.json \
#         --output outputs/masks_3d.json
# """

# import numpy as np
# import json
# import cv2
# import logging
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional
# from tqdm import tqdm

# logger = logging.getLogger(__name__)


# def load_depth_image(depth_path: str) -> np.ndarray:
#     """
#     Load depth image (PNG 16-bit, mm units)

#     Returns:
#         depth_map: (H, W) in meters
#     """
#     depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

#     if depth_mm is None:
#         raise FileNotFoundError(f"Depth image not found: {depth_path}")

#     # mm → m
#     depth_m = depth_mm.astype(np.float32) / 1000.0

#     return depth_m


# def get_nearest_valid_depth(depth_map: np.ndarray, u: int, v: int, max_search_radius: int = 10) -> float:
#     """
#     Get depth value at (u, v), or nearest valid depth if invalid.

#     Args:
#         depth_map: Depth map (H, W) in meters
#         u, v: Pixel coordinates
#         max_search_radius: Maximum search radius in pixels

#     Returns:
#         depth: Valid depth value, or 0.0 if not found
#     """
#     H, W = depth_map.shape

#     # Clamp to image bounds
#     v = max(0, min(H - 1, v))
#     u = max(0, min(W - 1, u))

#     # Check direct pixel
#     depth = depth_map[v, u]

#     if depth > 0 and depth < 10.0:  # Valid depth (0-10m)
#         return depth

#     # Search in expanding radius
#     for radius in range(1, max_search_radius + 1):
#         # Search in square around (u, v)
#         for dv in range(-radius, radius + 1):
#             for du in range(-radius, radius + 1):
#                 # Only check border of square (not interior)
#                 if abs(dv) != radius and abs(du) != radius:
#                     continue

#                 v_search = v + dv
#                 u_search = u + du

#                 # Check bounds
#                 if not (0 <= v_search < H and 0 <= u_search < W):
#                     continue

#                 depth_search = depth_map[v_search, u_search]

#                 if depth_search > 0 and depth_search < 10.0:
#                     return depth_search

#     # No valid depth found
#     return 0.0


# def sample_polygon_boundary(polygon: List[Tuple[float, float]], spacing: int = 5) -> List[Tuple[int, int]]:
#     """
#     Sample polygon boundary uniformly.

#     Args:
#         polygon: List of vertices [(u1, v1), (u2, v2), ...]
#         spacing: Pixel spacing between samples

#     Returns:
#         sampled_pixels: List of (u, v) pixel coordinates
#     """
#     if len(polygon) < 2:
#         return []

#     sampled = []

#     for i in range(len(polygon)):
#         p1 = polygon[i]
#         p2 = polygon[(i + 1) % len(polygon)]

#         # Bresenham line
#         x1, y1 = int(p1[0]), int(p1[1])
#         x2, y2 = int(p2[0]), int(p2[1])

#         line_pixels = bresenham_line(x1, y1, x2, y2)

#         # Sample every 'spacing' pixels
#         for j in range(0, len(line_pixels), spacing):
#             sampled.append(line_pixels[j])

#     return sampled


# def bresenham_line(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
#     """Bresenham line drawing algorithm"""
#     pixels = []
#     dx = abs(x2 - x1)
#     dy = abs(y2 - y1)
#     sx = 1 if x1 < x2 else -1
#     sy = 1 if y1 < y2 else -1
#     err = dx - dy

#     while True:
#         pixels.append((x1, y1))

#         if x1 == x2 and y1 == y2:
#             break

#         e2 = 2 * err
#         if e2 > -dy:
#             err -= dy
#             x1 += sx
#         if e2 < dx:
#             err += dx
#             y1 += sy

#     return pixels


# def backproject_to_camera(u: float, v: float, depth: float, K: np.ndarray) -> np.ndarray:
#     """
#     Backproject pixel to 3D camera coordinates.

#     Args:
#         u, v: Pixel coordinates
#         depth: Depth value (meters)
#         K: Camera intrinsic matrix (3x3)

#     Returns:
#         P_cam: 3D point in camera coordinates [X, Y, Z]
#     """
#     fx, fy = K[0, 0], K[1, 1]
#     cx, cy = K[0, 2], K[1, 2]

#     X_cam = (u - cx) * depth / fx
#     Y_cam = (v - cy) * depth / fy
#     Z_cam = depth

#     return np.array([X_cam, Y_cam, Z_cam])


# def transform_to_world(P_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
#     """
#     Transform point from camera to world coordinates.

#     Args:
#         P_cam: 3D point in camera coordinates (3,)
#         R: Rotation matrix, World to Camera (3x3)
#         t: Translation vector, Camera in World (3,)

#     Returns:
#         P_world: 3D point in world coordinates (3,)
#     """
#     # Camera to World: R^T @ (P_cam - t)
#     P_world = R.T @ (P_cam - t)

#     return P_world


# def project_mask_to_3d(
#     mask: Dict,
#     depth_map: np.ndarray,
#     pose: Dict,
#     K: np.ndarray,
#     image_id: str,
#     mask_id: int,
#     config: Dict
# ) -> Optional[Dict]:
#     """
#     Project single mask to 3D.

#     Args:
#         mask: YOLO mask dict with 'polygon', 'class', 'score'
#         depth_map: Depth map (H, W) in meters
#         pose: Camera pose dict with 'R', 't'
#         K: Camera intrinsic matrix
#         image_id: Image identifier
#         mask_id: Mask index in image
#         config: Configuration dict

#     Returns:
#         mask_3d: 3D mask dict, or None if insufficient valid points
#     """
#     polygon = mask['polygon']

#     # Sample polygon boundary
#     sampled_pixels = sample_polygon_boundary(
#         polygon,
#         spacing=config.get('polygon_sampling_spacing', 5)
#     )

#     if len(sampled_pixels) == 0:
#         logger.warning(f"Mask {image_id}/{mask_id}: Empty polygon")
#         return None

#     # Backproject to 3D
#     R = np.array(pose['R'])
#     t = np.array(pose['t']).flatten()  # Ensure (3,) shape

#     points_3d_world = []
#     valid_count = 0
#     fallback_count = 0

#     H, W = depth_map.shape

#     for (u, v) in sampled_pixels:
#         # Skip out-of-bounds pixels (polygon may extend beyond image)
#         if not (0 <= v < H and 0 <= u < W):
#             continue

#         # Get depth (with nearest fallback)
#         depth = get_nearest_valid_depth(
#             depth_map,
#             u, v,
#             max_search_radius=config.get('depth_search_radius', 10)
#         )

#         if depth <= 0:
#             continue

#         valid_count += 1

#         # Check if fallback was used (original pixel had no depth)
#         if depth_map[v, u] <= 0:
#             fallback_count += 1

#         # Backproject
#         P_cam = backproject_to_camera(u, v, depth, K)

#         # Transform to world
#         P_world = transform_to_world(P_cam, R, t)

#         points_3d_world.append(P_world)

#     # Quality check
#     min_valid_points = config.get('min_valid_points', 5)

#     if len(points_3d_world) < min_valid_points:
#         logger.debug(
#             f"Mask {image_id}/{mask_id}: Only {len(points_3d_world)} valid points "
#             f"(min={min_valid_points}), skipping"
#         )
#         return None

#     points_3d_world = np.array(points_3d_world)

#     # Compute metadata
#     centroid_3d = np.mean(points_3d_world, axis=0)
#     bbox_min = np.min(points_3d_world, axis=0)
#     bbox_max = np.max(points_3d_world, axis=0)

#     coverage = valid_count / len(sampled_pixels)
#     fallback_ratio = fallback_count / valid_count if valid_count > 0 else 0.0

#     # Warnings
#     if coverage < config.get('warn_low_coverage', 0.25):
#         logger.warning(
#             f"Mask {image_id}/{mask_id}: Low coverage {coverage*100:.1f}% "
#             f"({valid_count}/{len(sampled_pixels)} pixels)"
#         )

#     if fallback_ratio > 0.5:
#         logger.debug(
#             f"Mask {image_id}/{mask_id}: High fallback ratio {fallback_ratio*100:.1f}% "
#             f"({fallback_count}/{valid_count} pixels)"
#         )

#     # Create 3D mask
#     mask_3d = {
#         'image_id': image_id,
#         'mask_id': mask_id,
#         'class': mask['class'],
#         'confidence': mask['score'],
#         'points_3d': points_3d_world.tolist(),
#         'centroid_3d': centroid_3d.tolist(),
#         'bbox_3d': {
#             'min': bbox_min.tolist(),
#             'max': bbox_max.tolist()
#         },
#         'n_sampled': len(sampled_pixels),
#         'n_valid': valid_count,
#         'n_fallback': fallback_count,
#         'coverage': coverage,
#         'fallback_ratio': fallback_ratio
#     }

#     return mask_3d


# def run_projection(
#     masks_dir: str,
#     poses_json: str,
#     depth_dir: str,
#     calib_rgb: str,
#     output_json: str,
#     config: Optional[Dict] = None
# ):
#     """
#     Project all YOLO masks to 3D.

#     Args:
#         masks_dir: Directory with YOLO mask JSONs
#         poses_json: SfM poses JSON
#         depth_dir: Directory with upsampled depth images
#         calib_rgb: RGB camera calibration JSON
#         output_json: Output masks_3d JSON path
#         config: Configuration dict
#     """
#     if config is None:
#         config = {
#             'polygon_sampling_spacing': 5,
#             'depth_search_radius': 10,
#             'min_valid_points': 5,
#             'warn_low_coverage': 0.25
#         }

#     logger.info("=" * 80)
#     logger.info("Project YOLO Masks to 3D")
#     logger.info("=" * 80)

#     # Load poses
#     logger.info(f"Loading poses: {poses_json}")
#     with open(poses_json, 'r') as f:
#         poses = json.load(f)
#     logger.info(f"  Loaded {len(poses)} poses")

#     # Load camera calibration
#     logger.info(f"Loading RGB calibration: {calib_rgb}")
#     with open(calib_rgb, 'r') as f:
#         calib = json.load(f)

#     K = np.array(calib['K'])
#     logger.info(f"  Camera matrix K:\n{K}")

#     # Process all masks
#     masks_dir_path = Path(masks_dir)
#     mask_files = sorted(masks_dir_path.glob("*.json"))

#     logger.info(f"Found {len(mask_files)} mask files")

#     all_masks_3d = []
#     skipped_count = 0
#     total_masks = 0

#     for mask_file in tqdm(mask_files, desc="Processing masks"):
#         image_id = mask_file.stem

#         # Check if pose exists (poses.json may use .png extension)
#         image_key = image_id
#         if image_id not in poses:
#             # Try with .png extension
#             image_key = f"{image_id}.png"
#             if image_key not in poses:
#                 logger.warning(f"No pose for {image_id}, skipping")
#                 continue

#         pose = poses[image_key]

#         # Load YOLO masks
#         with open(mask_file, 'r') as f:
#             yolo_data = json.load(f)

#         masks = yolo_data.get('masks', [])
#         total_masks += len(masks)

#         # Load depth map
#         depth_filename = image_id.replace('camera_RGB_', 'camera_DPT_') + '.png'
#         depth_path = Path(depth_dir) / depth_filename

#         if not depth_path.exists():
#             logger.warning(f"Depth not found: {depth_path}, skipping image")
#             skipped_count += len(masks)
#             continue

#         depth_map = load_depth_image(str(depth_path))

#         # Process each mask
#         for mask_idx, mask in enumerate(masks):
#             mask_3d = project_mask_to_3d(
#                 mask,
#                 depth_map,
#                 pose,
#                 K,
#                 image_id,
#                 mask_idx,
#                 config
#             )

#             if mask_3d is not None:
#                 all_masks_3d.append(mask_3d)
#             else:
#                 skipped_count += 1

#     # Statistics
#     logger.info("=" * 80)
#     logger.info("Projection Statistics")
#     logger.info("=" * 80)
#     logger.info(f"  Total masks: {total_masks}")
#     logger.info(f"  Valid 3D masks: {len(all_masks_3d)}")

#     if total_masks > 0:
#         logger.info(f"  Skipped: {skipped_count} ({skipped_count/total_masks*100:.1f}%)")
#     else:
#         logger.info(f"  Skipped: {skipped_count} (0.0%)")

#     # Coverage statistics
#     if len(all_masks_3d) > 0:
#         coverages = [m['coverage'] for m in all_masks_3d]
#         fallback_ratios = [m['fallback_ratio'] for m in all_masks_3d]

#         logger.info(f"  Mean coverage: {np.mean(coverages)*100:.1f}%")
#         logger.info(f"  Median coverage: {np.median(coverages)*100:.1f}%")
#         logger.info(f"  Mean fallback ratio: {np.mean(fallback_ratios)*100:.1f}%")
#     else:
#         logger.warning("  No valid 3D masks generated!")

#     # Save
#     output_path = Path(output_json)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     result = {
#         'metadata': {
#             'total_input_masks': total_masks,
#             'valid_3d_masks': len(all_masks_3d),
#             'skipped_masks': skipped_count,
#             'mean_coverage': float(np.mean(coverages)) if len(all_masks_3d) > 0 else 0.0,
#             'config': config
#         },
#         'masks': all_masks_3d
#     }

#     with open(output_json, 'w') as f:
#         json.dump(result, f, indent=2)

#     logger.info(f"Saved 3D masks: {output_json}")
#     logger.info("=" * 80)

#     return all_masks_3d


# if __name__ == '__main__':
#     import argparse
#     from .utils import setup_logging

#     parser = argparse.ArgumentParser(description='Project YOLO masks to 3D')
#     parser.add_argument('--masks-dir', required=True,
#                        help='YOLO masks directory')
#     parser.add_argument('--poses-json', required=True,
#                        help='SfM poses JSON')
#     parser.add_argument('--depth-dir', required=True,
#                        help='Upsampled depth directory (RGB resolution)')
#     parser.add_argument('--calib-rgb', required=True,
#                        help='RGB camera calibration JSON')
#     parser.add_argument('--output', required=True,
#                        help='Output masks_3d JSON')
#     parser.add_argument('--polygon-spacing', type=int, default=5,
#                        help='Polygon sampling spacing (pixels)')
#     parser.add_argument('--depth-search-radius', type=int, default=10,
#                        help='Nearest depth search radius (pixels)')
#     parser.add_argument('--min-valid-points', type=int, default=5,
#                        help='Minimum valid 3D points')
#     parser.add_argument('--log-level', default='INFO',
#                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

#     args = parser.parse_args()

#     setup_logging(args.log_level)

#     config = {
#         'polygon_sampling_spacing': args.polygon_spacing,
#         'depth_search_radius': args.depth_search_radius,
#         'min_valid_points': args.min_valid_points,
#         'warn_low_coverage': 0.25
#     }

#     try:
#         masks_3d = run_projection(
#             args.masks_dir,
#             args.poses_json,
#             args.depth_dir,
#             args.calib_rgb,
#             args.output,
#             config
#         )

#         print(f"\n✅ Projection complete!")
#         print(f"   Valid 3D masks: {len(masks_3d)}")
#         print(f"   Output: {args.output}")

#     except Exception as e:
#         logger.error(f"Projection failed: {e}", exc_info=True)
#         import sys
#         sys.exit(1)



"""
Project YOLO 2D Masks to 3D Global Coordinates

Transforms 2D mask polygons to 3D points using:
- Upsampled depth maps (RGB resolution)
- Camera poses from SfM
- Camera intrinsics

Usage:
    python -m src.project_masks_to_3d \
        --masks-dir data/yolo_masks \
        --poses-json data/sfm/poses.json \
        --depth-dir data/depth_upsampled \
        --calib-rgb calib/rgb_camera_info.json \
        --output outputs/masks_3d.json
"""

import numpy as np
import json
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_depth_image(depth_path: str) -> np.ndarray:
    """
    Load depth image (PNG 16-bit, mm units)

    Returns:
        depth_map: (H, W) in meters
    """
    depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth_mm is None:
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    # mm → m
    depth_m = depth_mm.astype(np.float32) / 1000.0

    return depth_m


def get_nearest_valid_depth(depth_map: np.ndarray, u: int, v: int, max_search_radius: int = 10) -> float:
    """
    Get depth value at (u, v), or nearest valid depth if invalid.

    Args:
        depth_map: Depth map (H, W) in meters
        u, v: Pixel coordinates
        max_search_radius: Maximum search radius in pixels

    Returns:
        depth: Valid depth value, or 0.0 if not found
    """
    H, W = depth_map.shape

    # Clamp to image bounds
    v = max(0, min(H - 1, v))
    u = max(0, min(W - 1, u))

    # Check direct pixel
    depth = depth_map[v, u]

    if depth > 0 and depth < 10.0:  # Valid depth (0-10m)
        return depth

    # Search in expanding radius
    for radius in range(1, max_search_radius + 1):
        # Search in square around (u, v)
        for dv in range(-radius, radius + 1):
            for du in range(-radius, radius + 1):
                # Only check border of square (not interior)
                if abs(dv) != radius and abs(du) != radius:
                    continue

                v_search = v + dv
                u_search = u + du

                # Check bounds
                if not (0 <= v_search < H and 0 <= u_search < W):
                    continue

                depth_search = depth_map[v_search, u_search]

                if depth_search > 0 and depth_search < 10.0:
                    return depth_search

    # No valid depth found
    return 0.0


def sample_polygon_boundary(polygon: List[Tuple[float, float]], spacing: int = 5) -> List[Tuple[int, int]]:
    """
    Sample polygon boundary uniformly.

    Args:
        polygon: List of vertices [(u1, v1), (u2, v2), ...]
        spacing: Pixel spacing between samples

    Returns:
        sampled_pixels: List of (u, v) pixel coordinates
    """
    if len(polygon) < 2:
        return []

    sampled = []

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]

        # Bresenham line
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])

        line_pixels = bresenham_line(x1, y1, x2, y2)

        # Sample every 'spacing' pixels
        for j in range(0, len(line_pixels), spacing):
            sampled.append(line_pixels[j])

    return sampled


def bresenham_line(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    """Bresenham line drawing algorithm"""
    pixels = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        pixels.append((x1, y1))

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return pixels


def backproject_to_camera(u: float, v: float, depth: float, K: np.ndarray) -> np.ndarray:
    """
    Backproject pixel to 3D camera coordinates.

    Args:
        u, v: Pixel coordinates
        depth: Depth value (meters)
        K: Camera intrinsic matrix (3x3)

    Returns:
        P_cam: 3D point in camera coordinates [X, Y, Z]
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X_cam = (u - cx) * depth / fx
    Y_cam = (v - cy) * depth / fy
    Z_cam = depth

    return np.array([X_cam, Y_cam, Z_cam])


def transform_to_world(P_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Transform point from camera to world coordinates.

    Args:
        P_cam: 3D point in camera coordinates (3,)
        R: Rotation matrix, World to Camera (3x3)
        t: Translation vector, Camera in World (3,)

    Returns:
        P_world: 3D point in world coordinates (3,)
    """
    # Camera to World: R^T @ (P_cam - t)
    P_world = R.T @ (P_cam - t)

    return P_world


def project_mask_to_3d(
    mask: Dict,
    depth_map: np.ndarray,
    pose: Dict,
    K: np.ndarray,
    image_id: str,
    mask_id: int,
    config: Dict
) -> Optional[Dict]:
    """
    Project single mask to 3D.

    Args:
        mask: YOLO mask dict with 'polygon', 'class', 'score'
        depth_map: Depth map (H, W) in meters
        pose: Camera pose dict with 'R', 't'
        K: Camera intrinsic matrix
        image_id: Image identifier
        mask_id: Mask index in image
        config: Configuration dict

    Returns:
        mask_3d: 3D mask dict, or None if insufficient valid points
    """
    polygon = mask['polygon']

    # Sample polygon boundary
    sampled_pixels = sample_polygon_boundary(
        polygon,
        spacing=config.get('polygon_sampling_spacing', 5)
    )

    if len(sampled_pixels) == 0:
        logger.warning(f"Mask {image_id}/{mask_id}: Empty polygon")
        return None

    # Backproject to 3D
    R = np.array(pose['R'])
    t = np.array(pose['t']).flatten()  # Ensure (3,) shape

    points_3d_world = []
    valid_count = 0
    fallback_count = 0

    H, W = depth_map.shape

    for (u, v) in sampled_pixels:
        # Skip out-of-bounds pixels (polygon may extend beyond image)
        if not (0 <= v < H and 0 <= u < W):
            continue

        # Get depth (with nearest fallback)
        depth = get_nearest_valid_depth(
            depth_map,
            u, v,
            max_search_radius=config.get('depth_search_radius', 10)
        )

        if depth <= 0:
            continue

        valid_count += 1

        # Check if fallback was used (original pixel had no depth)
        if depth_map[v, u] <= 0:
            fallback_count += 1

        # Backproject
        P_cam = backproject_to_camera(u, v, depth, K)

        # Transform to world
        P_world = transform_to_world(P_cam, R, t)

        points_3d_world.append(P_world)

    # Quality check
    min_valid_points = config.get('min_valid_points', 5)

    if len(points_3d_world) < min_valid_points:
        logger.debug(
            f"Mask {image_id}/{mask_id}: Only {len(points_3d_world)} valid points "
            f"(min={min_valid_points}), skipping"
        )
        return None

    points_3d_world = np.array(points_3d_world)

    # Compute metadata
    centroid_3d = np.mean(points_3d_world, axis=0)
    bbox_min = np.min(points_3d_world, axis=0)
    bbox_max = np.max(points_3d_world, axis=0)

    coverage = valid_count / len(sampled_pixels)
    fallback_ratio = fallback_count / valid_count if valid_count > 0 else 0.0

    # Warnings
    if coverage < config.get('warn_low_coverage', 0.25):
        logger.warning(
            f"Mask {image_id}/{mask_id}: Low coverage {coverage*100:.1f}% "
            f"({valid_count}/{len(sampled_pixels)} pixels)"
        )

    if fallback_ratio > 0.5:
        logger.debug(
            f"Mask {image_id}/{mask_id}: High fallback ratio {fallback_ratio*100:.1f}% "
            f"({fallback_count}/{valid_count} pixels)"
        )

    # Create 3D mask
    mask_3d = {
        'image_id': image_id,
        'mask_id': mask_id,
        'class': mask['class'],
        'confidence': mask['score'],
        'points_3d': points_3d_world.tolist(),
        'centroid_3d': centroid_3d.tolist(),
        'bbox_3d': {
            'min': bbox_min.tolist(),
            'max': bbox_max.tolist()
        },
        'n_sampled': len(sampled_pixels),
        'n_valid': valid_count,
        'n_fallback': fallback_count,
        'coverage': coverage,
        'fallback_ratio': fallback_ratio
    }

    return mask_3d


def run_projection(
    masks_dir: str,
    poses_json: str,
    depth_dir: str,
    calib_rgb: str,
    output_json: str,
    config: Optional[Dict] = None
):
    """
    Project all YOLO masks to 3D.

    Args:
        masks_dir: Directory with YOLO mask JSONs
        poses_json: SfM poses JSON
        depth_dir: Directory with upsampled depth images
        calib_rgb: RGB camera calibration JSON
        output_json: Output masks_3d JSON path
        config: Configuration dict
    """
    if config is None:
        config = {
            'polygon_sampling_spacing': 5,
            'depth_search_radius': 10,
            'min_valid_points': 5,
            'warn_low_coverage': 0.25
        }

    logger.info("=" * 80)
    logger.info("Project YOLO Masks to 3D")
    logger.info("=" * 80)

    # Load poses
    logger.info(f"Loading poses: {poses_json}")
    with open(poses_json, 'r') as f:
        poses = json.load(f)
    logger.info(f"  Loaded {len(poses)} poses")

    # Load camera calibration
    logger.info(f"Loading RGB calibration: {calib_rgb}")
    with open(calib_rgb, 'r') as f:
        calib = json.load(f)

    K = np.array(calib['K'])
    logger.info(f"  Camera matrix K:\n{K}")

    # Process all masks
    masks_dir_path = Path(masks_dir)
    mask_files = sorted(masks_dir_path.glob("*.json"))

    logger.info(f"Found {len(mask_files)} mask files")

    all_masks_3d = []
    skipped_count = 0
    filtered_count = 0  # Class filtering
    total_masks = 0

    for mask_file in tqdm(mask_files, desc="Processing masks"):
        image_id = mask_file.stem

        # Check if pose exists (poses.json may use .png extension)
        image_key = image_id
        if image_id not in poses:
            # Try with .png extension
            image_key = f"{image_id}.png"
            if image_key not in poses:
                logger.warning(f"No pose for {image_id}, skipping")
                continue

        pose = poses[image_key]

        # Load YOLO masks
        with open(mask_file, 'r') as f:
            yolo_data = json.load(f)

        masks = yolo_data.get('masks', [])
        total_masks += len(masks)

        # Load depth map
        depth_filename = image_id.replace('camera_RGB_', 'camera_DPT_') + '.png'
        depth_path = Path(depth_dir) / depth_filename

        if not depth_path.exists():
            logger.warning(f"Depth not found: {depth_path}, skipping image")
            skipped_count += len(masks)
            continue

        depth_map = load_depth_image(str(depth_path))

        # Process each mask
        for mask_idx, mask in enumerate(masks):
            # Filter by class
            filter_classes = config.get('filter_classes', None)
            if filter_classes is not None:
                if mask['class'] not in filter_classes:
                    filtered_count += 1
                    continue

            mask_3d = project_mask_to_3d(
                mask,
                depth_map,
                pose,
                K,
                image_id,
                mask_idx,
                config
            )

            if mask_3d is not None:
                all_masks_3d.append(mask_3d)
            else:
                skipped_count += 1

    # Statistics
    logger.info("=" * 80)
    logger.info("Projection Statistics")
    logger.info("=" * 80)
    logger.info(f"  Total masks: {total_masks}")

    if filtered_count > 0:
        logger.info(f"  Filtered by class: {filtered_count} ({filtered_count/total_masks*100:.1f}%)")

    logger.info(f"  Valid 3D masks: {len(all_masks_3d)}")

    if total_masks > 0:
        logger.info(f"  Skipped (insufficient points): {skipped_count} ({skipped_count/total_masks*100:.1f}%)")
    else:
        logger.info(f"  Skipped: {skipped_count} (0.0%)")

    # Coverage statistics
    if len(all_masks_3d) > 0:
        coverages = [m['coverage'] for m in all_masks_3d]
        fallback_ratios = [m['fallback_ratio'] for m in all_masks_3d]

        logger.info(f"  Mean coverage: {np.mean(coverages)*100:.1f}%")
        logger.info(f"  Median coverage: {np.median(coverages)*100:.1f}%")
        logger.info(f"  Mean fallback ratio: {np.mean(fallback_ratios)*100:.1f}%")
    else:
        logger.warning("  No valid 3D masks generated!")

    # Save
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        'metadata': {
            'total_input_masks': total_masks,
            'filtered_masks': filtered_count,
            'valid_3d_masks': len(all_masks_3d),
            'skipped_masks': skipped_count,
            'mean_coverage': float(np.mean(coverages)) if len(all_masks_3d) > 0 else 0.0,
            'config': config
        },
        'masks': all_masks_3d
    }

    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved 3D masks: {output_json}")
    logger.info("=" * 80)

    return all_masks_3d


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='Project YOLO masks to 3D')
    parser.add_argument('--masks-dir', required=True,
                       help='YOLO masks directory')
    parser.add_argument('--poses-json', required=True,
                       help='SfM poses JSON')
    parser.add_argument('--depth-dir', required=True,
                       help='Upsampled depth directory (RGB resolution)')
    parser.add_argument('--calib-rgb', required=True,
                       help='RGB camera calibration JSON')
    parser.add_argument('--output', required=True,
                       help='Output masks_3d JSON')
    parser.add_argument('--polygon-spacing', type=int, default=5,
                       help='Polygon sampling spacing (pixels)')
    parser.add_argument('--depth-search-radius', type=int, default=10,
                       help='Nearest depth search radius (pixels)')
    parser.add_argument('--min-valid-points', type=int, default=5,
                       help='Minimum valid 3D points')
    parser.add_argument('--filter-classes', nargs='+', default=['crack'],
                       help='Classes to process (default: crack only). Use "all" to process all classes.')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    # Handle filter_classes
    filter_classes = None  # Process all classes
    if args.filter_classes and args.filter_classes != ['all']:
        filter_classes = args.filter_classes
        logger.info(f"Class filtering enabled: {filter_classes}")

    config = {
        'polygon_sampling_spacing': args.polygon_spacing,
        'depth_search_radius': args.depth_search_radius,
        'min_valid_points': args.min_valid_points,
        'warn_low_coverage': 0.25,
        'filter_classes': filter_classes
    }

    try:
        masks_3d = run_projection(
            args.masks_dir,
            args.poses_json,
            args.depth_dir,
            args.calib_rgb,
            args.output,
            config
        )

        print(f"\n✅ Projection complete!")
        print(f"   Valid 3D masks: {len(masks_3d)}")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Projection failed: {e}", exc_info=True)
        import sys
        sys.exit(1)