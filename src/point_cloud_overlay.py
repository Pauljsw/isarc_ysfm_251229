# """
# Point Cloud Mask Overlay

# Overlays YOLO crack masks onto SFM sparse point cloud.
# Uses COLMAP track information to map 2D pixels to 3D points.

# Usage:
#     python -m src.point_cloud_overlay \
#         --sparse-dir data/sfm/sparse/0 \
#         --masks-dir data/yolo_masks \
#         --output outputs/sfm_masked_cloud.ply
# """
# import numpy as np
# import json
# import logging
# from pathlib import Path
# from typing import Dict, List, Tuple
# from shapely.geometry import Point as ShapelyPoint
# from shapely.geometry import Polygon

# from .colmap_io import read_points3D_binary, read_images_binary, read_cameras_binary

# logger = logging.getLogger(__name__)


# def load_yolo_mask(mask_path: Path) -> Dict:
#     """Load YOLO mask JSON"""
#     with open(mask_path, 'r') as f:
#         return json.load(f)


# def is_pixel_in_crack_mask(
#     mask_json: Dict,
#     pixel_xy: Tuple[float, float],
#     image_stem: str = None,
#     min_confidence: float = 0.0
# ) -> Tuple[bool, float]:
#     """
#     Check if pixel is inside any crack polygon with confidence validation.

#     Args:
#         mask_json: YOLO mask JSON data
#         pixel_xy: Pixel coordinates (u, v)
#         image_stem: Image identifier for logging (optional)
#         min_confidence: Minimum YOLO confidence to consider (default: 0.0)

#     Returns:
#         (is_inside, confidence): True if pixel is in valid crack mask, and confidence score
#     """
#     if 'masks' not in mask_json:
#         return False, 0.0

#     point = ShapelyPoint(pixel_xy)

#     for mask_idx, mask in enumerate(mask_json['masks']):
#         if mask['class'] != 'crack':
#             continue

#         confidence = mask.get('score', 0.0)

#         # Filter by minimum confidence
#         if confidence < min_confidence:
#             continue

#         polygon_coords = mask['polygon']

#         # Validate polygon has enough points
#         if len(polygon_coords) < 3:
#             if image_stem:
#                 logger.debug(f"[{image_stem}] Mask {mask_idx}: < 3 points, skipping")
#             continue

#         try:
#             poly = Polygon(polygon_coords)

#             # Auto-fix invalid polygons with buffer(0) trick
#             if not poly.is_valid:
#                 try:
#                     fixed_poly = poly.buffer(0)
                    
#                     # 🆕 Verify the fix was successful
#                     if not fixed_poly.is_valid or fixed_poly.is_empty:
#                         if image_stem:
#                             logger.debug(
#                                 f"[{image_stem}] Mask {mask_idx}: Cannot fix invalid polygon, skipping"
#                             )
#                         continue
                    
#                     # Use fixed polygon
#                     poly = fixed_poly
                    
#                     if image_stem:
#                         logger.debug(
#                             f"[{image_stem}] Mask {mask_idx}: Auto-fixed invalid polygon "
#                             f"(area={poly.area:.2f}, confidence={confidence:.3f})"
#                         )
                        
#                 except Exception as e:
#                     # If buffer(0) fails, skip this polygon
#                     if image_stem:
#                         logger.debug(
#                             f"[{image_stem}] Mask {mask_idx}: Buffer fix failed ({e}), skipping"
#                         )
#                     continue

#             # Check if polygon is too small (degenerate)
#             if poly.area < 1.0:
#                 if image_stem:
#                     logger.debug(
#                         f"[{image_stem}] Mask {mask_idx}: Area too small ({poly.area:.2f}px²)"
#                     )
#                 continue

#             # 🆕 Final safety check before contains()
#             if not poly.is_valid:
#                 if image_stem:
#                     logger.warning(
#                         f"[{image_stem}] Mask {mask_idx}: Still invalid after fix, skipping"
#                     )
#                 continue

#             if poly.contains(point):
#                 return True, confidence

#         except Exception as e:
#             if image_stem:
#                 logger.error(
#                     f"[{image_stem}] Mask {mask_idx}: Unexpected error - {e}"
#                 )
#             continue

#     return False, 0.0


# def overlay_masks_on_pointcloud(
#     sparse_dir: str,
#     masks_dir: str,
#     output_ply: str,
#     crack_color: Tuple[int, int, int] = (255, 0, 0),
#     min_track_length: int = 2,
#     vote_threshold: float = 0.5,
#     min_confidence: float = 0.25
# ):
#     """
#     Overlay YOLO masks on SFM point cloud with improved voting mechanism.

#     Args:
#         sparse_dir: COLMAP sparse/0 directory
#         masks_dir: YOLO masks directory
#         output_ply: Output PLY path
#         crack_color: RGB color for crack points (default: red)
#         min_track_length: Minimum track length to include point
#         vote_threshold: Minimum ratio of views that must agree (default: 0.5 = majority)
#         min_confidence: Minimum YOLO confidence to consider (default: 0.25)
#     """
#     logger.info("=" * 80)
#     logger.info("Point Cloud Mask Overlay")
#     logger.info("=" * 80)

#     sparse_path = Path(sparse_dir)
#     masks_path = Path(masks_dir)

#     # Read COLMAP data
#     logger.info(f"Reading COLMAP data from: {sparse_dir}")
#     points3D = read_points3D_binary(str(sparse_path / "points3D.bin"))
#     images = read_images_binary(str(sparse_path / "images.bin"))
#     cameras = read_cameras_binary(str(sparse_path / "cameras.bin"))

#     logger.info(f"  3D Points: {len(points3D)}")
#     logger.info(f"  Images: {len(images)}")
#     logger.info(f"  Cameras: {len(cameras)}")

#     # Build image name to ID mapping
#     image_name_to_id = {img.name: img_id for img_id, img in images.items()}

#     # Load all masks
#     logger.info(f"Loading YOLO masks from: {masks_dir}")
#     masks = {}
#     for mask_file in masks_path.glob("*.json"):
#         try:
#             mask_data = load_yolo_mask(mask_file)
#             # Match image name (without extension)
#             img_stem = mask_file.stem
#             masks[img_stem] = mask_data
#         except Exception as e:
#             logger.warning(f"Failed to load mask {mask_file}: {e}")

#     logger.info(f"  Loaded {len(masks)} masks")

#     # Process each 3D point with voting mechanism
#     logger.info("Processing 3D points with voting...")
#     logger.info(f"  Vote threshold: {vote_threshold} (need {vote_threshold*100:.0f}% agreement)")
#     logger.info(f"  Min confidence: {min_confidence}")

#     xyz_list = []
#     rgb_list = []
#     crack_count = 0
#     skipped_count = 0

#     # Statistics for quality metrics
#     vote_counts = []
#     confidence_scores = []
#     view_counts = []

#     for point_id, point in points3D.items():
#         xyz = point.xyz
#         original_rgb = point.rgb

#         # Skip points with short tracks (likely noise)
#         if len(point.image_ids) < min_track_length:
#             skipped_count += 1
#             continue

#         # Voting mechanism: count votes across all views
#         crack_votes = 0
#         total_votes = 0
#         confidence_sum = 0.0

#         for img_id, point2D_idx in zip(point.image_ids, point.point2D_idxs):
#             if img_id not in images:
#                 continue

#             image = images[img_id]
#             image_name = image.name
#             image_stem = Path(image_name).stem

#             # Get pixel coordinates
#             if point2D_idx >= len(image.xys):
#                 continue

#             pixel_xy = image.xys[point2D_idx]

#             # Skip if no mask for this image
#             if image_stem not in masks:
#                 continue

#             total_votes += 1

#             # Check mask with confidence filtering
#             is_in_mask, mask_confidence = is_pixel_in_crack_mask(
#                 masks[image_stem],
#                 pixel_xy,
#                 image_stem,
#                 min_confidence
#             )

#             if is_in_mask:
#                 crack_votes += 1
#                 confidence_sum += mask_confidence

#         # Decision based on voting
#         if total_votes > 0:
#             vote_ratio = crack_votes / total_votes
#             avg_confidence = confidence_sum / crack_votes if crack_votes > 0 else 0.0

#             is_crack = (vote_ratio >= vote_threshold and avg_confidence >= min_confidence)

#             # Collect statistics
#             view_counts.append(total_votes)
#             if is_crack:
#                 vote_counts.append(crack_votes)
#                 confidence_scores.append(avg_confidence)
#         else:
#             is_crack = False

#         # Assign color
#         if is_crack:
#             rgb = crack_color
#             crack_count += 1
#         else:
#             rgb = original_rgb

#         xyz_list.append(xyz)
#         rgb_list.append(rgb)

#     logger.info(f"  Total points: {len(xyz_list)}")
#     logger.info(f"  Crack points: {crack_count} ({crack_count/len(xyz_list)*100:.1f}%)")
#     logger.info(f"  Skipped (short track): {skipped_count}")

#     # Compute quality metrics
#     metrics = {
#         'total_points': len(xyz_list),
#         'crack_points': crack_count,
#         'skipped_points': skipped_count,
#         'crack_ratio': crack_count / len(xyz_list) if len(xyz_list) > 0 else 0,
#     }

#     # Voting statistics
#     if vote_counts:
#         metrics['avg_votes_per_crack'] = float(np.mean(vote_counts))
#         metrics['median_votes_per_crack'] = float(np.median(vote_counts))
#         metrics['min_votes'] = int(np.min(vote_counts))
#         metrics['max_votes'] = int(np.max(vote_counts))

#         logger.info(f"\n  Voting Statistics:")
#         logger.info(f"    Avg votes per crack: {metrics['avg_votes_per_crack']:.1f}")
#         logger.info(f"    Vote range: {metrics['min_votes']}-{metrics['max_votes']}")

#     # Confidence statistics
#     if confidence_scores:
#         metrics['avg_confidence'] = float(np.mean(confidence_scores))
#         metrics['min_confidence'] = float(np.min(confidence_scores))
#         metrics['max_confidence'] = float(np.max(confidence_scores))

#         logger.info(f"    Avg confidence: {metrics['avg_confidence']:.3f}")
#         logger.info(f"    Confidence range: {metrics['min_confidence']:.3f}-{metrics['max_confidence']:.3f}")

#     # View coverage
#     if view_counts:
#         metrics['avg_views_per_point'] = float(np.mean(view_counts))
#         metrics['points_single_view'] = int(sum(1 for v in view_counts if v == 1))
#         metrics['points_multi_view'] = int(sum(1 for v in view_counts if v >= 2))

#         logger.info(f"    Avg views per point: {metrics['avg_views_per_point']:.1f}")

#     # Quality warnings
#     warnings = []

#     if metrics.get('avg_confidence', 1.0) < 0.5:
#         warnings.append(f"Low average confidence: {metrics['avg_confidence']:.3f}")

#     if view_counts and metrics.get('points_single_view', 0) / len(view_counts) > 0.5:
#         single_view_ratio = metrics['points_single_view'] / len(view_counts)
#         warnings.append(f"High single-view ratio: {single_view_ratio*100:.1f}%")

#     if metrics['crack_ratio'] > 0.3:
#         warnings.append(f"Unusually high crack ratio: {metrics['crack_ratio']*100:.1f}%")

#     if warnings:
#         logger.warning("\n  Quality Warnings:")
#         for warning in warnings:
#             logger.warning(f"    ⚠️  {warning}")
#         metrics['quality_warnings'] = warnings
#     else:
#         metrics['quality_warnings'] = []
#         logger.info(f"\n  ✅ No quality warnings")

#     # Save PLY
#     logger.info(f"\nSaving point cloud to: {output_ply}")
#     save_ply(output_ply, xyz_list, rgb_list)

#     logger.info("=" * 80)
#     logger.info("Point cloud overlay complete!")
#     logger.info("=" * 80)

#     return metrics


# def save_ply(filename: str, xyz: List[np.ndarray], rgb: List[np.ndarray]):
#     """
#     Save point cloud as binary PLY.

#     Args:
#         filename: Output PLY path
#         xyz: List of 3D coordinates
#         rgb: List of RGB colors (uint8)
#     """
#     xyz = np.array(xyz)
#     rgb = np.array(rgb, dtype=np.uint8)

#     assert xyz.shape[0] == rgb.shape[0]
#     assert xyz.shape[1] == 3
#     assert rgb.shape[1] == 3

#     # Create output directory
#     output_path = Path(filename)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     # Write binary PLY
#     with open(filename, 'wb') as f:
#         # Header
#         header = f"""ply
# format binary_little_endian 1.0
# element vertex {len(xyz)}
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# end_header
# """
#         f.write(header.encode('ascii'))

#         # Vertices
#         for i in range(len(xyz)):
#             # xyz as float32
#             f.write(xyz[i].astype(np.float32).tobytes())
#             # rgb as uint8
#             f.write(rgb[i].tobytes())

#     logger.info(f"Saved {len(xyz)} points to {filename}")


# if __name__ == '__main__':
#     import argparse
#     from .utils import setup_logging

#     parser = argparse.ArgumentParser(
#         description='Overlay YOLO masks on SFM point cloud with voting mechanism'
#     )
#     parser.add_argument('--sparse-dir', required=True,
#                        help='COLMAP sparse directory (e.g., data/sfm/sparse/0)')
#     parser.add_argument('--masks-dir', required=True,
#                        help='YOLO masks directory')
#     parser.add_argument('--output', required=True,
#                        help='Output PLY path')
#     parser.add_argument('--crack-color', type=int, nargs=3, default=[255, 0, 0],
#                        help='RGB color for crack points (default: 255 0 0)')
#     parser.add_argument('--min-track-length', type=int, default=2,
#                        help='Minimum track length (default: 2)')
#     parser.add_argument('--vote-threshold', type=float, default=0.5,
#                        help='Minimum vote ratio to mark as crack (default: 0.5 = majority)')
#     parser.add_argument('--min-confidence', type=float, default=0.25,
#                        help='Minimum YOLO confidence to consider (default: 0.25)')
#     parser.add_argument('--log-level', default='INFO',
#                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

#     args = parser.parse_args()

#     setup_logging(args.log_level)

#     try:
#         result = overlay_masks_on_pointcloud(
#             args.sparse_dir,
#             args.masks_dir,
#             args.output,
#             tuple(args.crack_color),
#             args.min_track_length,
#             args.vote_threshold,
#             args.min_confidence
#         )

#         print(f"\n✅ Point cloud overlay complete!")
#         print(f"   Total points: {result['total_points']}")
#         print(f"   Crack points: {result['crack_points']} ({result['crack_ratio']*100:.1f}%)")
#         if 'avg_confidence' in result:
#             print(f"   Avg confidence: {result['avg_confidence']:.3f}")
#         if 'avg_votes_per_crack' in result:
#             print(f"   Avg votes per crack: {result['avg_votes_per_crack']:.1f}")
#         print(f"   Output: {args.output}")

#     except Exception as e:
#         logger.error(f"Overlay failed: {e}", exc_info=True)
#         import sys
#         sys.exit(1)
"""
Point Cloud Mask Overlay

Overlays YOLO crack masks onto SFM sparse point cloud.
Uses COLMAP track information to map 2D pixels to 3D points.

Usage:
    python -m src.point_cloud_overlay \
        --sparse-dir data/sfm/sparse/0 \
        --masks-dir data/yolo_masks \
        --output outputs/sfm_masked_cloud.ply
"""
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from .colmap_io import read_points3D_binary, read_images_binary, read_cameras_binary

logger = logging.getLogger(__name__)


def load_yolo_mask(mask_path: Path) -> Dict:
    """Load YOLO mask JSON"""
    with open(mask_path, 'r') as f:
        return json.load(f)


def is_pixel_in_crack_mask(
    mask_json: Dict,
    pixel_xy: Tuple[float, float],
    image_stem: str = None,
    min_confidence: float = 0.0,
    filter_class: str = None
) -> Tuple[bool, float, int, str]:
    """
    Check if pixel is inside any polygon with confidence validation.

    Args:
        mask_json: YOLO mask JSON data
        pixel_xy: Pixel coordinates (u, v)
        image_stem: Image identifier for logging (optional)
        min_confidence: Minimum YOLO confidence to consider (default: 0.0)
        filter_class: If specified, only check this class (e.g., 'crack'). If None, check all classes.

    Returns:
        (is_inside, confidence, mask_id, class_name): True if pixel is in valid mask,
                                          confidence score, mask index (-1 if not found), and class name
    """
    if 'masks' not in mask_json:
        return False, 0.0, -1, ''

    point = ShapelyPoint(pixel_xy)

    for mask_idx, mask in enumerate(mask_json['masks']):
        # Filter by class if specified
        if filter_class and mask['class'] != filter_class:
            continue

        confidence = mask.get('score', 0.0)

        # Filter by minimum confidence
        if confidence < min_confidence:
            continue

        polygon_coords = mask['polygon']

        # Validate polygon has enough points
        if len(polygon_coords) < 3:
            if image_stem:
                logger.debug(f"[{image_stem}] Mask {mask_idx}: < 3 points, skipping")
            continue

        try:
            poly = Polygon(polygon_coords)

            # Auto-fix invalid polygons with buffer(0) trick
            if not poly.is_valid:
                try:
                    fixed_poly = poly.buffer(0)

                    # 🆕 Verify the fix was successful
                    if not fixed_poly.is_valid or fixed_poly.is_empty:
                        if image_stem:
                            logger.debug(
                                f"[{image_stem}] Mask {mask_idx}: Cannot fix invalid polygon, skipping"
                            )
                        continue

                    # Use fixed polygon
                    poly = fixed_poly

                    if image_stem:
                        logger.debug(
                            f"[{image_stem}] Mask {mask_idx}: Auto-fixed invalid polygon "
                            f"(area={poly.area:.2f}, confidence={confidence:.3f})"
                        )

                except Exception as e:
                    # If buffer(0) fails, skip this polygon
                    if image_stem:
                        logger.debug(
                            f"[{image_stem}] Mask {mask_idx}: Buffer fix failed ({e}), skipping"
                        )
                    continue

            # Check if polygon is too small (degenerate)
            if poly.area < 1.0:
                if image_stem:
                    logger.debug(
                        f"[{image_stem}] Mask {mask_idx}: Area too small ({poly.area:.2f}px²)"
                    )
                continue

            # 🆕 Final safety check before contains()
            if not poly.is_valid:
                if image_stem:
                    logger.warning(
                        f"[{image_stem}] Mask {mask_idx}: Still invalid after fix, skipping"
                    )
                continue

            if poly.contains(point):
                class_name = mask.get('class', 'unknown')
                return True, confidence, mask_idx, class_name

        except Exception as e:
            if image_stem:
                logger.error(
                    f"[{image_stem}] Mask {mask_idx}: Unexpected error - {e}"
                )
            continue

    return False, 0.0, -1, ''


def overlay_masks_on_pointcloud(
    sparse_dir: str,
    masks_dir: str,
    output_ply: str,
    output_json: str = None,
    crack_color: Tuple[int, int, int] = (255, 0, 0),
    class_colors: Dict[str, Tuple[int, int, int]] = None,
    min_track_length: int = 2,
    vote_threshold: float = 0.5,
    class_vote_threshold: float = 0.5,
    min_confidence: float = 0.25,
    filter_class: str = None
):
    """
    Overlay YOLO masks on SFM point cloud with 2-stage voting mechanism.

    Args:
        sparse_dir: COLMAP sparse/0 directory
        masks_dir: YOLO masks directory
        output_ply: Output PLY path
        output_json: Output JSON path for defect points with mapping info (optional)
        crack_color: RGB color for crack points (default: red, kept for backward compatibility)
        class_colors: Dict of class_name -> RGB color. If None, uses crack_color for all.
        min_track_length: Minimum track length to include point
        vote_threshold: Stage 1 voting - minimum ratio of views detecting ANY defect (default: 0.5)
        class_vote_threshold: Stage 2 voting - minimum ratio for class assignment (default: 0.5)
        min_confidence: Minimum YOLO confidence to consider (default: 0.25)
        filter_class: If specified, only process this class (e.g., 'crack'). If None, process all classes.

    Voting Logic:
        Stage 1: Check if (detected_views / total_views) >= vote_threshold
                 If fails, point is classified as non-defect.
        Stage 2: Among detected views, select class with most votes.
                 Check if (class_votes / total_detected) >= class_vote_threshold
                 If passes, assign that class. Otherwise, non-defect.
    """
    logger.info("=" * 80)
    logger.info("Point Cloud Mask Overlay")
    logger.info("=" * 80)

    # Setup class colors
    if class_colors is None:
        # Default: use crack_color for all classes
        class_colors = {}

    # Ensure crack color is set (backward compatibility)
    if 'crack' not in class_colors:
        class_colors['crack'] = crack_color

    sparse_path = Path(sparse_dir)
    masks_path = Path(masks_dir)

    # Read COLMAP data
    logger.info(f"Reading COLMAP data from: {sparse_dir}")
    points3D = read_points3D_binary(str(sparse_path / "points3D.bin"))
    images = read_images_binary(str(sparse_path / "images.bin"))
    cameras = read_cameras_binary(str(sparse_path / "cameras.bin"))

    logger.info(f"  3D Points: {len(points3D)}")
    logger.info(f"  Images: {len(images)}")
    logger.info(f"  Cameras: {len(cameras)}")

    # Build image name to ID mapping
    image_name_to_id = {img.name: img_id for img_id, img in images.items()}

    # Load all masks
    logger.info(f"Loading YOLO masks from: {masks_dir}")
    masks = {}
    for mask_file in masks_path.glob("*.json"):
        try:
            mask_data = load_yolo_mask(mask_file)
            # Match image name (without extension)
            img_stem = mask_file.stem
            masks[img_stem] = mask_data
        except Exception as e:
            logger.warning(f"Failed to load mask {mask_file}: {e}")

    logger.info(f"  Loaded {len(masks)} masks")

    # Process each 3D point with 2-stage voting mechanism
    logger.info("Processing 3D points with 2-stage voting...")
    logger.info(f"  Stage 1 - Detection threshold: {vote_threshold} (need {vote_threshold*100:.0f}% views detecting defect)")
    logger.info(f"  Stage 2 - Class threshold: {class_vote_threshold} (need {class_vote_threshold*100:.0f}% agreement on class)")
    logger.info(f"  Min confidence: {min_confidence}")

    xyz_list = []
    rgb_list = []
    crack_count = 0
    skipped_count = 0

    # Statistics for quality metrics
    vote_counts = []
    confidence_scores = []
    view_counts = []

    # Crack points with mapping info
    crack_points_data = []

    for point_id, point in points3D.items():
        xyz = point.xyz
        original_rgb = point.rgb

        # Skip points with short tracks (likely noise)
        if len(point.image_ids) < min_track_length:
            skipped_count += 1
            continue

        # Voting mechanism: count votes across all views
        class_votes = {}  # class_name -> vote_count
        class_confidences = {}  # class_name -> list of confidences
        total_votes = 0
        source_masks = []  # Mapping info

        for img_id, point2D_idx in zip(point.image_ids, point.point2D_idxs):
            if img_id not in images:
                continue

            image = images[img_id]
            image_name = image.name
            image_stem = Path(image_name).stem

            # Get pixel coordinates
            if point2D_idx >= len(image.xys):
                continue

            pixel_xy = image.xys[point2D_idx]

            # Skip if no mask for this image
            if image_stem not in masks:
                continue

            total_votes += 1

            # Check mask with confidence filtering
            is_in_mask, mask_confidence, mask_id, detected_class = is_pixel_in_crack_mask(
                masks[image_stem],
                pixel_xy,
                image_stem,
                min_confidence,
                filter_class
            )

            if is_in_mask and detected_class:
                # Track votes per class
                if detected_class not in class_votes:
                    class_votes[detected_class] = 0
                    class_confidences[detected_class] = []

                class_votes[detected_class] += 1
                class_confidences[detected_class].append(mask_confidence)

                source_masks.append({
                    'image_id': image_stem,
                    'mask_id': mask_id,
                    'class': detected_class,
                    'confidence': mask_confidence,
                    'uv': [float(pixel_xy[0]), float(pixel_xy[1])]  # 2D pixel coordinates
                })

        # 2-Stage Voting Logic
        if total_votes > 0 and class_votes:
            # Stage 1: Check if enough views detect ANY defect (any class)
            total_detected_votes = sum(class_votes.values())
            detection_ratio = total_detected_votes / total_votes

            # Collect statistics for all points with votes
            view_counts.append(total_votes)

            if detection_ratio >= vote_threshold:
                # Stage 1 passed: This point is detected as defect by enough views

                # Stage 2: Among detected views, select class with most votes
                best_class = max(class_votes.items(), key=lambda x: x[1])[0]
                best_class_votes = class_votes[best_class]
                best_class_confidences = class_confidences[best_class]

                class_ratio = best_class_votes / total_detected_votes
                avg_confidence = sum(best_class_confidences) / len(best_class_confidences)

                # Stage 2: Check if the dominant class has enough agreement
                is_defect = (class_ratio >= class_vote_threshold and avg_confidence >= min_confidence)

                if is_defect:
                    vote_counts.append(best_class_votes)
                    confidence_scores.append(avg_confidence)

                    # Store defect point with mapping info
                    crack_points_data.append({
                        'point_id': int(point_id),
                        'xyz': xyz.tolist(),
                        'class': best_class,
                        'source_masks': source_masks,
                        'detection_ratio': detection_ratio,  # Stage 1 ratio
                        'class_ratio': class_ratio,          # Stage 2 ratio
                        'avg_confidence': avg_confidence,
                        'n_views': len(source_masks),
                        'total_votes': total_votes,
                        'detected_votes': total_detected_votes
                    })

                    # Assign class-specific color
                    rgb = class_colors.get(best_class, crack_color)
                    crack_count += 1

                    xyz_list.append(xyz)
                    rgb_list.append(rgb)
                else:
                    # Stage 2 failed - not enough class agreement
                    xyz_list.append(xyz)
                    rgb_list.append(original_rgb)
            else:
                # Stage 1 failed - not enough detection
                xyz_list.append(xyz)
                rgb_list.append(original_rgb)
        else:
            # No votes - use original color
            xyz_list.append(xyz)
            rgb_list.append(original_rgb)

    logger.info(f"  Total points: {len(xyz_list)}")
    logger.info(f"  Defect points: {crack_count} ({crack_count/len(xyz_list)*100:.1f}%)")
    logger.info(f"  Skipped (short track): {skipped_count}")

    # Log class distribution
    if crack_points_data:
        class_counts = {}
        for pt in crack_points_data:
            cls = pt.get('class', 'unknown')
            class_counts[cls] = class_counts.get(cls, 0) + 1
        logger.info(f"  Class distribution:")
        for cls, count in sorted(class_counts.items()):
            logger.info(f"    {cls}: {count}")

    # Compute quality metrics
    metrics = {
        'total_points': len(xyz_list),
        'crack_points': crack_count,
        'skipped_points': skipped_count,
        'crack_ratio': crack_count / len(xyz_list) if len(xyz_list) > 0 else 0,
    }

    # Voting statistics
    if vote_counts:
        metrics['avg_votes_per_crack'] = float(np.mean(vote_counts))
        metrics['median_votes_per_crack'] = float(np.median(vote_counts))
        metrics['min_votes'] = int(np.min(vote_counts))
        metrics['max_votes'] = int(np.max(vote_counts))

        logger.info(f"\n  Voting Statistics:")
        logger.info(f"    Avg votes per crack: {metrics['avg_votes_per_crack']:.1f}")
        logger.info(f"    Vote range: {metrics['min_votes']}-{metrics['max_votes']}")

    # Confidence statistics
    if confidence_scores:
        metrics['avg_confidence'] = float(np.mean(confidence_scores))
        metrics['min_confidence'] = float(np.min(confidence_scores))
        metrics['max_confidence'] = float(np.max(confidence_scores))

        logger.info(f"    Avg confidence: {metrics['avg_confidence']:.3f}")
        logger.info(f"    Confidence range: {metrics['min_confidence']:.3f}-{metrics['max_confidence']:.3f}")

    # View coverage
    if view_counts:
        metrics['avg_views_per_point'] = float(np.mean(view_counts))
        metrics['points_single_view'] = int(sum(1 for v in view_counts if v == 1))
        metrics['points_multi_view'] = int(sum(1 for v in view_counts if v >= 2))

        logger.info(f"    Avg views per point: {metrics['avg_views_per_point']:.1f}")

    # Quality warnings
    warnings = []

    if metrics.get('avg_confidence', 1.0) < 0.5:
        warnings.append(f"Low average confidence: {metrics['avg_confidence']:.3f}")

    if view_counts and metrics.get('points_single_view', 0) / len(view_counts) > 0.5:
        single_view_ratio = metrics['points_single_view'] / len(view_counts)
        warnings.append(f"High single-view ratio: {single_view_ratio*100:.1f}%")

    if metrics['crack_ratio'] > 0.3:
        warnings.append(f"Unusually high crack ratio: {metrics['crack_ratio']*100:.1f}%")

    if warnings:
        logger.warning("\n  Quality Warnings:")
        for warning in warnings:
            logger.warning(f"    ⚠️  {warning}")
        metrics['quality_warnings'] = warnings
    else:
        metrics['quality_warnings'] = []
        logger.info(f"\n  ✅ No quality warnings")

    # Save PLY
    logger.info(f"\nSaving point cloud to: {output_ply}")
    save_ply(output_ply, xyz_list, rgb_list)

    # Save JSON with mapping info (optional)
    if output_json:
        logger.info(f"Saving defect points with mapping info to: {output_json}")

        json_output = {
            'metadata': {
                'total_points': len(xyz_list),
                'defect_points': crack_count,  # Renamed from crack_points
                'skipped_points': skipped_count,
                'vote_threshold': vote_threshold,
                'min_confidence': min_confidence,
                'min_track_length': min_track_length,
                'avg_confidence': metrics.get('avg_confidence', 0.0),
                'avg_views_per_defect': metrics.get('avg_votes_per_crack', 0.0)
            },
            'points': crack_points_data
        }

        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_json, 'w') as f:
            json.dump(json_output, f, indent=2)

        logger.info(f"  Saved {len(crack_points_data)} defect points with mapping info")

    logger.info("=" * 80)
    logger.info("Point cloud overlay complete!")
    logger.info("=" * 80)

    return metrics


def save_ply(filename: str, xyz: List[np.ndarray], rgb: List[np.ndarray]):
    """
    Save point cloud as binary PLY.

    Args:
        filename: Output PLY path
        xyz: List of 3D coordinates
        rgb: List of RGB colors (uint8)
    """
    xyz = np.array(xyz)
    rgb = np.array(rgb, dtype=np.uint8)

    assert xyz.shape[0] == rgb.shape[0]
    assert xyz.shape[1] == 3
    assert rgb.shape[1] == 3

    # Create output directory
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write binary PLY
    with open(filename, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(xyz)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))

        # Vertices
        for i in range(len(xyz)):
            # xyz as float32
            f.write(xyz[i].astype(np.float32).tobytes())
            # rgb as uint8
            f.write(rgb[i].tobytes())

    logger.info(f"Saved {len(xyz)} points to {filename}")


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging, load_config

    parser = argparse.ArgumentParser(
        description='Overlay YOLO masks on SFM point cloud with voting mechanism'
    )
    parser.add_argument('--config', default=None,
                       help='Configuration YAML file (optional, for multi-class colors)')
    parser.add_argument('--sparse-dir', required=True,
                       help='COLMAP sparse directory (e.g., data/sfm/sparse/0)')
    parser.add_argument('--masks-dir', required=True,
                       help='YOLO masks directory')
    parser.add_argument('--output', required=True,
                       help='Output PLY path')
    parser.add_argument('--output-json', default=None,
                       help='Output JSON path for crack points with mapping info (optional)')
    parser.add_argument('--crack-color', type=int, nargs=3, default=[255, 0, 0],
                       help='RGB color for crack points (default: 255 0 0)')
    parser.add_argument('--min-track-length', type=int, default=2,
                       help='Minimum track length (default: 2)')
    parser.add_argument('--vote-threshold', type=float, default=0.5,
                       help='Stage 1: Minimum ratio of views detecting ANY defect (default: 0.5)')
    parser.add_argument('--class-vote-threshold', type=float, default=0.5,
                       help='Stage 2: Minimum ratio for class assignment among detected views (default: 0.5)')
    parser.add_argument('--min-confidence', type=float, default=0.25,
                       help='Minimum YOLO confidence to consider (default: 0.25)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    # Load class colors from config if provided
    class_colors = None
    if args.config:
        config = load_config(args.config)
        configured_colors = config.get('colors', {})
        if configured_colors:
            # Convert list colors to tuples
            class_colors = {
                name: tuple(color) for name, color in configured_colors.items()
            }
            logger.info(f"Loaded {len(class_colors)} class colors from config")

    try:
        result = overlay_masks_on_pointcloud(
            args.sparse_dir,
            args.masks_dir,
            args.output,
            args.output_json,
            tuple(args.crack_color),
            class_colors,
            args.min_track_length,
            args.vote_threshold,
            args.class_vote_threshold,
            args.min_confidence
        )

        print(f"\n✅ Point cloud overlay complete!")
        print(f"   Total points: {result['total_points']}")
        print(f"   Crack points: {result['crack_points']} ({result['crack_ratio']*100:.1f}%)")
        if 'avg_confidence' in result:
            print(f"   Avg confidence: {result['avg_confidence']:.3f}")
        if 'avg_votes_per_crack' in result:
            print(f"   Avg votes per crack: {result['avg_votes_per_crack']:.1f}")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Overlay failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
