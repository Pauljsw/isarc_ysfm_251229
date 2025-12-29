#!/usr/bin/env python3
"""
Mask merging and deduplication for crop-based inference.

Handles overlapping detections from adjacent crops using IoU-based
Non-Maximum Suppression (NMS) to eliminate duplicates while preserving
detection integrity.
"""

import logging
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def polygon_to_mask(
    polygon: List[List[float]],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert polygon to binary mask for IoU calculation.

    Args:
        polygon: List of [x, y] points
        image_shape: (height, width) of target mask

    Returns:
        Binary mask (H, W) with dtype=uint8
    """
    import cv2

    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    if len(polygon) < 3:
        return mask

    # Convert to integer coordinates
    polygon_np = np.array(polygon, dtype=np.int32)

    # Clip to image bounds
    polygon_np[:, 0] = np.clip(polygon_np[:, 0], 0, W - 1)
    polygon_np[:, 1] = np.clip(polygon_np[:, 1], 0, H - 1)

    # Fill polygon
    cv2.fillPoly(mask, [polygon_np], 1)

    return mask


def compute_polygon_iou(
    poly1: List[List[float]],
    poly2: List[List[float]],
    image_shape: Tuple[int, int],
    use_shapely: bool = False
) -> float:
    """
    Compute IoU between two polygons.

    Args:
        poly1, poly2: Polygons as list of [x, y] points
        image_shape: (height, width) for mask rendering
        use_shapely: Use Shapely library if available (more accurate)

    Returns:
        IoU value [0, 1]
    """
    if use_shapely:
        try:
            from shapely.geometry import Polygon
            from shapely.validation import make_valid

            p1 = Polygon(poly1)
            p2 = Polygon(poly2)

            # Handle invalid geometries
            if not p1.is_valid:
                p1 = make_valid(p1)
            if not p2.is_valid:
                p2 = make_valid(p2)

            if p1.area == 0 or p2.area == 0:
                return 0.0

            intersection = p1.intersection(p2).area
            union = p1.union(p2).area

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.debug(f"Shapely IoU failed: {e}, falling back to rasterization")

    # Fallback: Rasterization-based IoU
    mask1 = polygon_to_mask(poly1, image_shape)
    mask2 = polygon_to_mask(poly2, image_shape)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    return float(intersection / union) if union > 0 else 0.0


def merge_detections_nms(
    detections: List[Dict],
    iou_threshold: float = 0.5,
    use_shapely: bool = True,
    image_shape: Optional[Tuple[int, int]] = None
) -> List[Dict]:
    """
    Merge overlapping detections using Non-Maximum Suppression.

    Critical for crop-based inference where the same defect may be
    detected in multiple adjacent crops.

    Algorithm:
    1. Sort detections by confidence score (descending)
    2. For each detection (highest confidence first):
       - Keep if no overlap with higher-confidence detection
       - Suppress if IoU > threshold with same-class higher-confidence detection

    Args:
        detections: List of detection dicts with:
            - 'polygon': List of [x, y] points
            - 'class_id': Integer class ID
            - 'score': Confidence score
            - (optional) 'crop_id': Source crop identifier
        iou_threshold: IoU threshold for suppression
        use_shapely: Use Shapely for geometry (recommended)
        image_shape: (height, width) for fallback rasterization

    Returns:
        Deduplicated list of detections
    """
    if len(detections) == 0:
        return []

    # Infer image shape if not provided
    if image_shape is None and len(detections) > 0:
        # Get max coordinates from all polygons
        max_x = max_y = 0
        for det in detections:
            poly = det['polygon']
            for x, y in poly:
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        image_shape = (int(max_y) + 100, int(max_x) + 100)
        logger.debug(f"Inferred image shape: {image_shape}")

    # Sort by confidence (descending)
    sorted_indices = sorted(
        range(len(detections)),
        key=lambda i: detections[i]['score'],
        reverse=True
    )

    keep_indices = []
    suppressed = set()

    for i in sorted_indices:
        if i in suppressed:
            continue

        det_i = detections[i]
        poly_i = det_i['polygon']
        class_i = det_i['class_id']

        # Keep this detection
        keep_indices.append(i)

        # Check for suppressions
        for j in sorted_indices:
            if j <= i or j in suppressed:
                continue

            det_j = detections[j]
            poly_j = det_j['polygon']
            class_j = det_j['class_id']

            # Only suppress same-class detections
            if class_i != class_j:
                continue

            # Compute IoU
            try:
                iou = compute_polygon_iou(
                    poly_i, poly_j,
                    image_shape,
                    use_shapely=use_shapely
                )

                if iou > iou_threshold:
                    suppressed.add(j)
                    logger.debug(
                        f"Suppressed detection {j} (IoU={iou:.3f} with {i})"
                    )

            except Exception as e:
                logger.warning(f"IoU computation failed for {i},{j}: {e}")
                continue

    # Return kept detections
    merged = [detections[i] for i in keep_indices]

    logger.info(
        f"NMS: {len(detections)} detections → {len(merged)} "
        f"({len(detections) - len(merged)} suppressed)"
    )

    return merged


def analyze_crop_overlap(
    detections: List[Dict],
    n_crops: int
) -> Dict:
    """
    Analyze detection distribution across crops.

    Useful for debugging and validation.

    Args:
        detections: List of detections with 'crop_id' field
        n_crops: Total number of crops

    Returns:
        Statistics dict
    """
    crop_counts = {}
    for det in detections:
        crop_id = det.get('crop_id', None)
        if crop_id is not None:
            crop_counts[crop_id] = crop_counts.get(crop_id, 0) + 1

    stats = {
        'total_detections': len(detections),
        'n_crops': n_crops,
        'crops_with_detections': len(crop_counts),
        'detections_per_crop': crop_counts,
        'avg_detections_per_crop': len(detections) / n_crops if n_crops > 0 else 0
    }

    logger.debug(f"Crop overlap analysis: {stats}")
    return stats


def merge_class_aware(
    detections: List[Dict],
    class_names: List[str],
    iou_threshold: float = 0.5,
    per_class_threshold: Optional[Dict[str, float]] = None,
    image_shape: Optional[Tuple[int, int]] = None
) -> Tuple[List[Dict], Dict]:
    """
    Class-aware NMS with optional per-class IoU thresholds.

    Args:
        detections: List of detections
        class_names: List of class names (for lookup)
        iou_threshold: Default IoU threshold
        per_class_threshold: Optional dict of class_name -> threshold
        image_shape: Image dimensions

    Returns:
        (merged_detections, statistics)
    """
    if per_class_threshold is None:
        per_class_threshold = {}

    # Group by class
    by_class = {}
    for det in detections:
        class_id = det['class_id']
        if class_id not in by_class:
            by_class[class_id] = []
        by_class[class_id].append(det)

    # Merge each class independently
    merged_all = []
    class_stats = {}

    for class_id, class_dets in by_class.items():
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class_{class_id}"

        # Get threshold for this class
        threshold = per_class_threshold.get(class_name, iou_threshold)

        # Merge
        merged_class = merge_detections_nms(
            class_dets,
            iou_threshold=threshold,
            image_shape=image_shape
        )

        merged_all.extend(merged_class)

        class_stats[class_name] = {
            'before': len(class_dets),
            'after': len(merged_class),
            'suppressed': len(class_dets) - len(merged_class),
            'threshold': threshold
        }

    stats = {
        'total_before': len(detections),
        'total_after': len(merged_all),
        'total_suppressed': len(detections) - len(merged_all),
        'by_class': class_stats
    }

    logger.info(
        f"Class-aware NMS: {len(detections)} → {len(merged_all)} "
        f"({stats['total_suppressed']} suppressed)"
    )

    return merged_all, stats


if __name__ == '__main__':
    # Unit tests
    print("Testing mask merging...")

    # Test 1: IoU computation
    poly1 = [[0, 0], [100, 0], [100, 100], [0, 100]]
    poly2 = [[50, 50], [150, 50], [150, 150], [50, 150]]

    iou = compute_polygon_iou(poly1, poly2, (200, 200), use_shapely=False)
    print(f"IoU (rasterization): {iou:.3f}")

    # Test 2: NMS
    detections = [
        {'polygon': poly1, 'class_id': 0, 'score': 0.9},
        {'polygon': poly2, 'class_id': 0, 'score': 0.8},  # Should be kept (different enough)
        {'polygon': [[0, 0], [105, 0], [105, 105], [0, 105]], 'class_id': 0, 'score': 0.7},  # Should be suppressed
    ]

    merged = merge_detections_nms(detections, iou_threshold=0.5, use_shapely=False, image_shape=(200, 200))
    assert len(merged) == 2, f"Expected 2 detections after NMS, got {len(merged)}"

    # Test 3: Class-aware merging
    detections_multi = [
        {'polygon': poly1, 'class_id': 0, 'score': 0.9},
        {'polygon': poly2, 'class_id': 1, 'score': 0.8},  # Different class, should be kept
    ]

    merged_multi, stats = merge_class_aware(
        detections_multi,
        class_names=['crack', 'spalling'],
        image_shape=(200, 200)
    )

    assert len(merged_multi) == 2, "Different classes should not be suppressed"

    print("✅ All tests passed!")
