#!/usr/bin/env python3
"""
Measure crack clusters using 3D segmentation + 2D pixel measurement.

Pipeline:
1. 3D: Segment cluster along principal axis
2. 3D: For each segment, find best covering mask
3. 2D: Measure length/width using skeleton method
4. Aggregate measurements across segments

Inputs:
- crack_clusters.json (from DBSCAN clustering)
- crack_points.json (from point cloud overlay)
- YOLO masks directory
- pixel_mm_ratio.json (from calibration)

Output:
- cluster_measurements.json
"""

import json
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from skimage.morphology import skeletonize
from skimage.draw import polygon as draw_polygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


def generate_cluster_colors(n_clusters: int) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for clusters using HSV color space.
    Must match cluster_crack_points_dbscan.py for consistency.
    Uses golden ratio and varying saturation/value to avoid duplicates.
    """
    if n_clusters == 0:
        return []

    colors = []
    for i in range(n_clusters):
        # Vary hue across the spectrum using golden ratio
        hue = (i * 0.618033988749895) % 1.0

        # Vary saturation and value for more distinction
        sat_idx = (i // 12) % 3
        saturation = [1.0, 0.7, 0.85][sat_idx]
        value = [1.0, 0.9, 0.95][sat_idx]

        # HSV to RGB conversion
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))

    return colors


def rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB to approximate color name for user convenience.
    """
    r, g, b = rgb

    # Simple hue-based naming
    if r >= 200 and g < 100 and b < 100:
        return "Red"
    elif r >= 200 and g >= 200 and b < 100:
        return "Yellow"
    elif r < 100 and g >= 200 and b < 100:
        return "Green"
    elif r < 100 and g >= 200 and b >= 200:
        return "Cyan"
    elif r < 100 and g < 100 and b >= 200:
        return "Blue"
    elif r >= 200 and g < 100 and b >= 200:
        return "Magenta"
    elif r >= 200 and g >= 100 and b < 100:
        return "Orange"
    elif r >= 100 and g < 100 and b >= 100:
        return "Purple"
    elif r < 150 and g >= 150 and b < 150:
        return "Lime"
    elif r < 100 and g >= 100 and b >= 150:
        return "Teal"
    else:
        # Return hex for ambiguous colors
        return f"#{r:02X}{g:02X}{b:02X}"


def setup_logging(level: str = 'INFO'):
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_scale_with_fallback(
    scale_map: np.ndarray,
    r: int,
    c: int,
    search_radius: int = 20
) -> float:
    """
    Get scale value at (r, c) with fallback to nearby valid pixels.

    If scale_map[r, c] is 0 or invalid, search in expanding squares
    for nearby valid scale values and return their mean.
    """
    H, W = scale_map.shape

    # First try direct value
    if 0 <= r < H and 0 <= c < W:
        val = scale_map[r, c]
        if val > 0:
            return float(val)

    # Search in expanding squares for valid neighbors
    for radius in range(1, search_radius + 1):
        valid_values = []

        # Check pixels at this radius (square boundary)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                # Only check boundary pixels of the square
                if abs(dr) != radius and abs(dc) != radius:
                    continue

                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    val = scale_map[nr, nc]
                    if val > 0:
                        valid_values.append(val)

        if valid_values:
            return float(np.mean(valid_values))

    return 0.0


# =============================================================================
# 3D Functions: Cluster Segmentation
# =============================================================================

def compute_principal_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal axis using PCA.

    Returns:
        (centroid, principal_direction)
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    try:
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_idx = np.argmax(eigenvalues)
        principal_dir = eigenvectors[:, principal_idx]
        return centroid, principal_dir
    except:
        return centroid, np.array([1.0, 0.0, 0.0])


def segment_cluster_3d(
    cluster_points: List[Dict],
    n_segments: int = 5,
    adaptive: bool = True
) -> List[Dict]:
    """
    Segment cluster along principal axis.

    Args:
        cluster_points: List of point dicts with 'point_id', 'xyz', 'source_masks'
        n_segments: Number of segments (if not adaptive)
        adaptive: Use adaptive segmentation based on point distribution

    Returns:
        List of segment dicts with 'point_ids' and 'source_masks'
    """
    if len(cluster_points) < 2:
        # Single segment for very small clusters
        return [{
            'segment_id': 0,
            'point_ids': [p['point_id'] for p in cluster_points],
            'points': cluster_points
        }]

    # Get 3D coordinates
    xyz = np.array([p['xyz'] for p in cluster_points])

    # Compute principal axis
    centroid, principal_dir = compute_principal_axis(xyz)

    # Project points onto principal axis
    projections = np.dot(xyz - centroid, principal_dir)

    # Determine segment boundaries
    min_proj, max_proj = projections.min(), projections.max()

    if adaptive and len(cluster_points) > 20:
        # Adaptive: adjust n_segments based on crack extent
        extent = max_proj - min_proj
        # Roughly one segment per 0.05m (5cm)
        n_segments = max(2, min(10, int(extent / 0.05) + 1))

    # Create segment boundaries
    boundaries = np.linspace(min_proj, max_proj, n_segments + 1)

    # Assign points to segments
    segments = []
    for i in range(n_segments):
        low, high = boundaries[i], boundaries[i + 1]

        # Include boundary points in segment
        if i == n_segments - 1:
            mask = (projections >= low) & (projections <= high)
        else:
            mask = (projections >= low) & (projections < high)

        segment_points = [p for p, m in zip(cluster_points, mask) if m]

        if segment_points:
            segments.append({
                'segment_id': i,
                'point_ids': [p['point_id'] for p in segment_points],
                'points': segment_points
            })

    return segments


def find_best_mask_for_segment(segment: Dict) -> Optional[Tuple[str, int]]:
    """
    Find the best (image_id, mask_id) for a segment based on point contribution.

    Note: Synthetic points (from upsampling) are excluded from voting
    since they don't have valid source information.

    Returns:
        (image_id, mask_id) or None if no masks found
    """
    # Count contributions from each (image_id, mask_id)
    mask_contributions = defaultdict(int)

    for point in segment['points']:
        # Skip synthetic points - they don't have valid source information
        if point.get('is_synthetic', False):
            continue

        for source in point.get('source_masks', []):
            key = (source['image_id'], source['mask_id'])
            mask_contributions[key] += 1

    if not mask_contributions:
        return None

    # Find the mask with most contributions
    best_mask = max(mask_contributions.items(), key=lambda x: x[1])
    return best_mask[0]


# =============================================================================
# 2D Functions: Skeleton-based Measurement
# =============================================================================

def load_mask_polygon(
    masks_dir: Path,
    image_id: str,
    mask_id: int
) -> Optional[List[Tuple[float, float]]]:
    """
    Load mask polygon from YOLO masks JSON.

    Returns:
        List of (x, y) polygon points or None
    """
    # Try different filename patterns
    patterns = [
        f"{image_id}.json",
        f"{image_id}.png.json",
    ]

    for pattern in patterns:
        json_path = masks_dir / pattern
        if json_path.exists():
            with open(json_path) as f:
                masks_data = json.load(f)

            # Get masks array from the JSON structure
            masks_list = None
            if isinstance(masks_data, dict) and 'masks' in masks_data:
                # Format: {"masks": [...]}
                masks_list = masks_data['masks']
            elif isinstance(masks_data, list):
                # Format: [...]
                masks_list = masks_data

            if masks_list and mask_id < len(masks_list):
                mask = masks_list[mask_id]
                return mask.get('polygon', [])

    return None


def polygon_to_binary_mask(
    polygon: List[Tuple[float, float]],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert polygon to binary mask.

    Args:
        polygon: List of (x, y) points
        image_shape: (height, width)

    Returns:
        Binary mask array
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    if len(polygon) < 3:
        return mask

    # Convert to row, col format for skimage
    cols = np.array([p[0] for p in polygon])
    rows = np.array([p[1] for p in polygon])

    # Clip to image bounds
    cols = np.clip(cols, 0, width - 1)
    rows = np.clip(rows, 0, height - 1)

    # Draw filled polygon
    rr, cc = draw_polygon(rows, cols, shape=(height, width))
    mask[rr, cc] = 1

    return mask


def calculate_skeleton_length(
    skeleton: np.ndarray,
    scale_map: np.ndarray
) -> float:
    """
    Calculate crack length from skeleton using direction-based method with per-pixel scale.

    Args:
        skeleton: Binary skeleton image (1px thick)
        scale_map: Per-pixel mm/px scale map (same size as skeleton)

    Returns:
        Length in mm
    """
    # Find skeleton pixels
    rows, cols = np.where(skeleton > 0)

    if len(rows) < 2:
        return 0.0

    # Create set for fast lookup
    skeleton_pixels = set(zip(rows, cols))

    total_length = 0.0
    visited_edges = set()

    for r, c in skeleton_pixels:
        # Get scale at this pixel with fallback to nearby valid pixels
        D = get_scale_with_fallback(scale_map, r, c)

        if D == 0:
            continue

        # Check 8-connected neighbors
        neighbors = [
            (r-1, c),   # up (V)
            (r+1, c),   # down (V)
            (r, c-1),   # left (H)
            (r, c+1),   # right (H)
            (r-1, c-1), # top-left (D)
            (r-1, c+1), # top-right (D)
            (r+1, c-1), # bottom-left (D)
            (r+1, c+1), # bottom-right (D)
        ]

        for i, (nr, nc) in enumerate(neighbors):
            if (nr, nc) in skeleton_pixels:
                # Create edge key (sorted to avoid double counting)
                edge = tuple(sorted([(r, c), (nr, nc)]))

                if edge not in visited_edges:
                    visited_edges.add(edge)

                    # Use average scale of the two pixels (with fallback)
                    D_neighbor = get_scale_with_fallback(scale_map, nr, nc)
                    if D_neighbor == 0:
                        D_neighbor = D
                    D_avg = (D + D_neighbor) / 2.0

                    # Determine distance based on direction
                    if i < 4:  # H or V
                        total_length += D_avg
                    else:  # Diagonal
                        total_length += D_avg * np.sqrt(2)

    return total_length


def calculate_skeleton_width(
    skeleton: np.ndarray,
    binary_mask: np.ndarray,
    scale_map: np.ndarray,
    sample_interval: int = 5
) -> Tuple[float, float]:
    """
    Calculate crack width by measuring perpendicular to skeleton with per-pixel scale.

    Args:
        skeleton: Binary skeleton image
        binary_mask: Original binary mask
        scale_map: Per-pixel mm/px scale map
        sample_interval: Sample every N skeleton pixels

    Returns:
        (average_width_mm, max_width_mm)
    """
    # Find skeleton pixels
    rows, cols = np.where(skeleton > 0)

    if len(rows) < 3:
        return 0.0, 0.0

    # Sample skeleton pixels
    n_pixels = len(rows)
    sample_indices = range(0, n_pixels, sample_interval)

    widths = []

    for idx in sample_indices:
        r, c = rows[idx], cols[idx]

        # Get scale at this pixel with fallback
        D = get_scale_with_fallback(scale_map, r, c)
        if D == 0:
            continue

        # Estimate local direction from nearby skeleton pixels
        window = 3
        nearby_rows = rows[max(0, idx-window):min(n_pixels, idx+window+1)]
        nearby_cols = cols[max(0, idx-window):min(n_pixels, idx+window+1)]

        if len(nearby_rows) < 2:
            continue

        # Fit line to get direction
        dr = nearby_rows[-1] - nearby_rows[0]
        dc = nearby_cols[-1] - nearby_cols[0]

        # Perpendicular direction
        length = np.sqrt(dr**2 + dc**2)
        if length < 1e-6:
            continue

        # Normal vector (perpendicular to skeleton direction)
        nr, nc = -dc / length, dr / length

        # Measure width along normal direction
        width_pixels = measure_width_along_normal(
            binary_mask, r, c, nr, nc
        )

        if width_pixels > 0:
            widths.append(width_pixels * D)

    if not widths:
        return 0.0, 0.0

    return np.mean(widths), np.max(widths)


def measure_width_along_normal(
    mask: np.ndarray,
    r: int, c: int,
    nr: float, nc: float,
    max_distance: int = 100
) -> int:
    """
    Measure mask width along normal direction from point (r, c).

    Returns:
        Width in pixels
    """
    height, width = mask.shape

    # Search in both directions along normal
    positive_dist = 0
    negative_dist = 0

    # Positive direction
    for d in range(1, max_distance):
        new_r = int(round(r + d * nr))
        new_c = int(round(c + d * nc))

        if not (0 <= new_r < height and 0 <= new_c < width):
            break

        if mask[new_r, new_c] == 0:
            break

        positive_dist = d

    # Negative direction
    for d in range(1, max_distance):
        new_r = int(round(r - d * nr))
        new_c = int(round(c - d * nc))

        if not (0 <= new_r < height and 0 <= new_c < width):
            break

        if mask[new_r, new_c] == 0:
            break

        negative_dist = d

    # Total width = positive + negative + 1 (center pixel)
    return positive_dist + negative_dist + 1


def calculate_direct_scan_width(
    skeleton: np.ndarray,
    grayscale: np.ndarray,
    mask: np.ndarray,
    scale_map: np.ndarray,
    sample_interval: int = 5,
    intensity_threshold: float = 0.7,
    max_distance: int = 50
) -> Tuple[float, float]:
    """
    Measure crack width by directly scanning grayscale along skeleton perpendicular.

    At each skeleton pixel, scan perpendicular direction and count consecutive
    dark pixels (below threshold relative to local background).

    Args:
        skeleton: Binary skeleton image
        grayscale: Grayscale image
        mask: YOLO mask (to get local background)
        scale_map: Per-pixel mm/px scale map
        sample_interval: Sample every N skeleton pixels
        intensity_threshold: Ratio to background intensity (0.7 = 70% of background)
        max_distance: Maximum scan distance in pixels

    Returns:
        (average_width_mm, max_width_mm)
    """
    # Find skeleton pixels
    rows, cols = np.where(skeleton > 0)

    if len(rows) < 3:
        return 0.0, 0.0

    # Get background intensity from mask edge region
    # Dilate mask and subtract original to get edge pixels
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    edge_region = dilated - mask.astype(np.uint8)
    edge_pixels = grayscale[edge_region > 0]

    if len(edge_pixels) > 0:
        background_intensity = np.median(edge_pixels)
    else:
        background_intensity = np.median(grayscale[mask > 0])

    dark_threshold = background_intensity * intensity_threshold

    # Sample skeleton pixels
    n_pixels = len(rows)
    sample_indices = range(0, n_pixels, sample_interval)

    widths = []

    for idx in sample_indices:
        r, c = rows[idx], cols[idx]

        # Check if center pixel (skeleton point) is dark enough
        # If skeleton is not on a dark crack pixel, skip this sample
        if grayscale[r, c] > dark_threshold:
            continue

        # Get scale at this pixel with fallback
        D = get_scale_with_fallback(scale_map, r, c)
        if D == 0:
            continue

        # Estimate local direction from nearby skeleton pixels
        window = 3
        nearby_rows = rows[max(0, idx-window):min(n_pixels, idx+window+1)]
        nearby_cols = cols[max(0, idx-window):min(n_pixels, idx+window+1)]

        if len(nearby_rows) < 2:
            continue

        # Get direction
        dr = nearby_rows[-1] - nearby_rows[0]
        dc = nearby_cols[-1] - nearby_cols[0]

        # Perpendicular direction (normal)
        length = np.sqrt(dr**2 + dc**2)
        if length < 1e-6:
            continue

        nr, nc = -dc / length, dr / length

        # Scan in both directions to find consecutive dark pixels
        height, width = grayscale.shape
        positive_dist = 0
        negative_dist = 0

        # Positive direction
        for d in range(1, max_distance):
            new_r = int(round(r + d * nr))
            new_c = int(round(c + d * nc))

            if not (0 <= new_r < height and 0 <= new_c < width):
                break

            if grayscale[new_r, new_c] > dark_threshold:
                break

            positive_dist = d

        # Negative direction
        for d in range(1, max_distance):
            new_r = int(round(r - d * nr))
            new_c = int(round(c - d * nc))

            if not (0 <= new_r < height and 0 <= new_c < width):
                break

            if grayscale[new_r, new_c] > dark_threshold:
                break

            negative_dist = d

        # Total width in pixels (including center)
        width_pixels = positive_dist + negative_dist + 1
        width_mm = width_pixels * D
        widths.append(width_mm)

    if widths:
        return float(np.mean(widths)), float(np.max(widths))
    else:
        return 0.0, 0.0


def detect_crack_pixels_in_mask(
    grayscale: np.ndarray,
    mask: np.ndarray,
    method: str = 'adaptive',
    min_component_ratio: float = 0.1
) -> np.ndarray:
    """
    Detect actual crack pixels within YOLO mask region.

    Methods:
    - gradient: High gradient pixels (crack edges) - most robust
    - percentile: Darkest N% pixels
    - adaptive: Local adaptive thresholding
    - otsu: Otsu's automatic thresholding

    Uses connectivity filtering to keep only continuous/linear crack patterns,
    removing scattered isolated pixels.

    Args:
        grayscale: Grayscale image
        mask: YOLO mask (candidate region)
        method: 'gradient', 'adaptive', 'otsu', or 'percentile'
        min_component_ratio: Minimum component size as ratio of largest component

    Returns:
        Binary image where 1 = actual crack pixel, 0 = background
    """
    # Apply mask to get ROI
    masked_gray = grayscale.copy()
    masked_gray[mask == 0] = 255  # Set non-mask areas to white

    if method == 'gradient' or method.startswith('gradient'):
        # Gradient-based detection - finds crack edges where intensity changes rapidly
        # This naturally forms connected lines along crack boundaries

        # Compute gradient magnitude using Sobel
        grad_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize to 0-255
        gradient_mag = (gradient_mag / gradient_mag.max() * 255).astype(np.uint8)

        # Get gradient values within mask
        mask_gradients = gradient_mag[mask > 0]
        if len(mask_gradients) == 0:
            return np.zeros_like(mask)

        # Extract percentile from method string (e.g., 'gradient_70' means top 30%)
        percentile_val = 70  # default: top 30% of gradients
        if '_' in method:
            try:
                percentile_val = float(method.split('_')[1])
            except:
                pass

        threshold = np.percentile(mask_gradients, percentile_val)
        crack_binary = (gradient_mag > threshold).astype(np.uint8) * 255

    elif method == 'otsu':
        # Otsu's method - good for bimodal distribution
        mask_pixels = grayscale[mask > 0]
        if len(mask_pixels) == 0:
            return np.zeros_like(mask)

        threshold, _ = cv2.threshold(
            mask_pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        crack_binary = (grayscale < threshold).astype(np.uint8) * 255

    elif method == 'adaptive':
        # Adaptive thresholding - handles varying illumination
        crack_binary = cv2.adaptiveThreshold(
            masked_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )

    elif method.startswith('percentile'):
        # Use darkest percentile within mask as crack
        mask_pixels = grayscale[mask > 0]
        if len(mask_pixels) == 0:
            return np.zeros_like(mask)

        # Extract percentile value from method string (e.g., 'percentile_25')
        percentile_val = 30  # default
        if '_' in method:
            try:
                percentile_val = float(method.split('_')[1])
            except:
                pass
        threshold = np.percentile(mask_pixels, percentile_val)
        crack_binary = (grayscale < threshold).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply original mask to limit to ROI
    crack_binary = cv2.bitwise_and(crack_binary, crack_binary, mask=mask.astype(np.uint8))

    # === Connectivity filtering to keep only continuous crack patterns ===

    # Step 1: Morphological opening with larger kernel to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    crack_binary = cv2.morphologyEx(crack_binary, cv2.MORPH_OPEN, kernel)

    # Step 2: Connected component analysis - keep only significant components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        crack_binary, connectivity=8
    )

    if num_labels <= 1:  # Only background
        return crack_binary

    # Find the largest component (excluding background at label 0)
    component_sizes = stats[1:, cv2.CC_STAT_AREA]
    if len(component_sizes) == 0:
        return crack_binary

    max_size = np.max(component_sizes)
    min_size_threshold = max_size * min_component_ratio

    # Create filtered binary image with only significant components
    filtered_binary = np.zeros_like(crack_binary)
    for label_id in range(1, num_labels):
        component_size = stats[label_id, cv2.CC_STAT_AREA]
        if component_size >= min_size_threshold:
            filtered_binary[labels == label_id] = 255

    # Step 3: Optional morphological closing to connect nearby fragments
    kernel_close = np.ones((2, 2), np.uint8)
    filtered_binary = cv2.morphologyEx(filtered_binary, cv2.MORPH_CLOSE, kernel_close)

    return filtered_binary


def calculate_intensity_based_width(
    skeleton: np.ndarray,
    grayscale: np.ndarray,
    mask: np.ndarray,
    scale_map: np.ndarray,
    sample_interval: int = 5,
    detection_method: str = 'percentile',
    min_component_ratio: float = 0.1,
    max_width_filter: float = None
) -> Tuple[float, float]:
    """
    Calculate crack width by detecting actual dark crack pixels within mask.

    This gives true crack width in pixels, not mask width.

    Args:
        skeleton: Binary skeleton image
        grayscale: Grayscale image
        mask: YOLO mask (candidate region)
        scale_map: Per-pixel mm/px scale map
        sample_interval: Sample every N skeleton pixels
        detection_method: Method for crack pixel detection
        min_component_ratio: Minimum component size as ratio of largest
        max_width_filter: Exclude width samples exceeding this value (mm)

    Returns:
        (average_width_mm, max_width_mm)
    """
    # Detect actual crack pixels within mask
    crack_binary = detect_crack_pixels_in_mask(grayscale, mask, detection_method, min_component_ratio)

    # Find skeleton pixels
    rows, cols = np.where(skeleton > 0)

    if len(rows) < 3:
        return 0.0, 0.0

    # Sample skeleton pixels
    n_pixels = len(rows)
    sample_indices = range(0, n_pixels, sample_interval)

    widths = []

    for idx in sample_indices:
        r, c = rows[idx], cols[idx]

        # Get scale at this pixel with fallback
        D = get_scale_with_fallback(scale_map, r, c)
        if D == 0:
            continue

        # Estimate local direction from nearby skeleton pixels
        window = 3
        nearby_rows = rows[max(0, idx-window):min(n_pixels, idx+window+1)]
        nearby_cols = cols[max(0, idx-window):min(n_pixels, idx+window+1)]

        if len(nearby_rows) < 2:
            continue

        # Fit line to get direction
        dr = nearby_rows[-1] - nearby_rows[0]
        dc = nearby_cols[-1] - nearby_cols[0]

        # Perpendicular direction (normal)
        length = np.sqrt(dr**2 + dc**2)
        if length < 1e-6:
            continue

        nr, nc = -dc / length, dr / length

        # Measure width along normal using detected crack pixels
        width_pixels = measure_width_along_normal(
            crack_binary, r, c, nr, nc, max_distance=50
        )

        if width_pixels > 0:
            widths.append(width_pixels * D)

    if not widths:
        return 0.0, 0.0

    # Filter out outliers exceeding max_width_filter
    if max_width_filter is not None and max_width_filter > 0:
        widths = [w for w in widths if w <= max_width_filter]
        if not widths:
            return 0.0, 0.0

    return float(np.mean(widths)), float(np.max(widths))


def visualize_measurement(
    rgb_img: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
    crack_binary: np.ndarray,
    scale_map: np.ndarray,
    cluster_id: int,
    segment_id: int,
    length_mm: float,
    avg_width_mm: float,
    max_width_mm: float,
    output_path: Path,
    sample_interval: int = 5
):
    """
    Create 4-panel visualization of measurement process.

    Panels:
    (a) RGB + Mask overlay
    (b) Skeleton (length calculation)
    (c) Detected crack pixels (intensity-based)
    (d) Width measurement lines
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Cluster {cluster_id} - Segment {segment_id}\n'
        f'Length: {length_mm:.2f}mm | Width: avg={avg_width_mm:.2f}mm, max={max_width_mm:.2f}mm',
        fontsize=14, fontweight='bold'
    )

    # (a) RGB + Mask overlay
    ax = axes[0, 0]
    rgb_display = rgb_img.copy()
    if len(rgb_display.shape) == 2:
        rgb_display = cv2.cvtColor(rgb_display, cv2.COLOR_GRAY2RGB)
    elif rgb_display.shape[2] == 3:
        rgb_display = cv2.cvtColor(rgb_display, cv2.COLOR_BGR2RGB)

    # Create mask overlay (semi-transparent yellow)
    overlay = rgb_display.copy()
    overlay[mask > 0] = [255, 255, 0]  # Yellow
    blended = cv2.addWeighted(rgb_display, 0.7, overlay, 0.3, 0)

    ax.imshow(blended)
    ax.set_title('(a) RGB + YOLO Mask', fontsize=12)
    ax.axis('off')

    # (b) Skeleton for length
    ax = axes[0, 1]
    skeleton_display = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    skeleton_display[mask > 0] = [50, 50, 50]  # Dark gray for mask
    skeleton_display[skeleton > 0] = [0, 255, 0]  # Green for skeleton

    ax.imshow(skeleton_display)
    ax.set_title('(b) Skeleton (Length Calculation)', fontsize=12)
    ax.axis('off')

    # Add skeleton pixel count
    n_skeleton = np.sum(skeleton > 0)
    ax.text(0.02, 0.98, f'Skeleton pixels: {n_skeleton}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # (c) Detected crack pixels
    ax = axes[1, 0]
    crack_display = np.zeros((*crack_binary.shape, 3), dtype=np.uint8)
    crack_display[mask > 0] = [100, 100, 100]  # Gray for mask region
    crack_display[crack_binary > 0] = [255, 0, 0]  # Red for crack pixels

    ax.imshow(crack_display)
    ax.set_title('(c) Detected Crack Pixels (Intensity)', fontsize=12)
    ax.axis('off')

    # Add crack pixel count
    n_crack = np.sum(crack_binary > 0)
    ratio = n_crack / max(np.sum(mask > 0), 1) * 100
    ax.text(0.02, 0.98, f'Crack pixels: {n_crack} ({ratio:.1f}% of mask)',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # (d) Width measurement lines
    ax = axes[1, 1]
    width_display = rgb_display.copy()

    # Draw skeleton
    width_display[skeleton > 0] = [0, 255, 0]

    # Draw width measurement lines
    rows, cols = np.where(skeleton > 0)
    if len(rows) > 3:
        n_pixels = len(rows)
        sample_indices = range(0, n_pixels, sample_interval)

        for idx in sample_indices:
            r, c = rows[idx], cols[idx]

            # Estimate direction
            window = 3
            nearby_rows = rows[max(0, idx-window):min(n_pixels, idx+window+1)]
            nearby_cols = cols[max(0, idx-window):min(n_pixels, idx+window+1)]

            if len(nearby_rows) < 2:
                continue

            dr = nearby_rows[-1] - nearby_rows[0]
            dc = nearby_cols[-1] - nearby_cols[0]
            length = np.sqrt(dr**2 + dc**2)
            if length < 1e-6:
                continue

            # Normal direction
            nr, nc = -dc / length, dr / length

            # Measure width
            width_pixels = measure_width_along_normal(crack_binary, r, c, nr, nc, 50)

            if width_pixels > 0:
                # Draw line
                half_width = width_pixels // 2
                r1 = int(r - half_width * nr)
                c1 = int(c - half_width * nc)
                r2 = int(r + half_width * nr)
                c2 = int(c + half_width * nc)

                cv2.line(width_display, (c1, r1), (c2, r2), (255, 0, 255), 1)  # Magenta

    ax.imshow(width_display)
    ax.set_title('(d) Width Measurement Lines', fontsize=12)
    ax.axis('off')

    # Add legend
    green_patch = mpatches.Patch(color='green', label='Skeleton')
    magenta_patch = mpatches.Patch(color='magenta', label='Width lines')
    ax.legend(handles=[green_patch, magenta_patch], loc='lower right', fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.debug(f"Saved visualization: {output_path}")


def calculate_edge_based_width(
    skeleton: np.ndarray,
    grayscale: np.ndarray,
    roi_mask: np.ndarray,
    scale_map: np.ndarray,
    sample_interval: int = 5,
    canny_low: int = 50,
    canny_high: int = 150
) -> Tuple[float, float]:
    """
    Calculate crack width using edge detection on grayscale image with per-pixel scale.

    This measures the actual crack boundaries based on intensity changes,
    not the YOLO mask boundaries.

    Args:
        skeleton: Binary skeleton image
        grayscale: Grayscale image (ROI region)
        roi_mask: ROI mask to limit edge detection area
        scale_map: Per-pixel mm/px scale map
        sample_interval: Sample every N skeleton pixels
        canny_low: Canny edge detection low threshold
        canny_high: Canny edge detection high threshold

    Returns:
        (average_width_mm, max_width_mm)
    """
    # Find skeleton pixels
    rows, cols = np.where(skeleton > 0)

    if len(rows) < 3:
        return 0.0, 0.0

    # Apply edge detection within ROI
    # Preprocess: enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(grayscale)

    # Apply ROI mask
    enhanced_roi = cv2.bitwise_and(enhanced, enhanced, mask=roi_mask.astype(np.uint8))

    # Edge detection
    edges = cv2.Canny(enhanced_roi, canny_low, canny_high)

    # Sample skeleton pixels
    n_pixels = len(rows)
    sample_indices = range(0, n_pixels, sample_interval)

    widths = []

    for idx in sample_indices:
        r, c = rows[idx], cols[idx]

        # Get scale at this pixel with fallback
        D = get_scale_with_fallback(scale_map, r, c)
        if D == 0:
            continue

        # Estimate local direction from nearby skeleton pixels
        window = 3
        nearby_rows = rows[max(0, idx-window):min(n_pixels, idx+window+1)]
        nearby_cols = cols[max(0, idx-window):min(n_pixels, idx+window+1)]

        if len(nearby_rows) < 2:
            continue

        # Fit line to get direction
        dr = nearby_rows[-1] - nearby_rows[0]
        dc = nearby_cols[-1] - nearby_cols[0]

        # Perpendicular direction (normal)
        length = np.sqrt(dr**2 + dc**2)
        if length < 1e-6:
            continue

        nr, nc = -dc / length, dr / length

        # Measure width along normal direction using edges
        width_pixels = measure_edge_width_along_normal(
            edges, r, c, nr, nc
        )

        if width_pixels > 0:
            widths.append(width_pixels * D)

    if not widths:
        return 0.0, 0.0

    return np.mean(widths), np.max(widths)


def measure_edge_width_along_normal(
    edges: np.ndarray,
    r: int, c: int,
    nr: float, nc: float,
    max_distance: int = 100
) -> int:
    """
    Measure crack width by finding edge pixels along normal direction.

    Returns:
        Width in pixels (distance between two edges)
    """
    height, width = edges.shape

    # Search in both directions along normal for edge pixels
    positive_edge = -1
    negative_edge = -1

    # Positive direction - find first edge
    for d in range(1, max_distance):
        new_r = int(round(r + d * nr))
        new_c = int(round(c + d * nc))

        if not (0 <= new_r < height and 0 <= new_c < width):
            break

        if edges[new_r, new_c] > 0:
            positive_edge = d
            break

    # Negative direction - find first edge
    for d in range(1, max_distance):
        new_r = int(round(r - d * nr))
        new_c = int(round(c - d * nc))

        if not (0 <= new_r < height and 0 <= new_c < width):
            break

        if edges[new_r, new_c] > 0:
            negative_edge = d
            break

    # Width = distance between two edges
    if positive_edge > 0 and negative_edge > 0:
        return positive_edge + negative_edge
    elif positive_edge > 0:
        return positive_edge * 2  # Estimate: double one side
    elif negative_edge > 0:
        return negative_edge * 2
    else:
        return 0


def measure_segment_2d(
    masks_dir: Path,
    image_id: str,
    mask_id: int,
    scale_map: np.ndarray,
    segment_uvs: List[List[float]] = None,
    image_shape: Tuple[int, int] = (2160, 3840),
    margin: int = 50,
    rgb_dir: Path = None,
    use_edge_width: bool = True,
    detection_method: str = 'percentile_30',
    sample_interval: int = 5,
    min_component_ratio: float = 0.1,
    max_width_filter: float = None,
    viz_dir: Path = None,
    cluster_id: int = 0,
    segment_id: int = 0
) -> Dict:
    """
    Measure a segment using 2D skeleton method with per-pixel scale map.

    Args:
        masks_dir: Path to YOLO masks
        image_id: Image identifier
        mask_id: Mask index
        scale_map: Per-pixel mm/px scale map (H, W)
        segment_uvs: List of [u, v] coordinates for this segment (for cropping)
        image_shape: (height, width)
        margin: Margin around bounding box for cropping
        rgb_dir: Path to RGB images (for edge-based width measurement)
        use_edge_width: Use edge detection for width measurement

    Returns:
        Dict with length_mm, avg_width_mm, max_width_mm
    """
    # Load mask polygon
    polygon = load_mask_polygon(masks_dir, image_id, mask_id)

    if not polygon:
        logger.warning(f"Could not load mask: {image_id}, mask {mask_id}")
        return {'length_mm': 0, 'avg_width_mm': 0, 'max_width_mm': 0}

    # Convert to binary mask
    binary_mask = polygon_to_binary_mask(polygon, image_shape)

    if binary_mask.sum() == 0:
        return {'length_mm': 0, 'avg_width_mm': 0, 'max_width_mm': 0}

    # Crop mask to segment region if UV coordinates provided
    if segment_uvs and len(segment_uvs) > 0:
        uvs = np.array(segment_uvs)
        u_coords = uvs[:, 0]
        v_coords = uvs[:, 1]

        # Calculate bounding box with margin
        u_min = max(0, int(np.min(u_coords) - margin))
        u_max = min(image_shape[1], int(np.max(u_coords) + margin))
        v_min = max(0, int(np.min(v_coords) - margin))
        v_max = min(image_shape[0], int(np.max(v_coords) + margin))

        # Crop the mask to segment region
        # Only keep mask pixels within the bounding box
        cropped_mask = np.zeros_like(binary_mask)
        cropped_mask[v_min:v_max, u_min:u_max] = binary_mask[v_min:v_max, u_min:u_max]
        binary_mask = cropped_mask

        if binary_mask.sum() == 0:
            return {'length_mm': 0, 'avg_width_mm': 0, 'max_width_mm': 0}

    # Skeletonize
    skeleton = skeletonize(binary_mask > 0)

    # Calculate length using per-pixel scale
    length_mm = calculate_skeleton_length(skeleton, scale_map)

    # Calculate width
    avg_width_mm, max_width_mm = 0.0, 0.0
    width_method = 'mask'

    # Try intensity-based width measurement if RGB image available
    if use_edge_width and rgb_dir:
        # Load RGB image
        rgb_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = rgb_dir / f"{image_id}{ext}"
            if candidate.exists():
                rgb_path = candidate
                break

        if rgb_path:
            rgb_img = cv2.imread(str(rgb_path))
            if rgb_img is not None:
                # Convert to grayscale
                grayscale = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

                # Check detection method
                if detection_method.startswith('direct'):
                    # Direct scan mode: scan grayscale along skeleton perpendicular
                    # Extract intensity threshold from method string (e.g., 'direct_0.6')
                    intensity_threshold = 0.7  # default
                    if '_' in detection_method:
                        try:
                            intensity_threshold = float(detection_method.split('_')[1])
                        except:
                            pass

                    avg_width_mm, max_width_mm = calculate_direct_scan_width(
                        skeleton, grayscale, binary_mask, scale_map,
                        sample_interval=sample_interval,
                        intensity_threshold=intensity_threshold
                    )
                    width_method = 'direct'
                else:
                    # Gradient/percentile based detection
                    avg_width_mm, max_width_mm = calculate_intensity_based_width(
                        skeleton, grayscale, binary_mask, scale_map,
                        sample_interval=sample_interval,
                        detection_method=detection_method,
                        min_component_ratio=min_component_ratio,
                        max_width_filter=max_width_filter
                    )
                    width_method = 'intensity'

                # Generate visualization if requested
                if viz_dir:
                    if detection_method.startswith('direct'):
                        # For direct mode, create simple threshold binary for visualization
                        # Get background intensity for threshold
                        kernel = np.ones((5, 5), np.uint8)
                        dilated = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=2)
                        edge_region = dilated - binary_mask.astype(np.uint8)
                        edge_pixels = grayscale[edge_region > 0]
                        if len(edge_pixels) > 0:
                            bg_intensity = np.median(edge_pixels)
                        else:
                            bg_intensity = np.median(grayscale[binary_mask > 0])
                        dark_thresh = bg_intensity * intensity_threshold
                        crack_binary = ((grayscale < dark_thresh) & (binary_mask > 0)).astype(np.uint8) * 255
                    else:
                        crack_binary = detect_crack_pixels_in_mask(
                            grayscale, binary_mask, detection_method, min_component_ratio
                        )
                    viz_path = viz_dir / f"cluster_{cluster_id}_seg_{segment_id}.png"
                    visualize_measurement(
                        rgb_img, binary_mask, skeleton, crack_binary, scale_map,
                        cluster_id, segment_id, length_mm, avg_width_mm, max_width_mm,
                        viz_path, sample_interval
                    )

    # Fallback to mask-based width if intensity detection failed or not used
    if avg_width_mm == 0.0 and max_width_mm == 0.0:
        avg_width_mm, max_width_mm = calculate_skeleton_width(
            skeleton, binary_mask, scale_map
        )
        width_method = 'mask'

    return {
        'length_mm': round(length_mm, 2),
        'avg_width_mm': round(avg_width_mm, 2),
        'max_width_mm': round(max_width_mm, 2),
        'width_method': width_method
    }


# =============================================================================
# Main Measurement Function
# =============================================================================

def compute_obb_3d(points_3d: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute Oriented Bounding Box in 3D using PCA.

    Args:
        points_3d: Nx3 array of 3D points

    Returns:
        (length, width, principal_axes, center)
        - length: Length along longest axis (mm)
        - width: Width along second axis (mm)
        - principal_axes: 3x3 array of eigenvectors
        - center: 3D centroid
    """
    center = np.mean(points_3d, axis=0)
    centered = points_3d - center

    # PCA
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalues (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project points onto principal axes
    projected = centered @ eigenvectors

    # Compute bounding box dimensions
    min_proj = np.min(projected, axis=0)
    max_proj = np.max(projected, axis=0)
    dimensions = max_proj - min_proj

    # Length = longest axis, Width = second axis
    length = dimensions[0] * 1000  # Convert to mm
    width = dimensions[1] * 1000   # Convert to mm

    return length, width, eigenvectors, center


def compute_polygon_area_2d(points_2d: np.ndarray, scale_map: np.ndarray) -> float:
    """
    Compute polygon area using Shoelace formula with per-pixel scale.

    Args:
        points_2d: Nx2 array of 2D pixel coordinates
        scale_map: HxW array of mm/pixel scale factors

    Returns:
        Area in mm²
    """
    if len(points_2d) < 3:
        return 0.0

    # Get convex hull to ensure proper polygon
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
    except:
        hull_points = points_2d

    # Sample scale factors at hull vertices
    scales = []
    for pt in hull_points:
        x, y = int(pt[0]), int(pt[1])
        h, w = scale_map.shape
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        scales.append(scale_map[y, x])

    avg_scale = np.mean(scales)

    # Shoelace formula for area in pixels
    x = hull_points[:, 0]
    y = hull_points[:, 1]
    area_pixels = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Convert to mm² using average scale
    area_mm2 = area_pixels * (avg_scale ** 2)

    return area_mm2


def calculate_mask_area_with_scale(
    binary_mask: np.ndarray,
    scale_map: np.ndarray
) -> float:
    """
    Calculate actual area by summing D×D for each mask pixel.

    This is more accurate than 3D OBB because:
    - Uses dense 2D mask (not sparse 3D points)
    - Accounts for perspective distortion via per-pixel scale
    - Directly uses YOLO detection boundary

    Args:
        binary_mask: Binary mask (HxW)
        scale_map: Per-pixel mm/px scale map (HxW)

    Returns:
        Area in mm²
    """
    rows, cols = np.where(binary_mask > 0)

    if len(rows) == 0:
        return 0.0

    total_area = 0.0
    for r, c in zip(rows, cols):
        D = get_scale_with_fallback(scale_map, r, c)
        if D > 0:
            total_area += D * D  # Each pixel contributes D×D mm²

    return total_area


def calculate_2d_obb_with_scale(
    binary_mask: np.ndarray,
    scale_map: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate OBB length/width using PCA on 2D pixel coordinates.
    Convert to mm using median scale of the mask region.

    Args:
        binary_mask: Binary mask (HxW)
        scale_map: Per-pixel mm/px scale map (HxW)

    Returns:
        (length_mm, width_mm) - OBB dimensions
    """
    rows, cols = np.where(binary_mask > 0)

    if len(rows) < 3:
        return 0.0, 0.0

    # Get representative scale (median of all mask pixels)
    scales = []
    for r, c in zip(rows, cols):
        D = get_scale_with_fallback(scale_map, r, c)
        if D > 0:
            scales.append(D)

    if not scales:
        return 0.0, 0.0

    D_median = np.median(scales)

    # PCA to find principal axes
    coords = np.column_stack([rows, cols])
    coords_centered = coords - coords.mean(axis=0)

    # Covariance matrix
    cov = np.cov(coords_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest = major axis)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project onto principal axes
    projected = coords_centered @ eigenvectors

    # Extent along each axis (in pixels)
    length_px = np.ptp(projected[:, 0])  # major axis range
    width_px = np.ptp(projected[:, 1])   # minor axis range

    # Convert to mm
    length_mm = length_px * D_median
    width_mm = width_px * D_median

    return length_mm, width_mm


def measure_obb_defect(
    cluster: Dict,
    crack_points_lookup: Dict[int, Dict],
    masks_dir: Path,
    scale_maps_dir: Path,
    image_shape: Tuple[int, int]
) -> Dict:
    """
    Measure non-crack defect using 2D mask + scale map.

    Strategy:
    - Primary: Area calculated from 2D mask with per-pixel scale (D×D summation)
    - Secondary: Length/Width from 2D OBB with median scale
    - Reference: 3D OBB for comparison (optional)

    This approach ensures consistency with crack measurement methodology,
    where all defects use 2D image + depth-calibrated scale maps.

    Args:
        cluster: Cluster dict from crack_clusters.json
        crack_points_lookup: point_id -> point dict
        masks_dir: Path to YOLO masks
        scale_maps_dir: Path to scale map .npy files
        image_shape: (height, width)

    Returns:
        Measurement dict with width, length, area (all scale-corrected)
    """
    cluster_id = cluster['cluster_id']
    defect_class = cluster.get('class', 'unknown')
    point_ids = cluster['point_ids']

    # Get full point data (exclude synthetic points)
    cluster_points = []
    for pid in point_ids:
        if pid in crack_points_lookup:
            point = crack_points_lookup[pid]
            if not point.get('is_synthetic', False):
                cluster_points.append(point)

    if not cluster_points:
        logger.warning(f"Cluster {cluster_id} ({defect_class}): No points found")
        return {
            'cluster_id': cluster_id,
            'class': defect_class,
            'width_mm': 0,
            'length_mm': 0,
            'area_mm2': 0,
            'method': 'area_2d_scaled',
            'n_points': 0
        }

    # Find best source mask (mask with most points)
    source_masks = cluster.get('source_masks', [])
    if not source_masks:
        logger.warning(f"Cluster {cluster_id} ({defect_class}): No source masks")
        # Fallback to 3D OBB if no masks
        points_3d = np.array([p['xyz'] for p in cluster_points])
        length_3d, width_3d, _, _ = compute_obb_3d(points_3d)
        return {
            'cluster_id': cluster_id,
            'class': defect_class,
            'width_mm': width_3d,
            'length_mm': length_3d,
            'area_mm2': length_3d * width_3d,
            'method': 'obb_3d_fallback',
            'n_points': len(cluster_points)
        }

    # Select best source (first one, or could select by max point count)
    best_source = source_masks[0]
    image_id = best_source['image_id']
    mask_id = best_source['mask_id']

    # Load scale map
    timestamp_key = image_id
    for prefix in ['camera_RGB_', 'camera_DPT_']:
        if image_id.startswith(prefix):
            timestamp_key = image_id[len(prefix):]
            break

    scale_map = None
    scale_map_patterns = [
        f"scale_map_iso_camera_DPT_{timestamp_key}.npy",
        f"scale_map_iso_{image_id}.npy",
        f"scale_map_iso_{timestamp_key}.npy",
    ]

    for pattern in scale_map_patterns:
        scale_map_path = scale_maps_dir / pattern
        if scale_map_path.exists():
            scale_map = np.load(scale_map_path)
            break

    if scale_map is None:
        logger.warning(f"Cluster {cluster_id} ({defect_class}): Scale map not found for {image_id}")
        # Fallback to 3D OBB
        points_3d = np.array([p['xyz'] for p in cluster_points])
        length_3d, width_3d, _, _ = compute_obb_3d(points_3d)
        return {
            'cluster_id': cluster_id,
            'class': defect_class,
            'width_mm': width_3d,
            'length_mm': length_3d,
            'area_mm2': length_3d * width_3d,
            'method': 'obb_3d_fallback',
            'n_points': len(cluster_points)
        }

    # Load mask polygon from YOLO detection
    polygon = load_mask_polygon(masks_dir, image_id, mask_id)

    if not polygon:
        logger.warning(f"Cluster {cluster_id} ({defect_class}): Could not load mask polygon for {image_id}, mask {mask_id}")
        # Fallback to 3D OBB
        points_3d = np.array([p['xyz'] for p in cluster_points])
        length_3d, width_3d, _, _ = compute_obb_3d(points_3d)
        return {
            'cluster_id': cluster_id,
            'class': defect_class,
            'width_mm': width_3d,
            'length_mm': length_3d,
            'area_mm2': length_3d * width_3d,
            'method': 'obb_3d_fallback',
            'n_points': len(cluster_points)
        }

    # Convert polygon to binary mask
    binary_mask = polygon_to_binary_mask(polygon, image_shape)

    if binary_mask.sum() == 0:
        logger.warning(f"Cluster {cluster_id} ({defect_class}): Empty mask")
        return {
            'cluster_id': cluster_id,
            'class': defect_class,
            'width_mm': 0,
            'length_mm': 0,
            'area_mm2': 0,
            'method': 'area_2d_scaled',
            'n_points': len(cluster_points)
        }

    # === NEW: 2D Scale-based Measurement ===

    # 1. Calculate precise area using per-pixel scale (D×D summation)
    area_mm2 = calculate_mask_area_with_scale(binary_mask, scale_map)

    # 2. Calculate OBB length/width using 2D PCA with median scale
    length_mm, width_mm = calculate_2d_obb_with_scale(binary_mask, scale_map)

    # 3. Optional: Compute 3D OBB for comparison/validation
    points_3d = np.array([p['xyz'] for p in cluster_points])
    length_3d, width_3d, _, _ = compute_obb_3d(points_3d)

    logger.info(f"Cluster {cluster_id} ({defect_class}): "
                f"Area={area_mm2:.1f}mm² (2D scaled), "
                f"L={length_mm:.1f}mm, W={width_mm:.1f}mm (2D OBB), "
                f"L3D={length_3d:.1f}mm, W3D={width_3d:.1f}mm (3D OBB ref)")

    return {
        'cluster_id': cluster_id,
        'class': defect_class,
        'area_mm2': area_mm2,           # Primary metric (2D + scale)
        'length_mm': length_mm,         # 2D OBB major axis
        'width_mm': width_mm,           # 2D OBB minor axis
        'length_3d_mm': length_3d,      # 3D OBB reference
        'width_3d_mm': width_3d,        # 3D OBB reference
        'method': 'area_2d_scaled',
        'n_points': len(cluster_points),
        'image_id': image_id,
        'mask_id': mask_id
    }


def measure_cluster(
    cluster: Dict,
    crack_points_lookup: Dict[int, Dict],
    masks_dir: Path,
    scale_maps_dir: Path,
    image_shape: Tuple[int, int],
    n_segments: int = 5,
    rgb_dir: Path = None,
    use_edge_width: bool = True,
    detection_method: str = 'percentile_30',
    sample_interval: int = 5,
    min_component_ratio: float = 0.1,
    max_width_filter: float = None,
    viz_dir: Path = None
) -> Dict:
    """
    Measure a single cluster.
    Routes to appropriate measurement method based on defect class.

    Args:
        cluster: Cluster dict from crack_clusters.json
        crack_points_lookup: point_id -> point dict
        masks_dir: Path to YOLO masks
        scale_maps_dir: Path to scale map .npy files
        image_shape: (height, width)
        n_segments: Number of segments
        rgb_dir: Path to RGB images (for edge-based width)
        use_edge_width: Use edge detection for width measurement

    Returns:
        Measurement dict
    """
    cluster_id = cluster['cluster_id']
    defect_class = cluster.get('class', 'crack')

    # Route based on defect class
    if defect_class != 'crack':
        # Use OBB + Area for non-crack defects
        return measure_obb_defect(
            cluster, crack_points_lookup, masks_dir, scale_maps_dir, image_shape
        )

    # Below is the original crack measurement logic
    cluster_id = cluster['cluster_id']
    point_ids = cluster['point_ids']

    # Get full point data (exclude synthetic points)
    cluster_points = []
    for pid in point_ids:
        if pid in crack_points_lookup:
            point = crack_points_lookup[pid]
            # Skip synthetic points - they don't have valid source/UV information
            if not point.get('is_synthetic', False):
                cluster_points.append(point)

    if not cluster_points:
        logger.warning(f"Cluster {cluster_id}: No points found")
        return {
            'cluster_id': cluster_id,
            'total_length_mm': 0,
            'avg_width_mm': 0,
            'max_width_mm': 0,
            'n_segments': 0,
            'segments': []
        }

    # Segment cluster in 3D
    segments = segment_cluster_3d(cluster_points, n_segments)
    logger.info(f"Cluster {cluster_id}: Created {len(segments)} segments (requested {n_segments})")

    # Measure each segment in 2D
    segment_measurements = []

    for segment in segments:
        # Find best mask for this segment
        best_mask = find_best_mask_for_segment(segment)

        if not best_mask:
            # Count points with source_masks for debugging
            n_points = len(segment['points'])
            n_with_sources = sum(1 for p in segment['points']
                                 if p.get('source_masks') and not p.get('is_synthetic', False))
            logger.warning(f"Segment {segment['segment_id']}: No mask found "
                          f"(points={n_points}, with_sources={n_with_sources})")
            continue

        image_id, mask_id = best_mask

        # Load scale map for this image
        # Extract timestamp part from image_id (e.g., camera_RGB_1761702052_213355008 -> 1761702052_213355008)
        timestamp_key = image_id
        for prefix in ['camera_RGB_', 'camera_DPT_']:
            if image_id.startswith(prefix):
                timestamp_key = image_id[len(prefix):]
                break

        # Try to find scale map file
        # d2c_and_pixel_scale.py saves as: scale_map_iso_camera_DPT_{timestamp}.npy
        scale_map = None
        scale_map_patterns = [
            f"scale_map_iso_camera_DPT_{timestamp_key}.npy",
            f"scale_map_iso_{image_id}.npy",
            f"scale_map_iso_{timestamp_key}.npy",
        ]

        for pattern in scale_map_patterns:
            scale_map_path = scale_maps_dir / pattern
            if scale_map_path.exists():
                scale_map = np.load(scale_map_path)
                break

        if scale_map is None:
            logger.warning(f"No scale map for {image_id}")
            continue

        # Extract UV coordinates for this segment from points that belong to best_mask
        # Note: Skip synthetic points as they don't have valid UV coordinates
        segment_uvs = []
        for point in segment['points']:
            if point.get('is_synthetic', False):
                continue
            for source in point.get('source_masks', []):
                if source['image_id'] == image_id and source['mask_id'] == mask_id:
                    if 'uv' in source:
                        segment_uvs.append(source['uv'])
                    break

        # Measure in 2D with segment cropping and intensity-based width
        measurement = measure_segment_2d(
            masks_dir, image_id, mask_id, scale_map, segment_uvs, image_shape,
            margin=50, rgb_dir=rgb_dir, use_edge_width=use_edge_width,
            detection_method=detection_method, sample_interval=sample_interval,
            min_component_ratio=min_component_ratio, max_width_filter=max_width_filter,
            viz_dir=viz_dir, cluster_id=cluster_id, segment_id=segment['segment_id']
        )

        measurement['segment_id'] = segment['segment_id']
        measurement['image_id'] = image_id
        measurement['mask_id'] = mask_id
        measurement['n_points'] = len(segment['point_ids'])

        segment_measurements.append(measurement)

    # Aggregate measurements
    if segment_measurements:
        total_length = sum(m['length_mm'] for m in segment_measurements)
        avg_widths = [m['avg_width_mm'] for m in segment_measurements if m['avg_width_mm'] > 0]
        max_widths = [m['max_width_mm'] for m in segment_measurements if m['max_width_mm'] > 0]

        avg_width = np.mean(avg_widths) if avg_widths else 0
        max_width = max(max_widths) if max_widths else 0
    else:
        total_length = 0
        avg_width = 0
        max_width = 0

    return {
        'cluster_id': cluster_id,
        'total_length_mm': round(total_length, 2),
        'avg_width_mm': round(avg_width, 2),
        'max_width_mm': round(max_width, 2),
        'n_segments': len(segment_measurements),
        'n_points': len(cluster_points),
        'segments': segment_measurements
    }


def run_measurement(
    clusters_json: str,
    crack_points_json: str,
    masks_dir: str,
    scale_maps_dir: str,
    output_json: str,
    image_width: int = 3840,
    image_height: int = 2160,
    n_segments: int = 5,
    rgb_dir: str = None,
    use_edge_width: bool = True,
    detection_method: str = 'gradient',
    crack_percentile: float = 30.0,
    gradient_percentile: float = 70.0,
    intensity_threshold: float = 0.7,
    sample_interval: int = 5,
    min_component_ratio: float = 0.1,
    max_width_filter: float = None,
    viz_dir: str = None
):
    """
    Run measurement on all clusters using per-pixel scale maps.
    """
    logger.info("=" * 80)
    logger.info("Cluster Measurement")
    logger.info("=" * 80)

    # Load inputs
    logger.info(f"Loading clusters: {clusters_json}")
    with open(clusters_json) as f:
        clusters_data = json.load(f)

    logger.info(f"Loading crack points: {crack_points_json}")
    with open(crack_points_json) as f:
        crack_points_data = json.load(f)

    # Build lookup tables
    crack_points_lookup = {
        p['point_id']: p for p in crack_points_data['points']
    }

    masks_path = Path(masks_dir)
    scale_maps_path = Path(scale_maps_dir)
    rgb_path = Path(rgb_dir) if rgb_dir else None
    viz_path = Path(viz_dir) if viz_dir else None
    image_shape = (image_height, image_width)

    clusters = clusters_data.get('clusters', [])
    logger.info(f"Measuring {len(clusters)} clusters...")
    logger.info(f"Scale maps directory: {scale_maps_dir}")
    if rgb_path and use_edge_width:
        logger.info(f"Using intensity-based width measurement with RGB from: {rgb_dir}")
        logger.info(f"  Detection method: {detection_method}")
        if detection_method == 'percentile':
            logger.info(f"  Crack percentile: {crack_percentile}%")
        logger.info(f"  Sample interval: {sample_interval}")
    else:
        logger.info("Using mask-based width measurement")

    if viz_path:
        logger.info(f"Visualization output: {viz_dir}")

    # Generate colors for clusters (same as visualization)
    cluster_colors = generate_cluster_colors(len(clusters))

    # Measure each cluster
    measurements = []

    for idx, cluster in enumerate(clusters):
        # Build detection method string with threshold value
        method_str = detection_method
        if detection_method == 'percentile':
            method_str = f'percentile_{crack_percentile}'
        elif detection_method == 'gradient':
            method_str = f'gradient_{gradient_percentile}'
        elif detection_method == 'direct':
            method_str = f'direct_{intensity_threshold}'

        measurement = measure_cluster(
            cluster,
            crack_points_lookup,
            masks_path,
            scale_maps_path,
            image_shape,
            n_segments,
            rgb_path,
            use_edge_width,
            method_str,
            sample_interval,
            min_component_ratio,
            max_width_filter,
            viz_path
        )

        # Add color information
        color_rgb = cluster_colors[idx] if idx < len(cluster_colors) else (128, 128, 128)
        color_name = rgb_to_color_name(color_rgb)
        measurement['color'] = {
            'rgb': list(color_rgb),
            'hex': f"#{color_rgb[0]:02X}{color_rgb[1]:02X}{color_rgb[2]:02X}",
            'name': color_name
        }

        measurements.append(measurement)

        # Log measurement based on defect class
        defect_class = measurement.get('class', 'crack')
        if defect_class == 'crack':
            logger.info(
                f"Cluster {measurement['cluster_id']} ({color_name}): "
                f"L={measurement['total_length_mm']:.1f}mm, "
                f"W_avg={measurement['avg_width_mm']:.2f}mm, "
                f"W_max={measurement['max_width_mm']:.2f}mm"
            )
        else:
            logger.info(
                f"Cluster {measurement['cluster_id']} ({defect_class}): "
                f"L={measurement['length_mm']:.1f}mm, "
                f"W={measurement['width_mm']:.1f}mm, "
                f"A={measurement['area_mm2']:.1f}mm²"
            )

    # Sort by length (largest first), handling both crack and non-crack measurements
    def get_length(m):
        if m.get('class', 'crack') == 'crack':
            return m.get('total_length_mm', 0)
        else:
            return m.get('length_mm', 0)

    measurements.sort(key=get_length, reverse=True)

    # Output
    output_data = {
        'metadata': {
            'n_clusters': len(measurements),
            'n_segments_per_cluster': n_segments,
            'image_shape': [image_height, image_width]
        },
        'measurements': measurements
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nSaved measurements: {output_json}")
    logger.info("=" * 80)

    return measurements


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Measure crack clusters using 3D segmentation + 2D pixel measurement'
    )

    parser.add_argument('--clusters', required=True,
                       help='Input crack_clusters.json')
    parser.add_argument('--crack-points', required=True,
                       help='Input crack_points.json')
    parser.add_argument('--masks-dir', required=True,
                       help='YOLO masks directory')
    parser.add_argument('--scale-maps-dir', required=True,
                       help='Directory containing scale_map_iso_*.npy files')
    parser.add_argument('--output', required=True,
                       help='Output cluster_measurements.json')
    parser.add_argument('--image-width', type=int, default=3840)
    parser.add_argument('--image-height', type=int, default=2160)
    parser.add_argument('--n-segments', type=int, default=5,
                       help='Number of segments per cluster (default: 5)')
    parser.add_argument('--rgb-dir', default=None,
                       help='RGB images directory (for intensity-based width measurement)')
    parser.add_argument('--no-edge-width', action='store_true',
                       help='Disable intensity-based width measurement (use mask-based)')
    parser.add_argument('--detection-method', default='gradient',
                       choices=['gradient', 'percentile', 'adaptive', 'otsu', 'direct'],
                       help='Crack detection: gradient (edge), percentile (dark), direct (skeleton scan)')
    parser.add_argument('--crack-percentile', type=float, default=30.0,
                       help='Percentile threshold for dark pixels (default: 30, lower=stricter)')
    parser.add_argument('--gradient-percentile', type=float, default=70.0,
                       help='Percentile threshold for gradient (default: 70, higher=stricter edge detection)')
    parser.add_argument('--intensity-threshold', type=float, default=0.7,
                       help='For direct mode: dark pixel threshold as ratio of background (default: 0.7)')
    parser.add_argument('--min-component-ratio', type=float, default=0.1,
                       help='Min component size as ratio of largest (default: 0.1, higher=stricter)')
    parser.add_argument('--max-width-filter', type=float, default=None,
                       help='Exclude width samples exceeding this value in mm (default: None)')
    parser.add_argument('--sample-interval', type=int, default=5,
                       help='Sample every N skeleton pixels for width (default: 5)')
    parser.add_argument('--viz-dir', default=None,
                       help='Output directory for measurement visualization images')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        measurements = run_measurement(
            args.clusters,
            args.crack_points,
            args.masks_dir,
            args.scale_maps_dir,
            args.output,
            args.image_width,
            args.image_height,
            args.n_segments,
            args.rgb_dir,
            not args.no_edge_width,
            args.detection_method,
            args.crack_percentile,
            args.gradient_percentile,
            args.intensity_threshold,
            args.sample_interval,
            args.min_component_ratio,
            args.max_width_filter,
            args.viz_dir
        )

        if measurements:
            print(f"\n✅ Measurement complete!")
            print(f"   Total clusters: {len(measurements)}")
            if measurements:
                largest = measurements[0]
                print(f"   Largest crack: {largest['total_length_mm']:.1f}mm length")
            print(f"   Output: {args.output}")
        else:
            print(f"\n⚠️ No measurements generated")

    except Exception as e:
        logger.error(f"Measurement failed: {e}", exc_info=True)
        import sys
        sys.exit(1)