#!/usr/bin/env python3
"""
Generate Figure 4 for paper: Scale-calibrated Measurement Comparison

Creates a 3-panel figure showing:
(a) Crack measurement (skeleton-based)
(b) Non-crack measurement (area-based)
(c) Scale map application comparison

Usage:
    python -m src.generate_figure4 \
        --clusters outputs/crack_clusters.json \
        --crack-points outputs/crack_points.json \
        --masks-dir outputs/yolo_masks \
        --scale-maps-dir outputs/scale_maps \
        --rgb-dir data/rgb \
        --output figures/figure4_measurement_comparison.png
"""

import json
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


def load_mask_polygon(masks_dir: Path, image_id: str, mask_id: int) -> Optional[list]:
    """Load mask polygon from YOLO masks JSON."""
    patterns = [f"{image_id}.json", f"{image_id}.png.json"]

    for pattern in patterns:
        json_path = masks_dir / pattern
        if json_path.exists():
            with open(json_path) as f:
                masks_data = json.load(f)

            masks_list = None
            if isinstance(masks_data, dict) and 'masks' in masks_data:
                masks_list = masks_data['masks']
            elif isinstance(masks_data, list):
                masks_list = masks_data

            if masks_list and mask_id < len(masks_list):
                mask = masks_list[mask_id]
                return mask.get('polygon', [])

    return None


def polygon_to_binary_mask(polygon: list, image_shape: Tuple[int, int]) -> np.ndarray:
    """Convert polygon to binary mask."""
    from skimage.draw import polygon as draw_polygon

    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    if len(polygon) < 3:
        return mask

    cols = np.array([p[0] for p in polygon])
    rows = np.array([p[1] for p in polygon])

    cols = np.clip(cols, 0, width - 1)
    rows = np.clip(rows, 0, height - 1)

    rr, cc = draw_polygon(rows, cols, shape=(height, width))
    mask[rr, cc] = 1

    return mask


def get_scale_with_fallback(scale_map: np.ndarray, r: int, c: int, search_radius: int = 20) -> float:
    """Get scale value at (r, c) with fallback to nearby valid pixels."""
    H, W = scale_map.shape

    if 0 <= r < H and 0 <= c < W:
        val = scale_map[r, c]
        if val > 0:
            return float(val)

    for radius in range(1, search_radius + 1):
        valid_values = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
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


def calculate_skeleton_length(skeleton: np.ndarray, scale_map: np.ndarray) -> float:
    """Calculate crack length from skeleton using scale map."""
    rows, cols = np.where(skeleton > 0)

    if len(rows) < 2:
        return 0.0

    skeleton_pixels = set(zip(rows, cols))
    total_length = 0.0
    visited_edges = set()

    for r, c in skeleton_pixels:
        D = get_scale_with_fallback(scale_map, r, c)
        if D == 0:
            continue

        neighbors = [
            (r-1, c), (r+1, c), (r, c-1), (r, c+1),
            (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)
        ]

        for i, (nr, nc) in enumerate(neighbors):
            if (nr, nc) in skeleton_pixels:
                edge = tuple(sorted([(r, c), (nr, nc)]))

                if edge not in visited_edges:
                    visited_edges.add(edge)
                    D_neighbor = get_scale_with_fallback(scale_map, nr, nc)
                    if D_neighbor == 0:
                        D_neighbor = D
                    D_avg = (D + D_neighbor) / 2.0

                    if i < 4:  # H or V
                        total_length += D_avg
                    else:  # Diagonal
                        total_length += D_avg * np.sqrt(2)

    return total_length


def calculate_mask_area_with_scale(binary_mask: np.ndarray, scale_map: np.ndarray) -> float:
    """Calculate area using per-pixel scale (D×D summation)."""
    rows, cols = np.where(binary_mask > 0)

    if len(rows) == 0:
        return 0.0

    total_area = 0.0
    for r, c in zip(rows, cols):
        D = get_scale_with_fallback(scale_map, r, c)
        if D > 0:
            total_area += D * D

    return total_area


def calculate_2d_obb_with_scale(binary_mask: np.ndarray, scale_map: np.ndarray) -> Tuple[float, float]:
    """Calculate OBB length/width using PCA with median scale."""
    rows, cols = np.where(binary_mask > 0)

    if len(rows) < 3:
        return 0.0, 0.0

    scales = []
    for r, c in zip(rows, cols):
        D = get_scale_with_fallback(scale_map, r, c)
        if D > 0:
            scales.append(D)

    if not scales:
        return 0.0, 0.0

    D_median = np.median(scales)

    coords = np.column_stack([rows, cols])
    coords_centered = coords - coords.mean(axis=0)

    cov = np.cov(coords_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    projected = coords_centered @ eigenvectors

    length_px = np.ptp(projected[:, 0])
    width_px = np.ptp(projected[:, 1])

    length_mm = length_px * D_median
    width_mm = width_px * D_median

    return length_mm, width_mm, eigenvectors, coords.mean(axis=0)


def draw_obb_on_image(img: np.ndarray, binary_mask: np.ndarray, eigenvectors: np.ndarray,
                      center: np.ndarray, length_px: float, width_px: float, color=(255, 0, 255)):
    """Draw oriented bounding box on image."""
    # Calculate corner points
    axis1 = eigenvectors[:, 0]
    axis2 = eigenvectors[:, 1]

    corners = [
        center + (length_px/2) * axis1 + (width_px/2) * axis2,
        center + (length_px/2) * axis1 - (width_px/2) * axis2,
        center - (length_px/2) * axis1 - (width_px/2) * axis2,
        center - (length_px/2) * axis1 + (width_px/2) * axis2
    ]

    corners = np.array(corners, dtype=np.int32)
    cv2.polylines(img, [corners], True, color, 2)

    return img


def generate_panel_a_crack(
    rgb_img: np.ndarray,
    binary_mask: np.ndarray,
    skeleton: np.ndarray,
    scale_map: np.ndarray,
    length_mm: float,
    avg_width_mm: float
) -> np.ndarray:
    """Generate panel (a): Crack measurement visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('(a) Crack Measurement (Skeleton-based)', fontsize=14, fontweight='bold')

    # Convert RGB
    if len(rgb_img.shape) == 2:
        rgb_display = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
    elif rgb_img.shape[2] == 3:
        rgb_display = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    else:
        rgb_display = rgb_img.copy()

    # Panel 1: Original mask
    ax = axes[0]
    overlay = rgb_display.copy()
    overlay[binary_mask > 0] = [255, 255, 0]
    blended = cv2.addWeighted(rgb_display, 0.6, overlay, 0.4, 0)
    ax.imshow(blended)
    ax.set_title('Original Mask', fontsize=11)
    ax.axis('off')

    # Panel 2: Skeleton extraction
    ax = axes[1]
    skeleton_display = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    skeleton_display[binary_mask > 0] = [50, 50, 50]
    skeleton_display[skeleton > 0] = [0, 255, 0]
    ax.imshow(skeleton_display)
    ax.set_title('Skeleton Extraction', fontsize=11)
    ax.axis('off')

    # Panel 3: Scale applied measurement
    ax = axes[2]
    result = rgb_display.copy()
    result[skeleton > 0] = [0, 255, 0]
    ax.imshow(result)
    ax.set_title('Scale Applied', fontsize=11)
    ax.axis('off')

    # Add measurement text
    text_str = f'Length: {length_mm:.1f}mm\nWidth: {avg_width_mm:.2f}mm'
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.tight_layout()

    # Convert to numpy array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img_array


def generate_panel_b_noncrack(
    rgb_img: np.ndarray,
    binary_mask: np.ndarray,
    scale_map: np.ndarray,
    area_mm2: float,
    length_mm: float,
    width_mm: float
) -> np.ndarray:
    """Generate panel (b): Non-crack measurement visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('(b) Non-crack Measurement (Area-based)', fontsize=14, fontweight='bold')

    # Convert RGB
    if len(rgb_img.shape) == 2:
        rgb_display = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
    elif rgb_img.shape[2] == 3:
        rgb_display = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    else:
        rgb_display = rgb_img.copy()

    # Panel 1: Original mask
    ax = axes[0]
    overlay = rgb_display.copy()
    overlay[binary_mask > 0] = [255, 100, 100]
    blended = cv2.addWeighted(rgb_display, 0.6, overlay, 0.4, 0)
    ax.imshow(blended)
    ax.set_title('Original Mask', fontsize=11)
    ax.axis('off')

    # Panel 2: Pixel-wise area calculation
    ax = axes[1]
    # Create heatmap showing scale values
    scale_overlay = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
    rows, cols = np.where(binary_mask > 0)
    if len(rows) > 0:
        scales = np.array([get_scale_with_fallback(scale_map, r, c) for r, c in zip(rows, cols)])
        scale_min, scale_max = scales.min(), scales.max()
        for r, c, s in zip(rows, cols, scales):
            # Normalize scale to color
            normalized = (s - scale_min) / (scale_max - scale_min + 1e-6)
            color_val = int(normalized * 255)
            scale_overlay[r, c] = [color_val, 100, 255 - color_val]

    ax.imshow(scale_overlay)
    ax.set_title('Pixel-wise Scale (D²)', fontsize=11)
    ax.axis('off')

    # Panel 3: OBB extraction
    ax = axes[2]
    result = rgb_display.copy()
    overlay = result.copy()
    overlay[binary_mask > 0] = [255, 100, 100]
    result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)

    # Draw OBB
    length_mm_obb, width_mm_obb, eigenvectors, center = calculate_2d_obb_with_scale(binary_mask, scale_map)
    if length_mm_obb > 0:
        # Get median scale for pixel conversion
        rows, cols = np.where(binary_mask > 0)
        scales = [get_scale_with_fallback(scale_map, r, c) for r, c in zip(rows, cols)]
        D_median = np.median([s for s in scales if s > 0])

        length_px = length_mm_obb / D_median
        width_px = width_mm_obb / D_median

        result = draw_obb_on_image(result, binary_mask, eigenvectors, center, length_px, width_px)

    ax.imshow(result)
    ax.set_title('OBB Extraction', fontsize=11)
    ax.axis('off')

    # Add measurement text
    text_str = f'Area: {area_mm2:.0f}mm²\nMajor: {length_mm:.1f}mm\nMinor: {width_mm:.1f}mm'
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.tight_layout()

    # Convert to numpy array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img_array


def generate_panel_c_scalemap(scale_map: np.ndarray, rgb_img: np.ndarray) -> np.ndarray:
    """Generate panel (c): Scale map application comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('(c) Scale Map Application - Perspective Distortion Correction',
                 fontsize=14, fontweight='bold')

    # Convert RGB
    if len(rgb_img.shape) == 2:
        rgb_display = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
    elif rgb_img.shape[2] == 3:
        rgb_display = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    else:
        rgb_display = rgb_img.copy()

    # Panel 1: RGB Image
    ax = axes[0]
    ax.imshow(rgb_display)
    ax.set_title('RGB Image', fontsize=11)
    ax.axis('off')

    # Panel 2: Scale Map Heatmap
    ax = axes[1]
    # Mask invalid values
    scale_valid = scale_map.copy()
    scale_valid[scale_valid == 0] = np.nan

    im = ax.imshow(scale_valid, cmap='jet', interpolation='nearest')
    ax.set_title('Scale Map (mm/pixel)', fontsize=11)
    ax.axis('off')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('mm/pixel', rotation=270, labelpad=15)

    # Panel 3: Comparison visualization
    ax = axes[2]

    # Create comparison image showing two example regions
    H, W = scale_map.shape

    # Find near and far regions
    valid_mask = scale_map > 0
    if valid_mask.any():
        near_region = scale_map[valid_mask].min()
        far_region = scale_map[valid_mask].max()

        # Draw on RGB
        comparison = rgb_display.copy()

        # Find example pixels
        near_pixels = np.where((scale_map > 0) & (scale_map < near_region * 1.5))
        far_pixels = np.where((scale_map > far_region * 0.7) & (scale_map > 0))

        # Draw rectangles at example regions
        if len(near_pixels[0]) > 0:
            nr, nc = near_pixels[0][len(near_pixels[0])//2], near_pixels[1][len(near_pixels[1])//2]
            cv2.rectangle(comparison, (nc-50, nr-50), (nc+50, nr+50), (0, 255, 0), 3)
            cv2.putText(comparison, f'Near: {near_region:.2f}mm/px', (nc-50, nr-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if len(far_pixels[0]) > 0:
            fr, fc = far_pixels[0][len(far_pixels[0])//2], far_pixels[1][len(far_pixels[1])//2]
            cv2.rectangle(comparison, (fc-50, fr-50), (fc+50, fr+50), (255, 0, 0), 3)
            cv2.putText(comparison, f'Far: {far_region:.2f}mm/px', (fc-50, fr-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        ax.imshow(comparison)
        ax.set_title('Perspective Correction Effect', fontsize=11)
        ax.axis('off')

        # Add explanation text
        text_str = f'Same 100px → Near: {near_region*100:.0f}mm\n' \
                   f'             Far: {far_region*100:.0f}mm'
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.tight_layout()

    # Convert to numpy array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img_array


def generate_figure4(
    clusters_json: str,
    crack_points_json: str,
    masks_dir: str,
    scale_maps_dir: str,
    rgb_dir: str,
    output_path: str,
    crack_cluster_id: Optional[int] = None,
    noncrack_cluster_id: Optional[int] = None,
    image_shape: Tuple[int, int] = (2160, 3840)
):
    """
    Generate Figure 4 for paper.

    Args:
        clusters_json: Path to crack_clusters.json
        crack_points_json: Path to crack_points.json
        masks_dir: Path to YOLO masks directory
        scale_maps_dir: Path to scale maps directory
        rgb_dir: Path to RGB images directory
        output_path: Output figure path
        crack_cluster_id: Specific crack cluster ID (or None for auto-select)
        noncrack_cluster_id: Specific non-crack cluster ID (or None for auto-select)
        image_shape: (height, width)
    """
    logger.info("=" * 80)
    logger.info("Generating Figure 4: Scale-calibrated Measurement Comparison")
    logger.info("=" * 80)

    # Load data
    with open(clusters_json) as f:
        clusters_data = json.load(f)

    with open(crack_points_json) as f:
        crack_points_data = json.load(f)

    clusters = clusters_data['clusters']
    crack_points = {p['point_id']: p for p in crack_points_data['points']}

    # Select clusters
    crack_clusters = [c for c in clusters if c.get('class', 'crack') == 'crack']
    noncrack_clusters = [c for c in clusters if c.get('class', 'crack') != 'crack']

    if not crack_clusters:
        logger.error("No crack clusters found!")
        return

    # Auto-select crack cluster (largest)
    if crack_cluster_id is None:
        crack_cluster = max(crack_clusters, key=lambda x: x['n_points'])
        logger.info(f"Auto-selected crack cluster {crack_cluster['cluster_id']} ({crack_cluster['n_points']} points)")
    else:
        crack_cluster = next((c for c in crack_clusters if c['cluster_id'] == crack_cluster_id), None)
        if not crack_cluster:
            logger.error(f"Crack cluster {crack_cluster_id} not found!")
            return

    # Auto-select non-crack cluster
    if noncrack_clusters:
        if noncrack_cluster_id is None:
            noncrack_cluster = max(noncrack_clusters, key=lambda x: x['n_points'])
            logger.info(f"Auto-selected non-crack cluster {noncrack_cluster['cluster_id']} ({noncrack_cluster.get('class', 'unknown')})")
        else:
            noncrack_cluster = next((c for c in noncrack_clusters if c['cluster_id'] == noncrack_cluster_id), None)
            if not noncrack_cluster:
                logger.error(f"Non-crack cluster {noncrack_cluster_id} not found!")
                return
    else:
        logger.warning("No non-crack clusters found, will skip panel (b)")
        noncrack_cluster = None

    masks_path = Path(masks_dir)
    scale_maps_path = Path(scale_maps_dir)
    rgb_path = Path(rgb_dir)

    # Generate Panel (a): Crack
    logger.info("Generating panel (a): Crack measurement...")
    crack_source = crack_cluster['source_masks'][0]
    image_id = crack_source['image_id']
    mask_id = crack_source['mask_id']

    # Load RGB
    rgb_img = None
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = rgb_path / f"{image_id}{ext}"
        if candidate.exists():
            rgb_img = cv2.imread(str(candidate))
            break

    if rgb_img is None:
        logger.error(f"RGB image not found for {image_id}")
        return

    # Load scale map
    timestamp_key = image_id
    for prefix in ['camera_RGB_', 'camera_DPT_']:
        if image_id.startswith(prefix):
            timestamp_key = image_id[len(prefix):]
            break

    scale_map_patterns = [
        f"scale_map_iso_camera_DPT_{timestamp_key}.npy",
        f"scale_map_iso_{image_id}.npy",
        f"scale_map_iso_{timestamp_key}.npy",
    ]

    scale_map = None
    for pattern in scale_map_patterns:
        scale_map_path = scale_maps_path / pattern
        if scale_map_path.exists():
            scale_map = np.load(scale_map_path)
            break

    if scale_map is None:
        logger.error(f"Scale map not found for {image_id}")
        return

    # Load mask
    polygon = load_mask_polygon(masks_path, image_id, mask_id)
    if not polygon:
        logger.error(f"Mask not found for {image_id}, mask {mask_id}")
        return

    binary_mask = polygon_to_binary_mask(polygon, image_shape)
    skeleton = skeletonize(binary_mask > 0)

    # Calculate measurements
    length_mm = calculate_skeleton_length(skeleton, scale_map)

    # Simple width calculation (median distance transform)
    from scipy.ndimage import distance_transform_edt
    dist_transform = distance_transform_edt(binary_mask)
    skeleton_coords = np.where(skeleton > 0)
    if len(skeleton_coords[0]) > 0:
        widths = []
        for r, c in zip(skeleton_coords[0], skeleton_coords[1]):
            D = get_scale_with_fallback(scale_map, r, c)
            if D > 0:
                widths.append(dist_transform[r, c] * 2 * D)
        avg_width_mm = np.median(widths) if widths else 0
    else:
        avg_width_mm = 0

    panel_a = generate_panel_a_crack(rgb_img, binary_mask, skeleton, scale_map, length_mm, avg_width_mm)

    # Generate Panel (b): Non-crack (if available)
    if noncrack_cluster:
        logger.info("Generating panel (b): Non-crack measurement...")
        noncrack_source = noncrack_cluster['source_masks'][0]
        nc_image_id = noncrack_source['image_id']
        nc_mask_id = noncrack_source['mask_id']

        # Load RGB
        nc_rgb_img = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = rgb_path / f"{nc_image_id}{ext}"
            if candidate.exists():
                nc_rgb_img = cv2.imread(str(candidate))
                break

        if nc_rgb_img is None:
            logger.warning(f"RGB image not found for {nc_image_id}, using crack image for panel (b)")
            nc_rgb_img = rgb_img
            nc_image_id = image_id
            nc_mask_id = mask_id

        # Load scale map
        nc_timestamp_key = nc_image_id
        for prefix in ['camera_RGB_', 'camera_DPT_']:
            if nc_image_id.startswith(prefix):
                nc_timestamp_key = nc_image_id[len(prefix):]
                break

        nc_scale_map = None
        for pattern in [f"scale_map_iso_camera_DPT_{nc_timestamp_key}.npy",
                       f"scale_map_iso_{nc_image_id}.npy"]:
            nc_scale_map_path = scale_maps_path / pattern
            if nc_scale_map_path.exists():
                nc_scale_map = np.load(nc_scale_map_path)
                break

        if nc_scale_map is None:
            logger.warning(f"Scale map not found for {nc_image_id}, using crack scale map")
            nc_scale_map = scale_map

        # Load mask
        nc_polygon = load_mask_polygon(masks_path, nc_image_id, nc_mask_id)
        if nc_polygon:
            nc_binary_mask = polygon_to_binary_mask(nc_polygon, image_shape)

            # Calculate measurements
            area_mm2 = calculate_mask_area_with_scale(nc_binary_mask, nc_scale_map)
            length_mm_nc, width_mm_nc, _, _ = calculate_2d_obb_with_scale(nc_binary_mask, nc_scale_map)

            panel_b = generate_panel_b_noncrack(nc_rgb_img, nc_binary_mask, nc_scale_map,
                                                area_mm2, length_mm_nc, width_mm_nc)
        else:
            logger.warning("Non-crack mask not found, skipping panel (b)")
            panel_b = None
    else:
        panel_b = None

    # Generate Panel (c): Scale map comparison
    logger.info("Generating panel (c): Scale map comparison...")
    panel_c = generate_panel_c_scalemap(scale_map, rgb_img)

    # Combine panels into final figure
    logger.info("Combining panels into final figure...")
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 1, figure=fig, hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(panel_a)
    ax1.axis('off')

    if panel_b is not None:
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(panel_b)
        ax2.axis('off')

    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(panel_c)
    ax3.axis('off')

    plt.tight_layout()

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✅ Saved Figure 4: {output_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    import argparse
    from src.utils import setup_logging

    parser = argparse.ArgumentParser(
        description='Generate Figure 4 for paper: Scale-calibrated measurement comparison'
    )
    parser.add_argument('--clusters', required=True,
                       help='Input crack_clusters.json')
    parser.add_argument('--crack-points', required=True,
                       help='Input crack_points.json')
    parser.add_argument('--masks-dir', required=True,
                       help='YOLO masks directory')
    parser.add_argument('--scale-maps-dir', required=True,
                       help='Scale maps directory')
    parser.add_argument('--rgb-dir', required=True,
                       help='RGB images directory')
    parser.add_argument('--output', required=True,
                       help='Output figure path (e.g., figures/figure4.png)')
    parser.add_argument('--crack-cluster-id', type=int, default=None,
                       help='Specific crack cluster ID (auto-select largest if not specified)')
    parser.add_argument('--noncrack-cluster-id', type=int, default=None,
                       help='Specific non-crack cluster ID (auto-select largest if not specified)')
    parser.add_argument('--image-width', type=int, default=3840)
    parser.add_argument('--image-height', type=int, default=2160)
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        generate_figure4(
            args.clusters,
            args.crack_points,
            args.masks_dir,
            args.scale_maps_dir,
            args.rgb_dir,
            args.output,
            args.crack_cluster_id,
            args.noncrack_cluster_id,
            (args.image_height, args.image_width)
        )

        print(f"\n✅ Figure 4 generated successfully!")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Failed to generate Figure 4: {e}", exc_info=True)
        import sys
        sys.exit(1)
