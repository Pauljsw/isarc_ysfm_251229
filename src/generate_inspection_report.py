#!/usr/bin/env python3
"""
Generate 2D inspection diagram and measurement table from SfM point cloud and crack clusters.

This module creates:
1. 2D inspection diagram (PNG) showing wall outline with cracks
2. Measurement table (CSV) with crack dimensions
"""

import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ply_with_colors(ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load point cloud with colors from PLY file.

    Returns:
        (points, colors): Nx3 XYZ array, Nx3 RGB array (0-255)
    """
    import struct

    # Property type sizes in bytes
    type_sizes = {
        'char': 1, 'uchar': 1, 'int8': 1, 'uint8': 1,
        'short': 2, 'ushort': 2, 'int16': 2, 'uint16': 2,
        'int': 4, 'uint': 4, 'int32': 4, 'uint32': 4,
        'float': 4, 'float32': 4,
        'double': 8, 'float64': 8
    }

    with open(ply_path, 'rb') as f:
        # Read header
        vertex_count = 0
        is_binary = False
        is_little_endian = True
        vertex_properties = []
        in_vertex_element = False

        while True:
            line = f.readline().decode('ascii').strip()

            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
                in_vertex_element = True
            elif line.startswith('element') and in_vertex_element:
                in_vertex_element = False
            elif line.startswith('property') and in_vertex_element:
                parts = line.split()
                if len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    vertex_properties.append((prop_name, prop_type))
            elif line.startswith('format'):
                if 'binary_little_endian' in line:
                    is_binary = True
                    is_little_endian = True
                elif 'binary_big_endian' in line:
                    is_binary = True
                    is_little_endian = False
            elif line == 'end_header':
                break

        # Find property indices
        prop_names = [p[0] for p in vertex_properties]
        has_color = 'red' in prop_names and 'green' in prop_names and 'blue' in prop_names

        # Calculate offsets for each property
        prop_offsets = {}
        offset = 0
        for name, ptype in vertex_properties:
            prop_offsets[name] = offset
            offset += type_sizes.get(ptype, 4)
        vertex_size = offset if offset > 0 else 12

        points = []
        colors = []

        if is_binary:
            endian = '<' if is_little_endian else '>'

            for _ in range(vertex_count):
                data = f.read(vertex_size)
                if len(data) < vertex_size:
                    break

                # Extract XYZ
                x, y, z = struct.unpack(f'{endian}fff', data[:12])
                points.append([x, y, z])

                # Extract RGB if available
                if has_color:
                    r = data[prop_offsets['red']]
                    g = data[prop_offsets['green']]
                    b = data[prop_offsets['blue']]
                    colors.append([r, g, b])
                else:
                    colors.append([128, 128, 128])
        else:
            # ASCII format
            for _ in range(vertex_count):
                line = f.readline().decode('ascii').strip()
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])

                        if has_color and len(parts) >= 6:
                            r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                            colors.append([r, g, b])
                        else:
                            colors.append([128, 128, 128])
                    except ValueError:
                        continue

        points = np.array(points)
        colors = np.array(colors)

        # Filter NaN and Inf values
        if len(points) > 0:
            valid_mask = np.all(np.isfinite(points), axis=1)
            n_invalid = np.sum(~valid_mask)
            if n_invalid > 0:
                logger.warning(f"Removed {n_invalid} points with NaN/Inf values")
            points = points[valid_mask]
            colors = colors[valid_mask]

        logger.info(f"Loaded {len(points)} points from {ply_path}")
        return points, colors


def group_points_by_color(points: np.ndarray, colors: np.ndarray) -> List[Tuple[Tuple[int, int, int], np.ndarray]]:
    """
    Group points by their RGB color.

    Returns:
        List of (color_tuple, points_array) sorted by x-coordinate of centroid
    """
    # Group by color
    color_groups = {}
    for i, (point, color) in enumerate(zip(points, colors)):
        color_key = tuple(color)
        if color_key not in color_groups:
            color_groups[color_key] = []
        color_groups[color_key].append(point)

    # Convert to arrays and sort by centroid x-coordinate
    result = []
    for color, pts in color_groups.items():
        pts_array = np.array(pts)
        result.append((color, pts_array))

    # Sort by centroid x-coordinate for consistent ordering
    result.sort(key=lambda x: x[1][:, 0].mean())

    logger.info(f"Found {len(result)} clusters by color")
    return result


def filter_noise_statistical(points: np.ndarray, k: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """
    Statistical outlier removal based on mean distance to k nearest neighbors.

    Points with mean distance > mean + std_ratio * std are removed.
    """
    if len(points) < k + 1:
        return points

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)
    distances, _ = nbrs.kneighbors(points)

    # Mean distance to k neighbors (exclude self)
    mean_distances = distances[:, 1:].mean(axis=1)

    # Filter outliers
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    threshold = global_mean + std_ratio * global_std

    mask = mean_distances <= threshold
    filtered = points[mask]

    logger.info(f"Statistical filtering: {len(points)} -> {len(filtered)} points "
                f"(removed {len(points) - len(filtered)})")
    return filtered


def filter_noise_density(points: np.ndarray, axis: int = 2,
                         grid_size: float = 0.1, min_count: int = 3) -> np.ndarray:
    """
    Density-based filtering along specified axis.

    Projects points to 2D (excluding specified axis) and filters by cell density.
    """
    # Project to 2D
    axes = [i for i in range(3) if i != axis]
    projected = points[:, axes]

    # Create grid
    min_vals = projected.min(axis=0)
    max_vals = projected.max(axis=0)

    # Assign points to grid cells
    cell_indices = ((projected - min_vals) / grid_size).astype(int)

    # Count points per cell
    cell_counts = {}
    for i, cell in enumerate(cell_indices):
        key = tuple(cell)
        if key not in cell_counts:
            cell_counts[key] = []
        cell_counts[key].append(i)

    # Keep points in cells with enough neighbors
    keep_indices = []
    for cell, indices in cell_counts.items():
        if len(indices) >= min_count:
            keep_indices.extend(indices)

    filtered = points[keep_indices]
    logger.info(f"Density filtering: {len(points)} -> {len(filtered)} points")
    return filtered


def fit_plane_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit plane to points using PCA.

    Returns:
        (centroid, normal, principal_axes)
        - centroid: center of the plane
        - normal: plane normal vector
        - principal_axes: 2x3 array of the two principal axes in the plane
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    pca = PCA(n_components=3)
    pca.fit(centered)

    # The smallest eigenvalue component is the normal
    normal = pca.components_[2]  # Smallest variance direction
    principal_axes = pca.components_[:2]  # Two largest variance directions

    logger.info(f"Plane fitted - Normal: {normal}")
    return centroid, normal, principal_axes


def project_to_plane(points: np.ndarray, centroid: np.ndarray,
                     principal_axes: np.ndarray) -> np.ndarray:
    """
    Project 3D points onto 2D plane defined by principal axes.

    Returns:
        Nx2 array of 2D coordinates
    """
    centered = points - centroid
    projected_2d = np.dot(centered, principal_axes.T)
    return projected_2d


def extract_boundary_alpha_shape(points_2d: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Extract boundary using alpha shape (concave hull).

    Simplified implementation using grid-based boundary detection.
    """
    if len(points_2d) < 3:
        return points_2d

    # Use convex hull as fallback, with shrinking based on density
    try:
        hull = ConvexHull(points_2d)
        boundary = points_2d[hull.vertices]
        return boundary
    except Exception as e:
        logger.warning(f"Convex hull failed: {e}")
        return points_2d


def extract_boundary_grid(points_2d: np.ndarray, grid_size: float = 0.05) -> np.ndarray:
    """
    Extract boundary using grid-based edge detection.

    Returns boundary points where grid cells touch empty space.
    """
    if len(points_2d) < 3:
        return points_2d

    min_vals = points_2d.min(axis=0)
    max_vals = points_2d.max(axis=0)

    # Create occupancy grid
    grid_shape = ((max_vals - min_vals) / grid_size + 1).astype(int)
    grid = np.zeros(grid_shape, dtype=bool)

    # Mark occupied cells
    cell_indices = ((points_2d - min_vals) / grid_size).astype(int)
    cell_indices = np.clip(cell_indices, 0, grid_shape - 1)

    for ci in cell_indices:
        grid[ci[0], ci[1]] = True

    # Find boundary cells (occupied cells with at least one empty neighbor)
    boundary_cells = []
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if not grid[i, j]:
                continue

            # Check 8-neighbors
            is_boundary = False
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= grid_shape[0] or nj < 0 or nj >= grid_shape[1]:
                        is_boundary = True
                        break
                    if not grid[ni, nj]:
                        is_boundary = True
                        break
                if is_boundary:
                    break

            if is_boundary:
                boundary_cells.append([i, j])

    if not boundary_cells:
        return points_2d

    # Convert back to coordinates
    boundary_coords = np.array(boundary_cells) * grid_size + min_vals + grid_size / 2

    # Order boundary points (approximate)
    if len(boundary_coords) > 3:
        try:
            hull = ConvexHull(boundary_coords)
            boundary_coords = boundary_coords[hull.vertices]
        except:
            pass

    return boundary_coords


def load_clusters(clusters_path: Path) -> List[Dict]:
    """Load crack clusters from JSON."""
    with open(clusters_path, 'r') as f:
        data = json.load(f)
    return data.get('clusters', data) if isinstance(data, dict) else data


def load_crack_points(crack_points_path: Path) -> Dict[int, Dict]:
    """Load crack points and create lookup by point_id."""
    with open(crack_points_path, 'r') as f:
        data = json.load(f)

    # Handle different JSON structures
    points = None
    if isinstance(data, dict):
        # Try different keys
        for key in ['points', 'crack_points']:
            if key in data and isinstance(data[key], list):
                points = data[key]
                logger.debug(f"Found points under '{key}' key")
                break

        if points is None:
            logger.error(f"crack_points.json keys: {list(data.keys())}")
            return {}
    elif isinstance(data, list):
        points = data
    else:
        logger.error(f"Unexpected crack_points format: {type(data)}")
        return {}

    # Build lookup
    lookup = {}
    for p in points:
        if isinstance(p, dict) and 'point_id' in p:
            lookup[p['point_id']] = p

    logger.info(f"Loaded {len(lookup)} crack points")
    return lookup


def load_measurements(measurements_path: Path) -> Dict[int, Dict]:
    """Load measurements and create lookup by cluster_id."""
    with open(measurements_path, 'r') as f:
        data = json.load(f)

    measurements = data.get('measurements', data) if isinstance(data, dict) else data
    return {m['cluster_id']: m for m in measurements}


def get_cluster_polyline(cluster: Dict, crack_points_lookup: Dict,
                         drop_axis: int = 2) -> Optional[np.ndarray]:
    """
    Get 2D polyline representation of a cluster.

    Projects cluster points to 2D and orders them along principal axis.

    Args:
        cluster: Cluster dict with point_ids
        crack_points_lookup: point_id -> point dict
        drop_axis: Axis to drop (0=X, 1=Y, 2=Z). Default Z for front view.

    Returns:
        Nx2 array of ordered 2D points for polyline
    """
    point_ids = cluster.get('point_ids', [])

    # Get 3D coordinates
    xyz_list = []
    for pid in point_ids:
        if pid in crack_points_lookup:
            point = crack_points_lookup[pid]
            if 'xyz' in point:
                xyz_list.append(point['xyz'])

    if len(xyz_list) < 2:
        return None

    xyz = np.array(xyz_list)

    # Simple 2D projection by dropping one axis
    axes_to_keep = [i for i in range(3) if i != drop_axis]
    points_2d = xyz[:, axes_to_keep]

    # Order points along principal axis for smooth polyline
    if len(points_2d) >= 2:
        pca = PCA(n_components=1)
        pca.fit(points_2d)

        center = points_2d.mean(axis=0)
        direction = pca.components_[0]

        # Project to principal axis and sort
        projections = np.dot(points_2d - center, direction)
        sorted_indices = np.argsort(projections)

        return points_2d[sorted_indices]

    return points_2d


def generate_inspection_diagram(
    crack_polylines: List[Tuple[int, np.ndarray]],
    output_path: Path,
    title: str = "Wall Inspection Diagram",
    figsize: Tuple[int, int] = (14, 10),
    margin_ratio: float = 0.1
):
    """
    Generate inspection diagram image with clean rectangular boundary.

    Args:
        crack_polylines: List of (cluster_id, polyline_points) tuples
        output_path: Output image path
        title: Plot title
        figsize: Figure size
        margin_ratio: Margin around cracks as ratio of extent
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')

    # Collect all crack points to determine bounding box
    all_points = []
    for item in crack_polylines:
        polyline = item[1]  # Second element is always polyline
        if polyline is not None and len(polyline) >= 2:
            all_points.extend(polyline.tolist())

    if not all_points:
        logger.warning("No crack points to draw")
        plt.close()
        return

    all_points = np.array(all_points)

    # Calculate bounding box with margin
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)

    x_range = x_max - x_min
    y_range = y_max - y_min

    margin_x = x_range * margin_ratio
    margin_y = y_range * margin_ratio

    # Wall boundary as clean rectangle
    wall_rect = np.array([
        [x_min - margin_x, y_min - margin_y],
        [x_max + margin_x, y_min - margin_y],
        [x_max + margin_x, y_max + margin_y],
        [x_min - margin_x, y_max + margin_y],
        [x_min - margin_x, y_min - margin_y]  # Close rectangle
    ])

    # Draw wall boundary (clean rectangle)
    ax.plot(wall_rect[:, 0], wall_rect[:, 1],
            'k-', linewidth=3)
    ax.fill(wall_rect[:-1, 0], wall_rect[:-1, 1],
            alpha=0.05, color='lightgray')

    # Draw cracks as polylines
    for item in crack_polylines:
        # Handle both (id, polyline) and (id, polyline, color) formats
        if len(item) == 3:
            cluster_id, polyline, color = item
        else:
            cluster_id, polyline = item
            color = 'black'  # Default color

        # Force all clusters to black
        color = 'black'

        if polyline is not None and len(polyline) >= 2:
            # Draw polyline (smooth crack line)
            ax.plot(polyline[:, 0], polyline[:, 1],
                    linewidth=2.5, color=color, solid_capstyle='round')

            # Add label near middle of crack
            label_pos = polyline[len(polyline)//2]
            ax.annotate(f'{cluster_id + 1}',
                       xy=label_pos,
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       ha='center', va='bottom',
                       color=color)

    ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved inspection diagram: {output_path}")


def generate_measurement_table(
    measurements: Dict[int, Dict],
    output_path: Path
):
    """
    Generate measurement table CSV.

    Columns: No., Type, 길이(mm), 폭(mm), 너비(mm), 면적(mm²)
    - Crack: 길이 + 폭 (선형 결함)
    - Others: 길이 + 너비 + 면적 (면적 결함)
    """
    rows = []

    for cluster_id in sorted(measurements.keys()):
        m = measurements[cluster_id]
        defect_class = m.get('class', 'crack')

        # Format defect type name
        type_name = defect_class.replace('_', ' ').title()

        # Get measurements based on defect type
        if defect_class == 'crack':
            # Crack: 길이 + 폭
            length = round(m.get('total_length_mm', 0), 1)
            width_crack = round(m.get('avg_width_mm', 0), 2)  # 폭
            width_defect = '-'  # 너비 (사용 안 함)
            area = '-'
        else:
            # 8종 결함: 길이 + 너비 + 면적
            length = round(m.get('length_mm', 0), 1)
            width_crack = '-'  # 폭 (사용 안 함)
            width_defect = round(m.get('width_mm', 0), 1)  # 너비
            area = round(m.get('area_mm2', 0), 1)

        rows.append({
            'No.': cluster_id + 1,
            'Type': type_name,
            'Length (mm)': length,
            'Width-Crack (mm)': width_crack,  # 폭
            'Width-Defect (mm)': width_defect,  # 너비
            'Area (mm²)': area
        })

    # Write CSV
    fieldnames = ['No.', 'Type', 'Length (mm)', 'Width-Crack (mm)', 'Width-Defect (mm)', 'Area (mm²)']
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved measurement table: {output_path}")

    # Also print to console
    print("\n" + "="*95)
    print("Measurement Table")
    print("="*95)
    print(f"{'No.':<6} {'Type':<14} {'Length(mm)':<13} {'Crack W(mm)':<14} {'Width(mm)':<12} {'Area(mm²)':<13}")
    print("-"*95)
    for row in rows:
        print(f"{row['No.']:<6} {row['Type']:<14} {row['Length (mm)']:<13} "
              f"{row['Width-Crack (mm)']:<14} {row['Width-Defect (mm)']:<12} {row['Area (mm²)']:<13}")
    print("="*95 + "\n")


def generate_combined_report(
    crack_polylines: List[Tuple[int, np.ndarray]],
    measurements: Dict[int, Dict],
    output_path: Path,
    title: str = "Wall Inspection Report",
    figsize: Tuple[int, int] = (20, 12)
):
    """
    Generate combined report with diagram on left and table on right.
    """
    fig = plt.figure(figsize=figsize, facecolor='white')

    # Left: Diagram (60% width)
    ax_diagram = fig.add_axes([0.05, 0.1, 0.5, 0.8])

    # Collect all points for bounding box
    all_points = []
    for item in crack_polylines:
        polyline = item[1]
        if polyline is not None and len(polyline) >= 2:
            all_points.extend(polyline.tolist())

    if not all_points:
        logger.warning("No crack points to draw")
        plt.close()
        return

    all_points = np.array(all_points)

    # Calculate bounding box
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    margin_x = x_range * 0.1
    margin_y = y_range * 0.1

    # Wall boundary
    wall_rect = np.array([
        [x_min - margin_x, y_min - margin_y],
        [x_max + margin_x, y_min - margin_y],
        [x_max + margin_x, y_max + margin_y],
        [x_min - margin_x, y_max + margin_y],
        [x_min - margin_x, y_min - margin_y]
    ])

    ax_diagram.plot(wall_rect[:, 0], wall_rect[:, 1], 'k-', linewidth=2)
    ax_diagram.fill(wall_rect[:-1, 0], wall_rect[:-1, 1], alpha=0.05, color='lightgray')

    # Draw cracks
    for item in crack_polylines:
        if len(item) == 3:
            cluster_id, polyline, color = item
        else:
            cluster_id, polyline = item
            color = 'black'

        # Force all clusters to black
        color = 'black'

        if polyline is not None and len(polyline) >= 2:
            ax_diagram.plot(polyline[:, 0], polyline[:, 1],
                          linewidth=2, color=color, solid_capstyle='round')
            label_pos = polyline[len(polyline)//2]
            ax_diagram.annotate(f'{cluster_id + 1}', xy=label_pos,
                              xytext=(3, 3), textcoords='offset points',
                              fontsize=8, fontweight='bold', color=color)

    ax_diagram.set_aspect('equal')
    ax_diagram.set_title('Defect Location Diagram', fontsize=12, fontweight='bold')
    ax_diagram.set_xticklabels([])
    ax_diagram.set_yticklabels([])

    # Right: Table (35% width)
    ax_table = fig.add_axes([0.58, 0.1, 0.38, 0.8])
    ax_table.axis('off')

    # Prepare table data with 6 columns
    table_data = [['No.', 'Type', 'Length\n(mm)', 'Crack W\n(mm)', 'Width\n(mm)', 'Area\n(mm²)']]
    for cluster_id in sorted(measurements.keys()):
        m = measurements[cluster_id]
        defect_class = m.get('class', 'crack')
        type_name = defect_class.replace('_', ' ').title()

        # Get measurements based on defect type
        if defect_class == 'crack':
            # Crack: Length + Crack Width
            length = f"{m.get('total_length_mm', 0):.1f}"
            width_crack = f"{m.get('avg_width_mm', 0):.2f}"
            width_defect = '-'
            area = '-'
        else:
            # 8 defect types: Length + Width + Area
            length = f"{m.get('length_mm', 0):.1f}"
            width_crack = '-'
            width_defect = f"{m.get('width_mm', 0):.1f}"
            area = f"{m.get('area_mm2', 0):.1f}"

        table_data.append([
            f'{cluster_id + 1}',
            type_name,
            length,
            width_crack,
            width_defect,
            area
        ])

    # Create table
    table = ax_table.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.10, 0.20, 0.15, 0.13, 0.13, 0.15]
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    # Header styling
    for j in range(6):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Alternating row colors
    for i in range(1, len(table_data)):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8E8E8')

    ax_table.set_title('Measurement Results', fontsize=12, fontweight='bold', pad=20)

    # Main title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved combined report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate 2D inspection diagram and measurement table'
    )
    parser.add_argument('--ply', type=str, required=True,
                        help='Path to clustered_cracks.ply')
    parser.add_argument('--crack-points', type=str, required=True,
                        help='Path to crack_points.json (to filter synthetic points)')
    parser.add_argument('--measurements', type=str, required=True,
                        help='Path to cluster measurements JSON')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--diagram-name', type=str, default='inspection_diagram.png',
                        help='Output diagram filename')
    parser.add_argument('--table-name', type=str, default='measurement_table.csv',
                        help='Output table filename')
    parser.add_argument('--report-name', type=str, default='inspection_report.png',
                        help='Output combined report filename (diagram + table)')
    parser.add_argument('--drop-axis', type=int, default=2, choices=[0, 1, 2],
                        help='Axis to drop for 2D projection (0=X, 1=Y, 2=Z). Default: 2 (front view)')
    parser.add_argument('--flip-x', action='store_true',
                        help='Flip X axis (horizontal mirror)')
    parser.add_argument('--flip-y', action='store_true',
                        help='Flip Y axis (vertical mirror)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("="*60)
    logger.info("Inspection Report Generation")
    logger.info("="*60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load clustered cracks PLY
    logger.info(f"Loading clustered cracks: {args.ply}")
    points, colors = load_ply_with_colors(Path(args.ply))

    # Load crack_points.json to identify synthetic points
    logger.info(f"Loading crack points: {args.crack_points}")
    crack_points_lookup = load_crack_points(Path(args.crack_points))

    # Build set of original (non-synthetic) point coordinates
    original_xyz_set = set()
    for pid, point_data in crack_points_lookup.items():
        if not point_data.get('is_synthetic', False):
            xyz = point_data.get('xyz')
            if xyz:
                # Round to avoid float precision issues
                key = (round(xyz[0], 6), round(xyz[1], 6), round(xyz[2], 6))
                original_xyz_set.add(key)

    logger.info(f"Original (non-synthetic) points: {len(original_xyz_set)}")

    # Filter PLY points to only include original points
    filtered_points = []
    filtered_colors = []
    for point, color in zip(points, colors):
        key = (round(point[0], 6), round(point[1], 6), round(point[2], 6))
        if key in original_xyz_set:
            filtered_points.append(point)
            filtered_colors.append(color)

    points = np.array(filtered_points) if filtered_points else np.array([])
    colors = np.array(filtered_colors) if filtered_colors else np.array([])
    logger.info(f"After filtering synthetic: {len(points)} points")

    # Group by color to identify clusters
    cluster_groups = group_points_by_color(points, colors)

    logger.info(f"Loading measurements: {args.measurements}")
    measurements = load_measurements(Path(args.measurements))

    # Generate crack polylines from PLY data
    logger.info(f"Generating crack polylines (dropping axis {args.drop_axis})...")
    axes_to_keep = [i for i in range(3) if i != args.drop_axis]

    crack_polylines = []
    for cluster_id, (color, cluster_points) in enumerate(cluster_groups):
        # Project to 2D
        points_2d = cluster_points[:, axes_to_keep]

        # Order points along principal axis
        if len(points_2d) >= 2:
            pca = PCA(n_components=1)
            pca.fit(points_2d)
            center = points_2d.mean(axis=0)
            direction = pca.components_[0]
            projections = np.dot(points_2d - center, direction)
            sorted_indices = np.argsort(projections)
            polyline = points_2d[sorted_indices]
        else:
            polyline = points_2d

        # Apply flip transformations
        if args.flip_x:
            polyline[:, 0] = -polyline[:, 0]
        if args.flip_y:
            polyline[:, 1] = -polyline[:, 1]

        # Use original color from PLY
        crack_polylines.append((cluster_id, polyline, np.array(color) / 255.0))

    logger.info(f"Generated {len(crack_polylines)} crack polylines")

    # Generate outputs
    diagram_path = output_dir / args.diagram_name
    table_path = output_dir / args.table_name
    report_path = output_dir / args.report_name

    generate_inspection_diagram(crack_polylines, diagram_path)
    generate_measurement_table(measurements, table_path)
    generate_combined_report(crack_polylines, measurements, report_path)

    logger.info("="*60)
    logger.info("Done!")
    logger.info(f"Combined report: {report_path}")
    logger.info("="*60)


if __name__ == '__main__':
    main()