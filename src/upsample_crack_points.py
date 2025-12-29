#!/usr/bin/env python3
"""
Upsample sparse crack points using KNN-based linear interpolation.

This phase increases point density for better DBSCAN clustering results.
Uses Two-Tier Point System:
- Original points: Have full source information (image_id, mask_id, uv)
- Synthetic points: Interpolated, marked with is_synthetic=true and parent_ids

Inputs:
- crack_points.json (from point_cloud_overlay)

Output:
- crack_points_upsampled.json (denser point cloud)
"""

import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def setup_logging(level: str = 'INFO'):
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_crack_points(json_path: str) -> Dict:
    """Load crack points from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def upsample_knn_interpolation(
    points: List[Dict],
    k_neighbors: int = 5,
    n_interpolations: int = 2,
    max_distance: float = 0.1
) -> List[Dict]:
    """
    Upsample points using KNN-based linear interpolation.

    For each original point, find K nearest neighbors and create
    interpolated points along the edges to those neighbors.

    Args:
        points: Original points with xyz, color, sources
        k_neighbors: Number of nearest neighbors to consider
        n_interpolations: Number of interpolated points per edge
        max_distance: Maximum distance (meters) for interpolation

    Returns:
        List of all points (original + synthetic)
    """
    if len(points) < 2:
        return points

    # Extract coordinates
    coords = np.array([p['xyz'] for p in points])
    n_original = len(points)

    # Find nearest neighbors
    k = min(k_neighbors + 1, n_original)  # +1 because point itself is included
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # Mark original points
    result_points = []
    for p in points:
        p_copy = p.copy()
        p_copy['is_synthetic'] = False
        result_points.append(p_copy)

    # Track which edges we've already interpolated to avoid duplicates
    interpolated_edges = set()
    synthetic_id = n_original

    for i in range(n_original):
        point_i = points[i]
        coord_i = coords[i]

        # Iterate through neighbors (skip index 0 which is the point itself)
        for j_idx in range(1, k):
            j = indices[i, j_idx]
            dist = distances[i, j_idx]

            # Skip if too far
            if dist > max_distance:
                continue

            # Create canonical edge key to avoid duplicates
            edge_key = tuple(sorted([i, j]))
            if edge_key in interpolated_edges:
                continue
            interpolated_edges.add(edge_key)

            point_j = points[j]
            coord_j = coords[j]

            # Interpolate along the edge
            for t in range(1, n_interpolations + 1):
                alpha = t / (n_interpolations + 1)

                # Linear interpolation of coordinates
                interp_coord = coord_i * (1 - alpha) + coord_j * alpha

                # Create synthetic point
                synthetic_point = {
                    'point_id': synthetic_id,
                    'xyz': interp_coord.tolist(),
                    'is_synthetic': True,
                    'parent_ids': [point_i['point_id'], point_j['point_id']]
                }

                # Interpolate color if available
                if 'color' in point_i and 'color' in point_j:
                    color_i = np.array(point_i['color'])
                    color_j = np.array(point_j['color'])
                    interp_color = (color_i * (1 - alpha) + color_j * alpha).astype(int).tolist()
                    synthetic_point['color'] = interp_color

                result_points.append(synthetic_point)
                synthetic_id += 1

    return result_points


def upsample_with_density_control(
    points: List[Dict],
    target_density: float = None,
    min_spacing: float = 0.005,
    k_neighbors: int = 5,
    max_distance: float = 0.1
) -> List[Dict]:
    """
    Upsample with density control based on local point spacing.

    Adaptively determines number of interpolations based on edge length
    to achieve more uniform density.

    Args:
        points: Original points
        target_density: Target points per meter (if None, uses 2x original)
        min_spacing: Minimum spacing between points (meters)
        k_neighbors: Number of nearest neighbors
        max_distance: Maximum interpolation distance

    Returns:
        Upsampled points
    """
    if len(points) < 2:
        return points

    coords = np.array([p['xyz'] for p in points])
    n_original = len(points)

    # Estimate current density
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    avg_spacing = np.mean(distances[:, 1])

    if target_density is None:
        target_spacing = avg_spacing / 2  # Double density
    else:
        target_spacing = 1.0 / target_density

    target_spacing = max(target_spacing, min_spacing)

    logger.info(f"Current avg spacing: {avg_spacing*1000:.2f}mm")
    logger.info(f"Target spacing: {target_spacing*1000:.2f}mm")

    # Find neighbors for interpolation
    k = min(k_neighbors + 1, n_original)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # Mark original points
    result_points = []
    for p in points:
        p_copy = p.copy()
        p_copy['is_synthetic'] = False
        result_points.append(p_copy)

    interpolated_edges = set()
    synthetic_id = n_original

    for i in range(n_original):
        point_i = points[i]
        coord_i = coords[i]

        for j_idx in range(1, k):
            j = indices[i, j_idx]
            dist = distances[i, j_idx]

            if dist > max_distance:
                continue

            edge_key = tuple(sorted([i, j]))
            if edge_key in interpolated_edges:
                continue
            interpolated_edges.add(edge_key)

            point_j = points[j]
            coord_j = coords[j]

            # Determine number of interpolations based on edge length
            n_interp = max(0, int(dist / target_spacing) - 1)

            if n_interp == 0:
                continue

            for t in range(1, n_interp + 1):
                alpha = t / (n_interp + 1)

                interp_coord = coord_i * (1 - alpha) + coord_j * alpha

                synthetic_point = {
                    'point_id': synthetic_id,
                    'xyz': interp_coord.tolist(),
                    'is_synthetic': True,
                    'parent_ids': [point_i['point_id'], point_j['point_id']]
                }

                # Interpolate color if available
                if 'color' in point_i and 'color' in point_j:
                    color_i = np.array(point_i['color'])
                    color_j = np.array(point_j['color'])
                    interp_color = (color_i * (1 - alpha) + color_j * alpha).astype(int).tolist()
                    synthetic_point['color'] = interp_color

                result_points.append(synthetic_point)
                synthetic_id += 1

    return result_points


def run_upsampling(
    input_json: str,
    output_json: str,
    method: str = 'density',
    k_neighbors: int = 5,
    n_interpolations: int = 2,
    max_distance: float = 0.1,
    min_spacing: float = 0.005
):
    """
    Run the upsampling pipeline.

    Args:
        input_json: Input crack_points.json path
        output_json: Output upsampled JSON path
        method: 'fixed' or 'density'
        k_neighbors: Number of neighbors for interpolation
        n_interpolations: Fixed number of interpolations (for 'fixed' method)
        max_distance: Maximum edge distance for interpolation (meters)
        min_spacing: Minimum point spacing (meters)
    """
    logger.info("=" * 80)
    logger.info("Crack Points Upsampling")
    logger.info("=" * 80)

    # Load input
    logger.info(f"Loading: {input_json}")
    data = load_crack_points(input_json)
    all_points = data['points']

    # Filter to only crack class for upsampling
    crack_points = [p for p in all_points if p.get('class') == 'crack']
    other_points = [p for p in all_points if p.get('class') != 'crack']

    n_crack = len(crack_points)
    n_other = len(other_points)
    n_original = len(all_points)

    logger.info(f"Original points: {n_original}")
    logger.info(f"  Crack points (to upsample): {n_crack}")
    logger.info(f"  Other defect points (no upsampling): {n_other}")

    # Upsample crack points only
    if n_crack > 0:
        if method == 'fixed':
            logger.info(f"Method: Fixed interpolation (k={k_neighbors}, n={n_interpolations})")
            upsampled_crack = upsample_knn_interpolation(
                crack_points, k_neighbors, n_interpolations, max_distance
            )
        else:
            logger.info(f"Method: Density-controlled (k={k_neighbors}, min_spacing={min_spacing*1000:.1f}mm)")
            upsampled_crack = upsample_with_density_control(
                crack_points, None, min_spacing, k_neighbors, max_distance
            )

        n_synthetic = len(upsampled_crack) - n_crack
        logger.info(f"Synthetic crack points added: {n_synthetic}")
        logger.info(f"Crack density increase: {len(upsampled_crack)/n_crack:.2f}x")
    else:
        upsampled_crack = []
        n_synthetic = 0
        logger.info("No crack points to upsample")

    # Combine upsampled crack points with original other defect points
    upsampled = upsampled_crack + other_points
    n_total = len(upsampled)

    logger.info(f"Total points after upsampling: {n_total}")
    logger.info(f"  Crack points (original + synthetic): {len(upsampled_crack)}")
    logger.info(f"  Other defect points (original only): {n_other}")

    # Prepare output
    output_data = {
        'metadata': {
            'original_points': n_original,
            'synthetic_points': n_synthetic,
            'total_points': n_total,
            'upsampling_method': method,
            'parameters': {
                'k_neighbors': k_neighbors,
                'max_distance': max_distance,
                'min_spacing': min_spacing if method == 'density' else None,
                'n_interpolations': n_interpolations if method == 'fixed' else None
            }
        },
        'points': upsampled
    }

    # Save output
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved to: {output_json}")

    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Upsample sparse crack points for better DBSCAN clustering'
    )

    parser.add_argument('--input', required=True,
                       help='Input crack_points.json')
    parser.add_argument('--output', required=True,
                       help='Output upsampled JSON')
    parser.add_argument('--method', default='density',
                       choices=['fixed', 'density'],
                       help='Upsampling method: fixed or density-controlled')
    parser.add_argument('--k-neighbors', type=int, default=5,
                       help='Number of nearest neighbors for interpolation')
    parser.add_argument('--n-interpolations', type=int, default=2,
                       help='Number of interpolations per edge (fixed method)')
    parser.add_argument('--max-distance', type=float, default=0.1,
                       help='Maximum edge distance for interpolation (meters)')
    parser.add_argument('--min-spacing', type=float, default=0.005,
                       help='Minimum point spacing (meters, density method)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        run_upsampling(
            args.input,
            args.output,
            args.method,
            args.k_neighbors,
            args.n_interpolations,
            args.max_distance,
            args.min_spacing
        )
        print(f"\n✅ Upsampling complete!")
        print(f"   Output: {args.output}")
    except Exception as e:
        logger.error(f"Upsampling failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
