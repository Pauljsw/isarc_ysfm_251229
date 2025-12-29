"""
Visualize 3D Mask Clusters in Global Coordinates

Creates PLY visualizations of clustered masks:
1. Standalone cluster PLY (global coordinates, colored by cluster)
2. Optional: SFM point cloud with cluster regions colored

Usage:
    python -m src.visualize_clusters_3d \
        --clusters outputs/mask_clusters.json \
        --masks-3d outputs/masks_3d.json \
        --output-clusters-ply outputs/clusters_3d.ply \
        --sfm-sparse-dir data/sfm/sparse/0 \
        --output-combined-ply outputs/sfm_with_clusters.ply
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import struct

logger = logging.getLogger(__name__)


def generate_cluster_colors(n_clusters: int, seed: int = 42) -> np.ndarray:
    """
    Generate distinct colors for clusters.

    Args:
        n_clusters: Number of clusters
        seed: Random seed

    Returns:
        colors: (n_clusters, 3) array of RGB colors (0-255)
    """
    np.random.seed(seed)

    # Generate bright, distinct colors
    colors = np.random.randint(50, 255, size=(n_clusters, 3), dtype=np.uint8)

    return colors


def save_ply_binary(filename: str, xyz: np.ndarray, rgb: np.ndarray):
    """
    Save point cloud as binary PLY.

    Args:
        filename: Output PLY path
        xyz: 3D coordinates (N, 3), float32
        rgb: RGB colors (N, 3), uint8
    """
    assert xyz.shape[0] == rgb.shape[0]
    assert xyz.shape[1] == 3
    assert rgb.shape[1] == 3

    # Ensure types
    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.uint8)

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
            f.write(xyz[i].tobytes())
            # rgb as uint8
            f.write(rgb[i].tobytes())

    logger.info(f"Saved PLY: {filename} ({len(xyz):,} points)")


def create_clusters_ply(
    clusters_json: str,
    masks_3d_json: str,
    output_ply: str
):
    """
    Create standalone cluster PLY (global coordinates, colored by cluster).

    Args:
        clusters_json: Clusters JSON
        masks_3d_json: Masks 3D JSON
        output_ply: Output PLY path
    """
    logger.info("=" * 80)
    logger.info("Creating Clusters PLY (Global Coordinates)")
    logger.info("=" * 80)

    # Load clusters
    logger.info(f"Loading clusters: {clusters_json}")
    with open(clusters_json, 'r') as f:
        clusters_data = json.load(f)

    clusters = clusters_data['clusters']
    logger.info(f"  Loaded {len(clusters)} clusters")

    # Load masks 3D
    logger.info(f"Loading 3D masks: {masks_3d_json}")
    with open(masks_3d_json, 'r') as f:
        masks_data = json.load(f)

    # Build mask lookup
    masks_lookup = {}
    for mask in masks_data['masks']:
        key = (mask['image_id'], mask['mask_id'])
        masks_lookup[key] = mask

    logger.info(f"  Loaded {len(masks_lookup)} 3D masks")

    # Generate cluster colors
    cluster_colors = generate_cluster_colors(len(clusters))

    # Collect all points
    all_points = []
    all_colors = []

    for cluster_idx, cluster in enumerate(clusters):
        cluster_id = cluster['cluster_id']
        cluster_color = cluster_colors[cluster_id % len(cluster_colors)]

        logger.debug(f"Cluster {cluster_id}: {cluster['n_masks']} masks, color={cluster_color}")

        # Collect points from all masks in cluster
        for mask_ref in cluster['masks']:
            image_id = mask_ref['image_id']
            mask_id = mask_ref['mask_id']

            key = (image_id, mask_id)

            if key not in masks_lookup:
                logger.warning(f"Mask {key} not found in masks_3d, skipping")
                continue

            mask_3d = masks_lookup[key]
            points_3d = np.array(mask_3d['points_3d'])

            # Add points with cluster color
            all_points.append(points_3d)

            # Same color for all points in this mask
            colors = np.tile(cluster_color, (len(points_3d), 1))
            all_colors.append(colors)

    # Concatenate
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    logger.info(f"Total points: {len(all_points):,}")

    # Save PLY (optional)
    if output_ply is not None:
        save_ply_binary(output_ply, all_points, all_colors)
        logger.info(f"Saved PLY: {output_ply} ({len(all_points):,} points)")
    else:
        logger.debug("output_ply=None, skipping save (returning points only)")

    logger.info("=" * 80)

    return all_points, all_colors


def load_sfm_point_cloud(sparse_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load SFM sparse point cloud from COLMAP binary.

    Args:
        sparse_dir: COLMAP sparse/0 directory

    Returns:
        xyz: (N, 3) coordinates
        rgb: (N, 3) colors
    """
    from .colmap_io import read_points3D_binary

    logger.info(f"Loading SFM point cloud: {sparse_dir}")

    sparse_path = Path(sparse_dir)
    points3D = read_points3D_binary(str(sparse_path / "points3D.bin"))

    logger.info(f"  Loaded {len(points3D)} SFM points")

    # Extract coordinates and colors
    xyz_list = []
    rgb_list = []

    for point_id, point in points3D.items():
        xyz_list.append(point.xyz)
        rgb_list.append(point.rgb)

    xyz = np.array(xyz_list)
    rgb = np.array(rgb_list, dtype=np.uint8)

    return xyz, rgb


def create_combined_ply(
    clusters_json: str,
    masks_3d_json: str,
    sfm_sparse_dir: str,
    output_ply: str,
    sfm_color: Tuple[int, int, int] = (200, 200, 200)
):
    """
    Create combined PLY: SFM point cloud + colored cluster regions.

    Args:
        clusters_json: Clusters JSON
        masks_3d_json: Masks 3D JSON
        sfm_sparse_dir: COLMAP sparse directory
        output_ply: Output PLY path
        sfm_color: RGB color for SFM points (default: gray)
    """
    logger.info("=" * 80)
    logger.info("Creating Combined PLY (SFM + Clusters)")
    logger.info("=" * 80)

    # Load SFM point cloud
    sfm_xyz, sfm_rgb = load_sfm_point_cloud(sfm_sparse_dir)

    # Override SFM colors to gray for contrast
    sfm_rgb_gray = np.tile(sfm_color, (len(sfm_xyz), 1)).astype(np.uint8)

    # Load cluster points (colored)
    cluster_xyz, cluster_rgb = create_clusters_ply(
        clusters_json,
        masks_3d_json,
        output_ply=None  # Don't save yet
    )

    # Combine
    combined_xyz = np.vstack([sfm_xyz, cluster_xyz])
    combined_rgb = np.vstack([sfm_rgb_gray, cluster_rgb])

    logger.info(f"Combined points: {len(combined_xyz):,} "
               f"(SFM: {len(sfm_xyz):,}, Clusters: {len(cluster_xyz):,})")

    # Save
    save_ply_binary(output_ply, combined_xyz, combined_rgb)

    logger.info("=" * 80)

    return combined_xyz, combined_rgb


def run_visualization(
    clusters_json: str,
    masks_3d_json: str,
    output_clusters_ply: str,
    sfm_sparse_dir: Optional[str] = None,
    output_combined_ply: Optional[str] = None
):
    """
    Run cluster visualization.

    Args:
        clusters_json: Clusters JSON
        masks_3d_json: Masks 3D JSON
        output_clusters_ply: Output clusters-only PLY
        sfm_sparse_dir: Optional COLMAP sparse directory
        output_combined_ply: Optional combined PLY output
    """
    logger.info("=" * 80)
    logger.info("3D Cluster Visualization")
    logger.info("=" * 80)

    # Create clusters PLY (always)
    cluster_xyz, cluster_rgb = create_clusters_ply(
        clusters_json,
        masks_3d_json,
        output_clusters_ply
    )

    # Create combined PLY if requested
    if sfm_sparse_dir and output_combined_ply:
        combined_xyz, combined_rgb = create_combined_ply(
            clusters_json,
            masks_3d_json,
            sfm_sparse_dir,
            output_combined_ply
        )

    logger.info("=" * 80)
    logger.info("Visualization complete!")
    logger.info("=" * 80)

    return cluster_xyz, cluster_rgb


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='Visualize 3D mask clusters')
    parser.add_argument('--clusters', required=True,
                       help='Clusters JSON')
    parser.add_argument('--masks-3d', required=True,
                       help='Masks 3D JSON')
    parser.add_argument('--output-clusters-ply', required=True,
                       help='Output clusters-only PLY')
    parser.add_argument('--sfm-sparse-dir', default=None,
                       help='Optional: COLMAP sparse directory for combined PLY')
    parser.add_argument('--output-combined-ply', default=None,
                       help='Optional: Output combined PLY (SFM + clusters)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        cluster_xyz, cluster_rgb = run_visualization(
            args.clusters,
            args.masks_3d,
            args.output_clusters_ply,
            args.sfm_sparse_dir,
            args.output_combined_ply
        )

        print(f"\n✅ Visualization complete!")
        print(f"   Cluster points: {len(cluster_xyz):,}")
        print(f"   Clusters PLY: {args.output_clusters_ply}")
        if args.output_combined_ply:
            print(f"   Combined PLY: {args.output_combined_ply}")

    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
