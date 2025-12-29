"""
3D Crack Clustering and Deduplication

Clusters crack points in 3D space to:
1. Remove duplicates (same crack detected in multiple images)
2. Merge fragments (one crack split across multiple detections)
3. Assign unique ID to each physical crack
4. Map clusters to visible images for 2D measurement

Uses DBSCAN clustering on dense point cloud crack points.

Usage:
    python -m src.cluster_cracks_3d \
        --crack-cloud outputs/dense_masked_cloud.ply \
        --sparse-dir data/sfm/dense/sparse \
        --output outputs/crack_clusters.json \
        --eps 0.05 \
        --min-samples 10
"""
import numpy as np
import json
import logging
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from .colmap_io import read_images_binary, read_cameras_binary

logger = logging.getLogger(__name__)


def load_crack_points_from_ply(ply_path: str, crack_color: Tuple[int, int, int] = (255, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load crack points from colored point cloud.

    Args:
        ply_path: Path to PLY with colored crack points
        crack_color: RGB color used for crack points

    Returns:
        (crack_points, crack_indices): Crack point coordinates and their indices in original cloud
    """
    logger.info(f"Loading point cloud: {ply_path}")

    pcd = o3d.io.read_point_cloud(ply_path)
    all_points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Convert colors to uint8
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    logger.info(f"  Total points: {len(all_points):,}")

    # Extract crack points (matching crack_color)
    crack_mask = np.all(colors == crack_color, axis=1)
    crack_indices = np.where(crack_mask)[0]
    crack_points = all_points[crack_mask]

    logger.info(f"  Crack points: {len(crack_points):,} ({len(crack_points)/len(all_points)*100:.1f}%)")

    return crack_points, crack_indices


def cluster_cracks_3d(
    crack_points: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 10,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Cluster crack points in 3D using DBSCAN.

    Args:
        crack_points: Crack points (N, 3) in arbitrary units
        eps: Maximum distance between points in same cluster (relative units)
        min_samples: Minimum points to form cluster
        metric: Distance metric

    Returns:
        labels: Cluster labels for each point (-1 = noise)
    """
    logger.info("Clustering crack points...")
    logger.info(f"  DBSCAN parameters: eps={eps} (relative units), min_samples={min_samples}")

    if len(crack_points) == 0:
        return np.array([])

    # Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = clustering.fit_predict(crack_points)

    # Statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    logger.info(f"  Clusters found: {n_clusters}")
    logger.info(f"  Noise points: {n_noise:,} ({n_noise/len(labels)*100:.1f}%)")

    return labels


def project_point_to_camera(
    point_3d: np.ndarray,
    qvec: np.ndarray,
    tvec: np.ndarray,
    camera_params: Dict,
    image_width: int,
    image_height: int
) -> Tuple[bool, np.ndarray]:
    """
    Project 3D point to camera image plane.

    Args:
        point_3d: 3D point in world coordinates (3,)
        qvec: Camera rotation as quaternion (w, x, y, z)
        tvec: Camera translation (3,)
        camera_params: Camera intrinsic parameters
        image_width: Image width
        image_height: Image height

    Returns:
        (is_visible, pixel_xy):
            is_visible: True if point projects within image bounds
            pixel_xy: Pixel coordinates (u, v)
    """
    # Quaternion to rotation matrix
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

    # World to camera transformation
    point_cam = R @ point_3d + tvec

    # Check if point is behind camera
    if point_cam[2] <= 0:
        return False, np.array([0, 0])

    # Project to normalized image coordinates
    x_norm = point_cam[0] / point_cam[2]
    y_norm = point_cam[1] / point_cam[2]

    # Apply camera intrinsics
    model = camera_params['model']
    params = camera_params['params']

    if model == 'SIMPLE_PINHOLE':
        f, cx, cy = params
        u = f * x_norm + cx
        v = f * y_norm + cy

    elif model == 'PINHOLE':
        fx, fy, cx, cy = params
        u = fx * x_norm + cx
        v = fy * y_norm + cy

    elif model == 'SIMPLE_RADIAL':
        f, cx, cy, k = params
        r2 = x_norm**2 + y_norm**2
        distortion = 1 + k * r2
        u = f * distortion * x_norm + cx
        v = f * distortion * y_norm + cy

    elif model == 'RADIAL':
        f, cx, cy, k1, k2 = params
        r2 = x_norm**2 + y_norm**2
        distortion = 1 + k1 * r2 + k2 * r2**2
        u = f * distortion * x_norm + cx
        v = f * distortion * y_norm + cy

    elif model == 'OPENCV':
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        r2 = x_norm**2 + y_norm**2
        r4 = r2 * r2
        radial = 1 + k1 * r2 + k2 * r4
        x_distorted = x_norm * radial + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
        y_distorted = y_norm * radial + p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm
        u = fx * x_distorted + cx
        v = fy * y_distorted + cy

    else:
        logger.warning(f"Unsupported camera model: {model}, using simple projection")
        f = params[0]
        cx, cy = params[1], params[2]
        u = f * x_norm + cx
        v = f * y_norm + cy

    # Check if pixel is within image bounds
    if 0 <= u < image_width and 0 <= v < image_height:
        return True, np.array([u, v])
    else:
        return False, np.array([u, v])


def map_cluster_to_images(
    cluster_points: np.ndarray,
    images: Dict,
    cameras: Dict,
    min_visible_points: int = 10
) -> Dict[str, int]:
    """
    Map a cluster to visible images.

    Args:
        cluster_points: Points in cluster (N, 3)
        images: COLMAP images dict
        cameras: COLMAP cameras dict
        min_visible_points: Minimum points visible to include image

    Returns:
        Dict mapping image_name to number of visible points
    """
    visible_images = {}

    for img_id, img in images.items():
        cam = cameras[img.camera_id]

        camera_params = {
            'model': cam.model,
            'params': cam.params
        }

        visible_count = 0

        for point_3d in cluster_points:
            is_visible, _ = project_point_to_camera(
                point_3d,
                img.qvec,
                img.tvec,
                camera_params,
                cam.width,
                cam.height
            )

            if is_visible:
                visible_count += 1

        if visible_count >= min_visible_points:
            visible_images[img.name] = visible_count

    return visible_images


def run_clustering(
    crack_cloud_ply: str,
    sparse_dir: str,
    output_json: str,
    output_clustered_ply: str = None,
    eps: float = 0.05,
    min_samples: int = 10,
    min_cluster_size: int = 50,
    min_visible_points: int = 10,
    crack_color: Tuple[int, int, int] = (255, 0, 0)
):
    """
    Run 3D crack clustering pipeline.

    Args:
        crack_cloud_ply: Input PLY with colored crack points
        sparse_dir: COLMAP sparse directory (for camera poses)
        output_json: Output JSON path
        output_clustered_ply: Output clustered PLY (optional)
        eps: DBSCAN epsilon (relative units)
        min_samples: DBSCAN min samples
        min_cluster_size: Minimum points to keep cluster
        min_visible_points: Minimum points visible in image to include
        crack_color: RGB color of crack points
    """
    logger.info("=" * 80)
    logger.info("3D Crack Clustering and Deduplication")
    logger.info("=" * 80)

    # Load crack points
    crack_points, crack_indices = load_crack_points_from_ply(crack_cloud_ply, crack_color)

    if len(crack_points) == 0:
        logger.warning("No crack points found! Check crack_color parameter.")
        return []

    # Load COLMAP data
    sparse_path = Path(sparse_dir)
    logger.info(f"Loading COLMAP data: {sparse_dir}")

    images = read_images_binary(str(sparse_path / "images.bin"))
    cameras = read_cameras_binary(str(sparse_path / "cameras.bin"))

    logger.info(f"  Images: {len(images)}")
    logger.info(f"  Cameras: {len(cameras)}")

    # Cluster
    labels = cluster_cracks_3d(crack_points, eps, min_samples)

    # Process each cluster
    clusters_data = []
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    logger.info("Processing clusters...")

    for label in sorted(unique_labels):
        if label == -1:
            continue  # Skip noise

        cluster_mask = labels == label
        cluster_points = crack_points[cluster_mask]
        cluster_point_indices = crack_indices[cluster_mask]

        # Filter small clusters
        if len(cluster_points) < min_cluster_size:
            logger.debug(f"  Cluster {label}: {len(cluster_points)} points (too small, skipped)")
            continue

        # Map to visible images
        visible_images = map_cluster_to_images(
            cluster_points,
            images,
            cameras,
            min_visible_points
        )

        if len(visible_images) == 0:
            logger.warning(f"  Cluster {label}: No visible images (skipped)")
            continue

        # Calculate metadata (relative coordinates)
        centroid = cluster_points.mean(axis=0)
        bbox_min = cluster_points.min(axis=0)
        bbox_max = cluster_points.max(axis=0)
        bbox_size = bbox_max - bbox_min

        cluster_data = {
            'cluster_id': int(label),
            'n_points': int(len(cluster_points)),
            'point_indices': cluster_point_indices.tolist(),
            'visible_images': visible_images,
            'centroid': centroid.tolist(),
            'bbox_min': bbox_min.tolist(),
            'bbox_max': bbox_max.tolist(),
            'bbox_size': bbox_size.tolist()
        }

        clusters_data.append(cluster_data)

        logger.info(f"  Cluster {label}: {len(cluster_points)} points, "
                   f"{len(visible_images)} images")

    logger.info(f"Total valid clusters: {len(clusters_data)}")

    # Save JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        'metadata': {
            'total_crack_points': int(len(crack_points)),
            'total_clusters': int(n_clusters),
            'valid_clusters': len(clusters_data),
            'eps': eps,
            'min_samples': min_samples,
            'min_cluster_size': min_cluster_size,
            'min_visible_points': min_visible_points,
            'crack_cloud_path': crack_cloud_ply,
            'sparse_dir': sparse_dir
        },
        'clusters': clusters_data
    }

    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved JSON: {output_json}")

    # Save clustered point cloud
    if output_clustered_ply:
        ply_path = Path(output_clustered_ply)
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        save_cluster_point_cloud(crack_points, labels, str(ply_path))

    logger.info("=" * 80)
    logger.info("Clustering complete!")
    logger.info("=" * 80)

    return clusters_data


def save_cluster_point_cloud(
    crack_points: np.ndarray,
    labels: np.ndarray,
    output_ply: str
):
    """
    Save clustered point cloud with different color per cluster.

    Args:
        crack_points: Crack points (N, 3)
        labels: Cluster labels (N,)
        output_ply: Output PLY path
    """
    logger.info(f"Saving clustered point cloud: {output_ply}")

    # Generate colors for each cluster
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Color map
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(n_clusters + 1, 3))

    # Assign colors
    point_colors = np.zeros((len(crack_points), 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        if label == -1:
            # Noise = gray
            point_colors[i] = [100, 100, 100]
        else:
            point_colors[i] = colors[label % len(colors)]

    # Save PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(crack_points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors / 255.0)

    o3d.io.write_point_cloud(output_ply, pcd, write_ascii=False)
    logger.info(f"  Saved {len(crack_points):,} points with {n_clusters} clusters")


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='3D crack clustering')
    parser.add_argument('--crack-cloud', required=True,
                       help='Input PLY with colored crack points')
    parser.add_argument('--sparse-dir', required=True,
                       help='COLMAP sparse directory (e.g., data/sfm/dense/sparse or data/sfm/sparse/0)')
    parser.add_argument('--output', required=True,
                       help='Output JSON path')
    parser.add_argument('--output-clustered-ply', default=None,
                       help='Output clustered PLY (optional)')
    parser.add_argument('--eps', type=float, default=0.05,
                       help='DBSCAN epsilon in relative units (default: 0.05)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='DBSCAN min samples (default: 10)')
    parser.add_argument('--min-cluster-size', type=int, default=50,
                       help='Minimum points to keep cluster (default: 50)')
    parser.add_argument('--min-visible-points', type=int, default=10,
                       help='Minimum points visible in image (default: 10)')
    parser.add_argument('--crack-color', type=int, nargs=3, default=[255, 0, 0],
                       help='RGB color of crack points (default: 255 0 0)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        clusters = run_clustering(
            args.crack_cloud,
            args.sparse_dir,
            args.output,
            args.output_clustered_ply,
            args.eps,
            args.min_samples,
            args.min_cluster_size,
            args.min_visible_points,
            tuple(args.crack_color)
        )

        print(f"\n✅ Clustering complete!")
        print(f"   Valid clusters: {len(clusters)}")
        if clusters:
            total_images = sum(len(c['visible_images']) for c in clusters)
            avg_images = total_images / len(clusters)
            print(f"   Average images per cluster: {avg_images:.1f}")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Clustering failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
