"""
Cluster-based 2D Measurement with Pixel Calibration

Measures crack clusters by:
1. Projecting 3D cluster points to visible images
2. Creating 2D masks from projected pixels
3. Measuring in 2D (skeleton + MST)
4. Converting to mm using pixel calibration
5. Aggregating across multiple views

Usage:
    python -m src.measure_clusters_2d \
        --clusters outputs/crack_clusters.json \
        --crack-cloud outputs/dense_masked_cloud.ply \
        --sparse-dir data/sfm/dense/sparse \
        --pixel-scales calibration/pixel_scales.json \
        --output outputs/cluster_measurements.csv
"""
import numpy as np
import cv2
import json
import logging
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Tuple
import csv
from scipy.spatial import ConvexHull

from skimage.morphology import skeletonize, binary_dilation, disk

from .colmap_io import read_images_binary, read_cameras_binary
from .measure_cracks_simple import (
    measure_skeleton_length_mst,
    measure_perpendicular_width,
    preprocess_mask
)

logger = logging.getLogger(__name__)


def load_dense_point_cloud(ply_path: str) -> np.ndarray:
    """
    Load dense point cloud.

    Args:
        ply_path: Path to PLY file

    Returns:
        All points (N, 3)
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    return points


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
    """
    # Quaternion to rotation matrix
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

    point_cam = R @ point_3d + tvec

    if point_cam[2] <= 0:
        return False, np.array([0, 0])

    x_norm = point_cam[0] / point_cam[2]
    y_norm = point_cam[1] / point_cam[2]

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
        f = params[0]
        cx, cy = params[1], params[2]
        u = f * x_norm + cx
        v = f * y_norm + cy

    if 0 <= u < image_width and 0 <= v < image_height:
        return True, np.array([u, v])
    else:
        return False, np.array([u, v])


def create_mask_from_pixels(
    pixels: np.ndarray,
    img_shape: Tuple[int, int],
    dilation_radius: int = 5
) -> np.ndarray:
    """
    Create 2D binary mask from projected pixels.

    Args:
        pixels: Pixel coordinates (N, 2)
        img_shape: Image shape (H, W)
        dilation_radius: Dilation radius for smoother mask

    Returns:
        Binary mask (H, W)
    """
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if len(pixels) < 3:
        # Too few points, just mark pixels
        for u, v in pixels:
            ui, vi = int(u), int(v)
            if 0 <= ui < w and 0 <= vi < h:
                mask[vi, ui] = 255
        return mask

    # Create convex hull
    try:
        hull = ConvexHull(pixels)
        hull_pixels = pixels[hull.vertices]
        hull_pixels_int = hull_pixels.astype(np.int32)
        cv2.fillPoly(mask, [hull_pixels_int], 255)
    except Exception as e:
        logger.debug(f"ConvexHull failed, using point marking: {e}")
        # Fallback: just mark pixels
        for u, v in pixels:
            ui, vi = int(u), int(v)
            if 0 <= ui < w and 0 <= vi < h:
                mask[vi, ui] = 255

    # Dilate to smooth
    if dilation_radius > 0:
        selem = disk(dilation_radius)
        mask = binary_dilation(mask > 0, selem).astype(np.uint8) * 255

    return mask


def measure_cluster_in_image(
    cluster_points_3d: np.ndarray,
    image: 'Image',
    camera: 'Camera',
    pixel_scale: float,
    img_shape: Tuple[int, int] = (2160, 3840)
) -> Dict:
    """
    Measure a cluster in one image.

    Args:
        cluster_points_3d: 3D points in cluster
        image: COLMAP Image
        camera: COLMAP Camera
        pixel_scale: Pixel-to-mm scale for this image
        img_shape: Image shape (H, W)

    Returns:
        Measurement dict or None if failed
    """
    camera_params = {
        'model': camera.model,
        'params': camera.params
    }

    # Project all 3D points to this image
    projected_pixels = []

    for point_3d in cluster_points_3d:
        is_visible, pixel_xy = project_point_to_camera(
            point_3d,
            image.qvec,
            image.tvec,
            camera_params,
            camera.width,
            camera.height
        )

        if is_visible:
            projected_pixels.append(pixel_xy)

    if len(projected_pixels) < 5:
        # Too few pixels
        return None

    projected_pixels = np.array(projected_pixels)

    # Create mask from projected pixels
    mask = create_mask_from_pixels(projected_pixels, img_shape, dilation_radius=3)

    if mask.sum() == 0:
        return None

    # Preprocess mask
    try:
        mask_processed = preprocess_mask(mask)
    except:
        mask_processed = mask

    # Skeletonize
    skeleton = skeletonize(mask_processed > 0).astype(np.uint8) * 255

    if skeleton.sum() == 0:
        return None

    # Measure
    length_px = measure_skeleton_length_mst(skeleton)
    width_px = measure_perpendicular_width(skeleton, mask_processed)

    # Convert to mm
    length_mm = length_px * pixel_scale
    width_mm = width_px * pixel_scale

    return {
        'length_px': length_px,
        'width_px': width_px,
        'length_mm': length_mm,
        'width_mm': width_mm,
        'n_pixels': len(projected_pixels),
        'pixel_scale_mm': pixel_scale
    }


def run_measurement(
    clusters_json: str,
    crack_cloud_ply: str,
    sparse_dir: str,
    pixel_scales_json: str,
    output_csv: str,
    img_shape: Tuple[int, int] = (2160, 3840)
):
    """
    Run cluster-based measurement.

    Args:
        clusters_json: Cluster JSON from cluster_cracks_3d.py
        crack_cloud_ply: Dense point cloud PLY
        sparse_dir: COLMAP sparse directory
        pixel_scales_json: Pixel scales JSON
        output_csv: Output CSV path
        img_shape: Image shape (H, W)
    """
    logger.info("=" * 80)
    logger.info("Cluster-based 2D Measurement")
    logger.info("=" * 80)

    # Load clusters
    with open(clusters_json, 'r') as f:
        clusters_data = json.load(f)

    clusters = clusters_data['clusters']
    logger.info(f"Loaded {len(clusters)} clusters")

    # Load dense point cloud
    all_points_3d = load_dense_point_cloud(crack_cloud_ply)
    logger.info(f"Loaded dense point cloud: {len(all_points_3d):,} points")

    # Load COLMAP data
    sparse_path = Path(sparse_dir)
    images = read_images_binary(str(sparse_path / "images.bin"))
    cameras = read_cameras_binary(str(sparse_path / "cameras.bin"))

    logger.info(f"Loaded COLMAP: {len(images)} images, {len(cameras)} cameras")

    # Build image name index
    image_name_to_id = {img.name: img_id for img_id, img in images.items()}

    # Load pixel scales
    with open(pixel_scales_json, 'r') as f:
        pixel_scales = json.load(f)

    logger.info(f"Loaded pixel scales for {len(pixel_scales)} images")

    # Measure each cluster
    all_measurements = []

    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        point_indices = cluster['point_indices']
        visible_images_dict = cluster['visible_images']

        logger.info(f"Processing cluster {cluster_id}: {len(point_indices)} points, "
                   f"{len(visible_images_dict)} visible images")

        # Get 3D points for this cluster
        cluster_points_3d = all_points_3d[point_indices]

        # Measure in each visible image
        measurements_per_image = []

        for image_name, _ in visible_images_dict.items():
            # Match image
            if image_name not in image_name_to_id:
                logger.debug(f"  Image {image_name} not in COLMAP")
                continue

            img_id = image_name_to_id[image_name]
            image = images[img_id]
            camera = cameras[image.camera_id]

            # Get pixel scale
            image_stem = Path(image_name).stem

            # Try multiple matching strategies
            pixel_scale_key = None

            # 1. Exact match
            if image_stem in pixel_scales:
                pixel_scale_key = image_stem
            # 2. Remove "camera_RGB_" prefix if exists
            elif image_stem.startswith('camera_RGB_'):
                stripped = image_stem.replace('camera_RGB_', '', 1)
                if stripped in pixel_scales:
                    pixel_scale_key = stripped
            # 3. Try adding "camera_RGB_" prefix
            elif f'camera_RGB_{image_stem}' in pixel_scales:
                pixel_scale_key = f'camera_RGB_{image_stem}'

            if pixel_scale_key is None:
                logger.warning(f"  No pixel scale for {image_stem}, skipping")
                continue

            pixel_scale = pixel_scales[pixel_scale_key].get('mean_scale_mm', 1.0)

            # Measure
            measurement = measure_cluster_in_image(
                cluster_points_3d,
                image,
                camera,
                pixel_scale,
                img_shape
            )

            if measurement:
                measurement['image_name'] = image_name
                measurements_per_image.append(measurement)
                logger.debug(f"    {image_name}: length={measurement['length_mm']:.1f}mm, "
                           f"width={measurement['width_mm']:.1f}mm")

        if len(measurements_per_image) == 0:
            logger.warning(f"  Cluster {cluster_id}: No valid measurements, skipping")
            continue

        # Aggregate measurements
        length_mm_values = [m['length_mm'] for m in measurements_per_image]
        width_mm_values = [m['width_mm'] for m in measurements_per_image]

        final_measurement = {
            'cluster_id': cluster_id,
            'n_points': len(point_indices),
            'n_views': len(measurements_per_image),
            'length_mm': np.median(length_mm_values),
            'width_mm': np.median(width_mm_values),
            'length_mm_std': np.std(length_mm_values),
            'width_mm_std': np.std(width_mm_values),
            'length_mm_min': np.min(length_mm_values),
            'length_mm_max': np.max(length_mm_values),
            'width_mm_min': np.min(width_mm_values),
            'width_mm_max': np.max(width_mm_values),
            'centroid_x': cluster['centroid'][0],
            'centroid_y': cluster['centroid'][1],
            'centroid_z': cluster['centroid'][2]
        }

        all_measurements.append(final_measurement)

        logger.info(f"  Cluster {cluster_id}: length={final_measurement['length_mm']:.1f}±{final_measurement['length_mm_std']:.1f}mm, "
                   f"width={final_measurement['width_mm']:.1f}±{final_measurement['width_mm_std']:.1f}mm "
                   f"({len(measurements_per_image)} views)")

    logger.info(f"Total measured clusters: {len(all_measurements)}")

    # Save CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='') as csvfile:
        if all_measurements:
            fieldnames = all_measurements[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_measurements)

    logger.info("=" * 80)
    logger.info(f"Measurement complete! Saved to: {output_csv}")
    logger.info("=" * 80)

    return all_measurements


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='Cluster-based 2D measurement')
    parser.add_argument('--clusters', required=True,
                       help='Cluster JSON from cluster_cracks_3d.py')
    parser.add_argument('--crack-cloud', required=True,
                       help='Dense point cloud PLY (same as used for clustering)')
    parser.add_argument('--sparse-dir', required=True,
                       help='COLMAP sparse directory')
    parser.add_argument('--pixel-scales', required=True,
                       help='Pixel scales JSON from pixel_calibration.py')
    parser.add_argument('--output', required=True,
                       help='Output CSV path')
    parser.add_argument('--img-shape', type=int, nargs=2, default=[2160, 3840],
                       help='Image shape (H W), default: 2160 3840')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        measurements = run_measurement(
            args.clusters,
            args.crack_cloud,
            args.sparse_dir,
            args.pixel_scales,
            args.output,
            tuple(args.img_shape)
        )

        print(f"\n✅ Measurement complete!")
        print(f"   Measured clusters: {len(measurements)}")
        if measurements:
            total_length = sum(m['length_mm'] for m in measurements)
            print(f"   Total crack length: {total_length:.1f} mm")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Measurement failed: {e}", exc_info=True)
        import sys
        sys.exit(1)