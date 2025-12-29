"""
Scale Alignment using Depth Ground Truth

Aligns COLMAP reconstruction scale to absolute metric scale using depth maps.

Usage:
    python -m src.scale_alignment \
        --sparse-dir data/sfm/sparse/0 \
        --rgb-dir data/rgb \
        --depth-dir data/depth \
        --calib calib/rgb_camera_info.json \
        --output data/sfm/scale_factor.json
"""
import numpy as np
import cv2
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .calib_io import load_camera_info
from .colmap_io import read_points3D_binary, read_images_binary, read_cameras_binary
from .utils import find_rgb_depth_pairs

logger = logging.getLogger(__name__)


def project_3d_to_2d(
    point_3d: np.ndarray,
    qvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray
) -> Tuple[bool, np.ndarray]:
    """
    Project 3D point to 2D image plane (simple pinhole).

    Args:
        point_3d: 3D point in world coordinates
        qvec: Camera rotation quaternion (w, x, y, z)
        tvec: Camera translation
        K: Camera intrinsic matrix

    Returns:
        (is_valid, pixel_xy)
    """
    # Quaternion to rotation matrix
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

    # Transform to camera coordinates
    point_cam = R @ point_3d + tvec

    # Check if behind camera
    if point_cam[2] <= 0:
        return False, np.array([0, 0])

    # Project
    point_proj = K @ point_cam
    pixel = point_proj[:2] / point_proj[2]

    return True, pixel


def compute_scale_factor_for_image(
    sparse_points: Dict,
    image: 'Image',
    depth_map: np.ndarray,
    K: np.ndarray,
    max_samples: int = 1000
) -> Tuple[float, int]:
    """
    Compute scale factor for one image by comparing COLMAP depths with real depths.

    Args:
        sparse_points: COLMAP 3D points
        image: COLMAP image with pose
        depth_map: Depth map in meters (H, W)
        K: Camera intrinsic matrix
        max_samples: Maximum points to sample

    Returns:
        (scale_factor, n_valid): scale = depth_real / depth_colmap
    """
    h, w = depth_map.shape

    # Collect valid correspondences
    depth_colmap_list = []
    depth_real_list = []

    # Sample from image's observations
    point_ids = image.point3D_ids
    xys = image.xys

    # Sample if too many
    if len(point_ids) > max_samples:
        indices = np.random.choice(len(point_ids), max_samples, replace=False)
    else:
        indices = range(len(point_ids))

    for idx in indices:
        point_id = point_ids[idx]
        if point_id == -1:  # Invalid point
            continue

        if point_id not in sparse_points:
            continue

        point_3d = sparse_points[point_id].xyz
        pixel_xy = xys[idx]

        # Get depth from COLMAP (distance from camera)
        qvec, tvec = image.qvec, image.tvec
        qw, qx, qy, qz = qvec
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        point_cam = R @ point_3d + tvec
        depth_colmap = point_cam[2]  # Z coordinate in camera frame

        if depth_colmap <= 0:
            continue

        # Get depth from depth map
        u, v = int(pixel_xy[0]), int(pixel_xy[1])
        if not (0 <= u < w and 0 <= v < h):
            continue

        depth_real = depth_map[v, u]

        # Valid depth check
        if depth_real < 0.1 or depth_real > 10.0 or np.isnan(depth_real) or np.isinf(depth_real):
            continue

        depth_colmap_list.append(depth_colmap)
        depth_real_list.append(depth_real)

    if len(depth_real_list) < 10:
        return 0.0, 0

    # Compute scale: depth_real = scale * depth_colmap
    depth_colmap_arr = np.array(depth_colmap_list)
    depth_real_arr = np.array(depth_real_list)

    # Robust estimation: use median of ratios
    ratios = depth_real_arr / depth_colmap_arr
    scale_factor = np.median(ratios)

    return float(scale_factor), len(depth_real_list)


def run_scale_alignment(
    sparse_dir: str,
    rgb_dir: str,
    depth_dir: str,
    calib_path: str,
    output_json: str,
    min_valid_images: int = 3
) -> Dict:
    """
    Compute scale factor by comparing COLMAP sparse points with depth ground truth.

    Args:
        sparse_dir: COLMAP sparse/0 directory
        rgb_dir: RGB images directory
        depth_dir: Depth images directory (mm or m)
        calib_path: RGB camera calibration JSON
        output_json: Output JSON for scale factor
        min_valid_images: Minimum images needed for reliable scale

    Returns:
        Scale alignment result
    """
    logger.info("=" * 80)
    logger.info("Scale Alignment (Depth Ground Truth)")
    logger.info("=" * 80)

    # Load COLMAP data
    sparse_path = Path(sparse_dir)
    logger.info(f"Loading COLMAP data: {sparse_dir}")

    points3D = read_points3D_binary(str(sparse_path / "points3D.bin"))
    images = read_images_binary(str(sparse_path / "images.bin"))
    cameras = read_cameras_binary(str(sparse_path / "cameras.bin"))

    logger.info(f"  3D Points: {len(points3D):,}")
    logger.info(f"  Images: {len(images)}")

    # Load calibration
    calib = load_camera_info(calib_path)
    logger.info(f"RGB camera: {calib.width}×{calib.height}")

    # Find RGB-Depth pairs
    pairs = find_rgb_depth_pairs(rgb_dir, depth_dir)
    logger.info(f"Found {len(pairs)} RGB-Depth pairs")

    if len(pairs) == 0:
        raise ValueError("No RGB-Depth pairs found!")

    # Build image name index
    image_name_to_id = {img.name: img_id for img_id, img in images.items()}

    # Compute scale factor for each image
    scale_factors = []
    valid_images = []

    for rgb_path, depth_path, pair_id in pairs:
        # Find matching COLMAP image
        rgb_filename = Path(rgb_path).name

        if rgb_filename not in image_name_to_id:
            logger.debug(f"  {pair_id}: not in COLMAP reconstruction")
            continue

        img_id = image_name_to_id[rgb_filename]
        image = images[img_id]

        # Load depth map
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            logger.warning(f"  {pair_id}: failed to load depth")
            continue

        # Convert to meters
        if depth.max() > 100:  # Likely in mm
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)

        # Compute scale for this image
        scale, n_valid = compute_scale_factor_for_image(
            points3D, image, depth, calib.K
        )

        if n_valid >= 10:
            scale_factors.append(scale)
            valid_images.append(pair_id)
            logger.info(f"  {pair_id}: scale={scale:.4f} ({n_valid} points)")
        else:
            logger.debug(f"  {pair_id}: insufficient points ({n_valid})")

    if len(scale_factors) < min_valid_images:
        raise ValueError(
            f"Insufficient valid images for scale estimation: {len(scale_factors)}/{min_valid_images} required"
        )

    # Compute robust global scale
    scale_factors = np.array(scale_factors)

    # Remove outliers (MAD-based)
    median_scale = np.median(scale_factors)
    mad = np.median(np.abs(scale_factors - median_scale))

    if mad > 0:
        threshold = 3 * 1.4826 * mad  # 3-sigma equivalent
        inliers = np.abs(scale_factors - median_scale) < threshold
        scale_factors_filtered = scale_factors[inliers]
    else:
        scale_factors_filtered = scale_factors

    final_scale = np.median(scale_factors_filtered)
    scale_std = np.std(scale_factors_filtered)

    logger.info("=" * 80)
    logger.info("Scale Alignment Results:")
    logger.info(f"  Valid images: {len(scale_factors)}/{len(pairs)}")
    logger.info(f"  Inliers: {len(scale_factors_filtered)}/{len(scale_factors)}")
    logger.info(f"  Scale factor: {final_scale:.4f} ± {scale_std:.4f}")
    logger.info(f"  1 COLMAP unit = {final_scale:.4f} meters")
    logger.info(f"  1 COLMAP unit = {final_scale*1000:.2f} mm")
    logger.info("=" * 80)

    # Save result
    result = {
        'scale_factor': float(final_scale),
        'scale_std': float(scale_std),
        'unit': 'meters',
        'n_valid_images': len(scale_factors_filtered),
        'n_total_images': len(pairs),
        'valid_images': valid_images,
        'scale_mm': float(final_scale * 1000),
        'interpretation': f'1 COLMAP unit = {final_scale:.4f} m = {final_scale*1000:.2f} mm'
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Scale factor saved to: {output_path}")

    return result


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='Scale alignment using depth ground truth')
    parser.add_argument('--sparse-dir', required=True,
                       help='COLMAP sparse directory (e.g., data/sfm/sparse/0)')
    parser.add_argument('--rgb-dir', required=True,
                       help='RGB images directory')
    parser.add_argument('--depth-dir', required=True,
                       help='Depth images directory')
    parser.add_argument('--calib', required=True,
                       help='RGB camera calibration JSON')
    parser.add_argument('--output', default='data/sfm/scale_factor.json',
                       help='Output JSON path')
    parser.add_argument('--min-valid-images', type=int, default=3,
                       help='Minimum valid images required (default: 3)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        result = run_scale_alignment(
            args.sparse_dir,
            args.rgb_dir,
            args.depth_dir,
            args.calib,
            args.output,
            args.min_valid_images
        )

        print(f"\n✅ Scale alignment complete!")
        print(f"   Scale factor: {result['scale_factor']:.4f}")
        print(f"   1 COLMAP unit = {result['scale_mm']:.2f} mm")
        print(f"   Valid images: {result['n_valid_images']}")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Scale alignment failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
