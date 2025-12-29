"""
Dense Point Cloud Mask Overlay

Overlays YOLO crack masks onto COLMAP dense point cloud (fused.ply).
Projects 3D points to camera views and checks mask coverage.

Usage:
    python -m src.point_cloud_overlay_dense \
        --dense-ply data/sfm/dense/fused.ply \
        --sparse-dir data/sfm/sparse/0 \
        --masks-dir data/yolo_masks \
        --output outputs/dense_masked_cloud.ply \
        --min-votes 1
"""
import numpy as np
import json
import logging
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Tuple
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon
from tqdm import tqdm

from .colmap_io import read_images_binary, read_cameras_binary

logger = logging.getLogger(__name__)


def load_yolo_mask(mask_path: Path) -> Dict:
    """Load YOLO mask JSON"""
    with open(mask_path, 'r') as f:
        return json.load(f)


def is_pixel_in_crack_mask(mask_json: Dict, pixel_xy: Tuple[float, float]) -> bool:
    """
    Check if pixel is inside any crack polygon.

    Args:
        mask_json: YOLO mask JSON data
        pixel_xy: Pixel coordinates (u, v)

    Returns:
        True if pixel is in crack mask
    """
    if 'masks' not in mask_json:
        return False

    point = ShapelyPoint(pixel_xy)

    for mask in mask_json['masks']:
        if mask['class'] == 'crack':
            polygon_coords = mask['polygon']
            if len(polygon_coords) < 3:
                continue

            try:
                poly = Polygon(polygon_coords)
                if poly.is_valid and poly.contains(point):
                    return True
            except Exception as e:
                logger.debug(f"Invalid polygon: {e}")
                continue

    return False


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
    # Support different COLMAP camera models
    model = camera_params['model']
    params = camera_params['params']

    if model == 'SIMPLE_PINHOLE':
        # params: f, cx, cy
        f, cx, cy = params
        u = f * x_norm + cx
        v = f * y_norm + cy

    elif model == 'PINHOLE':
        # params: fx, fy, cx, cy
        fx, fy, cx, cy = params
        u = fx * x_norm + cx
        v = fy * y_norm + cy

    elif model == 'SIMPLE_RADIAL':
        # params: f, cx, cy, k
        f, cx, cy, k = params
        r2 = x_norm**2 + y_norm**2
        distortion = 1 + k * r2
        u = f * distortion * x_norm + cx
        v = f * distortion * y_norm + cy

    elif model == 'RADIAL':
        # params: f, cx, cy, k1, k2
        f, cx, cy, k1, k2 = params
        r2 = x_norm**2 + y_norm**2
        distortion = 1 + k1 * r2 + k2 * r2**2
        u = f * distortion * x_norm + cx
        v = f * distortion * y_norm + cy

    elif model == 'OPENCV':
        # params: fx, fy, cx, cy, k1, k2, p1, p2
        # OpenCV distortion model with radial (k1, k2) and tangential (p1, p2) distortion
        fx, fy, cx, cy, k1, k2, p1, p2 = params

        r2 = x_norm**2 + y_norm**2
        r4 = r2 * r2

        # Radial distortion
        radial = 1 + k1 * r2 + k2 * r4

        # Tangential distortion
        x_distorted = x_norm * radial + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
        y_distorted = y_norm * radial + p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm

        u = fx * x_distorted + cx
        v = fy * y_distorted + cy

    else:
        logger.warning(f"Unsupported camera model: {model}, using simple projection")
        # Fallback to simple pinhole
        f = params[0]
        cx, cy = params[1], params[2]
        u = f * x_norm + cx
        v = f * y_norm + cy

    # Check if pixel is within image bounds
    if 0 <= u < image_width and 0 <= v < image_height:
        return True, np.array([u, v])
    else:
        return False, np.array([u, v])


def overlay_masks_on_dense_pointcloud(
    dense_ply: str,
    sparse_dir: str,
    masks_dir: str,
    output_ply: str,
    crack_color: Tuple[int, int, int] = (255, 0, 0),
    min_votes: int = 1,
    max_points: int = None
):
    """
    Overlay YOLO masks on dense point cloud.

    Projects each 3D point to all camera views and checks mask coverage.
    A point is marked as crack if it's inside a crack mask in at least min_votes views.

    Args:
        dense_ply: Dense point cloud PLY path (e.g., data/sfm/dense/fused.ply)
        sparse_dir: COLMAP sparse/0 directory (for camera poses)
        masks_dir: YOLO masks directory
        output_ply: Output PLY path
        crack_color: RGB color for crack points (default: red)
        min_votes: Minimum number of views that must see crack (default: 1)
        max_points: Maximum points to process (for testing, None = all)
    """
    logger.info("=" * 80)
    logger.info("Dense Point Cloud Mask Overlay")
    logger.info("=" * 80)

    # Load dense point cloud
    logger.info(f"Loading dense point cloud: {dense_ply}")

    # Check if file exists
    if not Path(dense_ply).exists():
        raise FileNotFoundError(f"Dense PLY file not found: {dense_ply}")

    pcd = o3d.io.read_point_cloud(dense_ply)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Check if point cloud is empty
    if len(points) == 0:
        raise ValueError(
            f"Point cloud is empty! File: {dense_ply}\n"
            f"Possible causes:\n"
            f"  1. Wrong file path (check fused.ply vs fused_photometric.ply)\n"
            f"  2. Dense reconstruction failed\n"
            f"  3. Corrupted PLY file"
        )

    # Convert colors from [0, 1] to [0, 255]
    if len(colors) == 0:
        logger.warning("No color information, using white for all points")
        colors = np.ones((len(points), 3), dtype=np.uint8) * 255
    elif colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    logger.info(f"  Total points: {len(points):,}")

    # Limit points for testing
    if max_points and len(points) > max_points:
        logger.info(f"  Limiting to {max_points:,} points for testing")
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    # Load COLMAP camera data
    sparse_path = Path(sparse_dir)
    logger.info(f"Loading COLMAP camera data: {sparse_dir}")

    images = read_images_binary(str(sparse_path / "images.bin"))
    cameras = read_cameras_binary(str(sparse_path / "cameras.bin"))

    logger.info(f"  Images: {len(images)}")
    logger.info(f"  Cameras: {len(cameras)}")

    # Prepare camera data
    camera_data = {}
    for img_id, img in images.items():
        cam = cameras[img.camera_id]

        # Determine camera model and parameters
        model_name = cam.model

        # COLMAP camera models
        # Reference: https://colmap.github.io/cameras.html
        camera_data[img_id] = {
            'name': img.name,
            'qvec': img.qvec,  # (w, x, y, z)
            'tvec': img.tvec,
            'model': model_name,
            'params': cam.params,
            'width': cam.width,
            'height': cam.height
        }

    # Load all masks
    logger.info(f"Loading YOLO masks from: {masks_dir}")
    masks_path = Path(masks_dir)
    masks = {}

    for mask_file in masks_path.glob("*.json"):
        try:
            mask_data = load_yolo_mask(mask_file)
            img_stem = mask_file.stem
            masks[img_stem] = mask_data
        except Exception as e:
            logger.warning(f"Failed to load mask {mask_file}: {e}")

    logger.info(f"  Loaded {len(masks)} masks")

    # Process each 3D point
    logger.info("Processing 3D points...")
    logger.info(f"  Min votes required: {min_votes}")

    crack_votes = np.zeros(len(points), dtype=np.int32)
    total_projections = 0
    valid_projections = 0

    # Project all points to all cameras
    for img_id, cam_data in tqdm(camera_data.items(), desc="Projecting to cameras"):
        image_name = cam_data['name']
        image_stem = Path(image_name).stem

        # Skip if no mask for this image
        if image_stem not in masks:
            continue

        mask_json = masks[image_stem]

        # Project all points to this camera
        for i, point_3d in enumerate(points):
            total_projections += 1

            is_visible, pixel_xy = project_point_to_camera(
                point_3d,
                cam_data['qvec'],
                cam_data['tvec'],
                {'model': cam_data['model'], 'params': cam_data['params']},
                cam_data['width'],
                cam_data['height']
            )

            if not is_visible:
                continue

            valid_projections += 1

            # Check if pixel is in crack mask
            if is_pixel_in_crack_mask(mask_json, pixel_xy):
                crack_votes[i] += 1

    # Assign colors based on votes
    crack_mask = crack_votes >= min_votes
    crack_count = np.sum(crack_mask)

    output_colors = colors.copy()
    output_colors[crack_mask] = crack_color

    logger.info(f"  Total projections: {total_projections:,}")
    logger.info(f"  Valid projections: {valid_projections:,}")
    logger.info(f"  Crack points: {crack_count:,} ({crack_count/len(points)*100:.1f}%)")

    # Vote statistics
    max_votes = np.max(crack_votes)
    logger.info(f"  Max votes per point: {max_votes}")
    if crack_count > 0:
        avg_votes = np.mean(crack_votes[crack_mask])
        logger.info(f"  Average votes for crack points: {avg_votes:.1f}")

    # Save output PLY
    logger.info(f"Saving point cloud to: {output_ply}")
    save_ply(output_ply, points, output_colors)

    logger.info("=" * 80)
    logger.info("Dense point cloud overlay complete!")
    logger.info("=" * 80)

    return {
        'total_points': len(points),
        'crack_points': int(crack_count),
        'total_projections': total_projections,
        'valid_projections': valid_projections,
        'max_votes': int(max_votes)
    }


def save_ply(filename: str, xyz: np.ndarray, rgb: np.ndarray):
    """
    Save point cloud as binary PLY.

    Args:
        filename: Output PLY path
        xyz: 3D coordinates (N, 3)
        rgb: RGB colors (N, 3), uint8
    """
    assert xyz.shape[0] == rgb.shape[0]
    assert xyz.shape[1] == 3
    assert rgb.shape[1] == 3

    # Ensure RGB is uint8
    if rgb.dtype != np.uint8:
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
            f.write(xyz[i].astype(np.float32).tobytes())
            # rgb as uint8
            f.write(rgb[i].tobytes())

    logger.info(f"Saved {len(xyz):,} points to {filename}")


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='Overlay YOLO masks on dense point cloud')
    parser.add_argument('--dense-ply', required=True,
                       help='Dense point cloud PLY (e.g., data/sfm/dense/fused.ply)')
    parser.add_argument('--sparse-dir', required=True,
                       help='COLMAP sparse directory for camera poses (e.g., data/sfm/sparse/0)')
    parser.add_argument('--masks-dir', required=True,
                       help='YOLO masks directory')
    parser.add_argument('--output', required=True,
                       help='Output PLY path')
    parser.add_argument('--crack-color', type=int, nargs=3, default=[255, 0, 0],
                       help='RGB color for crack points (default: 255 0 0)')
    parser.add_argument('--min-votes', type=int, default=1,
                       help='Minimum views required to mark as crack (default: 1)')
    parser.add_argument('--max-points', type=int, default=None,
                       help='Max points to process for testing (default: all)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        result = overlay_masks_on_dense_pointcloud(
            args.dense_ply,
            args.sparse_dir,
            args.masks_dir,
            args.output,
            tuple(args.crack_color),
            args.min_votes,
            args.max_points
        )

        print(f"\n✅ Dense point cloud overlay complete!")
        print(f"   Total points: {result['total_points']:,}")
        print(f"   Crack points: {result['crack_points']:,}")
        print(f"   Valid projections: {result['valid_projections']:,}")
        print(f"   Max votes: {result['max_votes']}")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Overlay failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
