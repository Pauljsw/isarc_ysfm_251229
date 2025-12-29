#!/usr/bin/env python3
"""
D2C Alignment + Per-pixel mm/px Scale Map Generator

1) Align depth (512x512, depth camera) to RGB camera (3840x2160) coordinates.
2) Compute per-pixel metric scale (mm/px) on the RGB image using aligned depth.

- Uses:
  - rgb_camera_info.json
  - depth_camera_info.json
  - extrinsic_depth_to_color.json
- Keeps ONLY measured depth (no hole filling / inpainting).

Usage example:
    python d2c_and_pixel_scale.py \
        --depth-dir data/depth_raw_512 \
        --rgb-calib calib/rgb_camera_info.json \
        --depth-calib calib/depth_camera_info.json \
        --extrinsic calib/extrinsic_depth_to_color.json \
        --output-dir outputs/d2c_scale_maps \
        --log-level INFO
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2

logger = logging.getLogger("D2C_PixelScale")


def setup_logging(level: str = "INFO"):
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="[%(levelname)s] %(message)s"
    )


def load_camera_info(json_path: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load camera intrinsics & distortion from JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    K = np.array(data["K"], dtype=np.float32).reshape(3, 3)
    D = np.array(data["D"], dtype=np.float32).reshape(-1, 1)
    width = int(data["width"])
    height = int(data["height"])

    return K, D, width, height


def load_extrinsic(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load depth->color extrinsic (R, t) from JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    R = np.array(data["R"], dtype=np.float32).reshape(3, 3)
    t_raw = np.array(data["t"], dtype=np.float32).reshape(3, 1)

    # Check if t is in mm (magnitude > 1) or meters (magnitude < 1)
    # If t values are large (> 1), assume mm and convert to meters
    if np.max(np.abs(t_raw)) > 1.0:
        t = t_raw / 1000.0  # mm -> meters
        logger.debug(f"Converted t from mm to meters: {t.flatten()}")
    else:
        t = t_raw  # already in meters
        logger.debug(f"t already in meters: {t.flatten()}")

    return R, t


def align_depth_to_color_single(
    depth_img_mm: np.ndarray,
    K_depth: np.ndarray,
    D_depth: np.ndarray,
    K_rgb: np.ndarray,
    D_rgb: np.ndarray,
    R_d2c: np.ndarray,
    t_d2c: np.ndarray,
    rgb_width: int,
    rgb_height: int,
    max_depth_m: float = 10.0,
) -> np.ndarray:
    """
    Align a single depth image (depth camera) to RGB camera coordinates.

    Args:
        depth_img_mm: (H_d, W_d) depth in mm (uint16 or float)
        K_depth, D_depth: depth intrinsics & distortion
        K_rgb,   D_rgb:   color intrinsics & distortion
        R_d2c, t_d2c:     transform from depth cam -> color cam
        rgb_width, rgb_height: RGB image size
        max_depth_m: maximum valid depth in meters

    Returns:
        aligned_depth_m: (H_rgb, W_rgb) depth in meters (float32),
                         0 where no measurement.
    """
    depth_img_mm = depth_img_mm.astype(np.float32)
    depth_m = depth_img_mm / 1000.0  # mm -> m

    H_d, W_d = depth_m.shape
    logger.debug(f"Depth image shape: {W_d}x{H_d}")

    # 1) Generate pixel grid in depth image
    v_coords, u_coords = np.meshgrid(
        np.arange(H_d, dtype=np.float32),
        np.arange(W_d, dtype=np.float32),
        indexing="ij"
    )  # v: row (y), u: col (x)

    depth_flat = depth_m.flatten()
    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()

    # 2) Valid depth mask
    valid_mask = (depth_flat > 0.0) & (depth_flat < max_depth_m)
    if not np.any(valid_mask):
        logger.warning("No valid depth pixels found in this image.")
        return np.zeros((rgb_height, rgb_width), dtype=np.float32)

    depth_valid = depth_flat[valid_mask]
    u_valid = u_flat[valid_mask]
    v_valid = v_flat[valid_mask]

    logger.debug(f"Valid depth pixels: {len(depth_valid)}")

    # 3) Undistort depth pixels to normalized coordinates (depth camera)
    pts_2d = np.stack([u_valid, v_valid], axis=1).astype(np.float32)
    pts_2d = pts_2d.reshape(-1, 1, 2)  # (N, 1, 2) for OpenCV

    # undistorted points in normalized coordinates of ideal depth camera
    undist_norm = cv2.undistortPoints(pts_2d, K_depth, D_depth)  # (N, 1, 2)
    undist_norm = undist_norm.reshape(-1, 2)
    x_d = undist_norm[:, 0]
    y_d = undist_norm[:, 1]

    # 4) Reconstruct 3D in depth camera coordinates
    Z_d = depth_valid  # already in meters
    X_d = x_d * Z_d
    Y_d = y_d * Z_d

    points_depth = np.stack([X_d, Y_d, Z_d], axis=1)  # (N, 3)

    # 5) Transform to color camera coordinates: P_c = R * P_d + t
    points_color = (R_d2c @ points_depth.T).T + t_d2c.T  # (N, 3)
    X_c = points_color[:, 0]
    Y_c = points_color[:, 1]
    Z_c = points_color[:, 2]

    # Invalid if behind camera or too far
    valid_Z = Z_c > 0
    if not np.any(valid_Z):
        logger.warning("All points ended up behind the color camera.")
        return np.zeros((rgb_height, rgb_width), dtype=np.float32)

    X_c = X_c[valid_Z]
    Y_c = Y_c[valid_Z]
    Z_c = Z_c[valid_Z]

    points_color_valid = np.stack([X_c, Y_c, Z_c], axis=1)

    # 6) Project to RGB image (apply color distortion as well)
    #    Since points_color_valid are already in color camera coordinates,
    #    we use rvec = [0,0,0], tvec = [0,0,0].
    obj_pts = points_color_valid.reshape(-1, 1, 3)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)

    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_rgb, D_rgb)
    img_pts = img_pts.reshape(-1, 2)
    u_c = img_pts[:, 0]
    v_c = img_pts[:, 1]

    # 7) Round & bounds checking
    u_i = np.round(u_c).astype(np.int32)
    v_i = np.round(v_c).astype(np.int32)

    in_bounds = (
        (u_i >= 0) & (u_i < rgb_width) &
        (v_i >= 0) & (v_i < rgb_height)
    )

    if not np.any(in_bounds):
        logger.warning("No projected points fall inside RGB image bounds.")
        return np.zeros((rgb_height, rgb_width), dtype=np.float32)

    u_i = u_i[in_bounds]
    v_i = v_i[in_bounds]
    Z_c = Z_c[in_bounds]

    # 8) Create aligned depth (meters) with z-buffer (nearest depth)
    aligned_depth_m = np.zeros((rgb_height, rgb_width), dtype=np.float32)

    for ui, vi, z in zip(u_i, v_i, Z_c):
        current = aligned_depth_m[vi, ui]
        if current == 0.0 or z < current:
            aligned_depth_m[vi, ui] = z

    return aligned_depth_m


def compute_mm_per_pixel_maps(
    aligned_depth_m: np.ndarray,
    K_rgb: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-pixel mm/px scale maps from aligned depth and RGB intrinsics.

    Args:
        aligned_depth_m: (H, W) depth in meters (0 where invalid)
        K_rgb: RGB camera intrinsics (3x3)

    Returns:
        mm_per_px_x, mm_per_px_y, mm_per_px_iso
    """
    fx = K_rgb[0, 0]
    fy = K_rgb[1, 1]

    Z = aligned_depth_m  # (H, W)
    mm_per_px_x = (Z / fx) * 1000.0  # mm/px in x
    mm_per_px_y = (Z / fy) * 1000.0  # mm/px in y
    mm_per_px_iso = 0.5 * (mm_per_px_x + mm_per_px_y)

    # invalid depth -> 0 scale
    invalid = (Z <= 0.0) | np.isnan(Z) | np.isinf(Z)
    mm_per_px_x[invalid] = 0.0
    mm_per_px_y[invalid] = 0.0
    mm_per_px_iso[invalid] = 0.0

    return mm_per_px_x.astype(np.float32), mm_per_px_y.astype(np.float32), mm_per_px_iso.astype(np.float32)


def process_depth_directory(
    depth_dir: str,
    rgb_calib_path: str,
    depth_calib_path: str,
    extrinsic_path: str,
    output_dir: str,
):
    depth_dir = Path(depth_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load camera parameters
    K_rgb, D_rgb, rgb_w, rgb_h = load_camera_info(rgb_calib_path)
    K_d, D_d, depth_w, depth_h = load_camera_info(depth_calib_path)
    R_d2c, t_d2c = load_extrinsic(extrinsic_path)

    logger.info(f"RGB camera: {rgb_w}x{rgb_h}")
    logger.info(f"Depth camera: {depth_w}x{depth_h}")

    depth_files = sorted(depth_dir.glob("*.png"))
    logger.info(f"Found {len(depth_files)} depth images in {depth_dir}")

    if not depth_files:
        logger.warning("No depth PNG files found. Nothing to do.")
        return

    for idx, depth_path in enumerate(depth_files):
        logger.info(f"[{idx+1}/{len(depth_files)}] Processing {depth_path.name}")

        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            logger.warning(f"  Failed to load depth image: {depth_path}")
            continue

        # Align depth to RGB
        aligned_depth_m = align_depth_to_color_single(
            depth_img_mm=depth_img,
            K_depth=K_d,
            D_depth=D_d,
            K_rgb=K_rgb,
            D_rgb=D_rgb,
            R_d2c=R_d2c,
            t_d2c=t_d2c,
            rgb_width=rgb_w,
            rgb_height=rgb_h,
        )

        # Save aligned depth (meters as .npy, mm as .png)
        base_name = depth_path.stem  # e.g., camera_DPT_...
        aligned_npy_path = output_dir / f"aligned_depth_{base_name}.npy"
        aligned_png_path = output_dir / f"aligned_depth_{base_name}.png"

        np.save(aligned_npy_path, aligned_depth_m)

        # Convert to mm for PNG (uint16, clip to 0..65535mm)
        aligned_depth_mm = np.clip(aligned_depth_m * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(aligned_png_path), aligned_depth_mm)

        logger.info(f"  Saved aligned depth (npy): {aligned_npy_path}")
        logger.info(f"  Saved aligned depth (png, mm): {aligned_png_path}")

        # Compute mm/px scale maps
        mm_px_x, mm_px_y, mm_px_iso = compute_mm_per_pixel_maps(aligned_depth_m, K_rgb)

        scale_iso_path = output_dir / f"scale_map_iso_{base_name}.npy"
        np.save(scale_iso_path, mm_px_iso)
        logger.info(f"  Saved mm/px iso scale map: {scale_iso_path}")

        # 필요하다면 x/y 방향도 따로 저장 가능
        # np.save(output_dir / f"scale_map_x_{base_name}.npy", mm_px_x)
        # np.save(output_dir / f"scale_map_y_{base_name}.npy", mm_px_y)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="D2C alignment + per-pixel mm/px scale computation"
    )
    parser.add_argument("--depth-dir", required=True, help="Directory of raw depth PNGs (mm unit)")
    parser.add_argument("--rgb-calib", required=True, help="Path to rgb_camera_info.json")
    parser.add_argument("--depth-calib", required=True, help="Path to depth_camera_info.json")
    parser.add_argument("--extrinsic", required=True, help="Path to extrinsic_depth_to_color.json")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    try:
        process_depth_directory(
            depth_dir=args.depth_dir,
            rgb_calib_path=args.rgb_calib,
            depth_calib_path=args.depth_calib,
            extrinsic_path=args.extrinsic,
            output_dir=args.output_dir,
        )
        logger.info("All done.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
