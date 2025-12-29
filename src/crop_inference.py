#!/usr/bin/env python3
"""
Crop-based inference utilities for multi-scale defect detection.

This module enables high-resolution semantic segmentation by subdividing
images into overlapping tiles, performing inference on each tile, and
merging results back into the original image coordinate system.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageCropper:
    """
    Subdivide images into grid tiles for multi-scale inference.

    Maintains coordinate mapping to ensure seamless integration with
    SfM-based 3D reconstruction pipelines.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (2, 2),
        overlap: int = 0,
        min_crop_size: int = 320
    ):
        """
        Initialize cropper.

        Args:
            grid_size: (rows, cols) grid subdivision
            overlap: Overlap pixels between adjacent crops (future use)
            min_crop_size: Minimum crop dimension (safety check)
        """
        self.grid_size = grid_size
        self.overlap = overlap
        self.min_crop_size = min_crop_size

        logger.debug(
            f"ImageCropper initialized: grid={grid_size}, overlap={overlap}"
        )

    def create_crops(
        self,
        image: np.ndarray
    ) -> List[Dict]:
        """
        Subdivide image into grid tiles.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            List of crop dicts containing:
            - 'image': Cropped image array
            - 'offset': (x_offset, y_offset) in original coordinates
            - 'crop_id': (row, col) grid position
            - 'shape': (height, width) of crop
            - 'original_shape': (height, width) of original image
        """
        H, W = image.shape[:2]
        rows, cols = self.grid_size

        # Calculate crop dimensions
        crop_h = H // rows
        crop_w = W // cols

        # Validate crop size
        if crop_h < self.min_crop_size or crop_w < self.min_crop_size:
            logger.warning(
                f"Crop size ({crop_w}×{crop_h}) smaller than minimum "
                f"({self.min_crop_size}). Using full image instead."
            )
            return [{
                'image': image.copy(),
                'offset': (0, 0),
                'crop_id': (0, 0),
                'shape': (H, W),
                'original_shape': (H, W)
            }]

        crops = []
        for r in range(rows):
            for c in range(cols):
                # Calculate boundaries
                y_start = r * crop_h
                y_end = (r + 1) * crop_h if r < rows - 1 else H
                x_start = c * crop_w
                x_end = (c + 1) * crop_w if c < cols - 1 else W

                # Extract crop
                crop_img = image[y_start:y_end, x_start:x_end].copy()

                crops.append({
                    'image': crop_img,
                    'offset': (x_start, y_start),
                    'crop_id': (r, c),
                    'shape': crop_img.shape[:2],
                    'original_shape': (H, W)
                })

                logger.debug(
                    f"Created crop [{r},{c}]: "
                    f"offset=({x_start},{y_start}), "
                    f"size=({crop_img.shape[1]}×{crop_img.shape[0]})"
                )

        logger.info(f"Created {len(crops)} crops from {W}×{H} image")
        return crops

    def transform_to_global(
        self,
        polygon: List[List[float]],
        offset: Tuple[int, int]
    ) -> List[List[float]]:
        """
        Transform crop-local coordinates to original image coordinates.

        CRITICAL: This ensures compatibility with SfM pixel coordinates.

        Args:
            polygon: List of [x, y] points in crop coordinates
            offset: (x_offset, y_offset) of crop in original image

        Returns:
            Polygon in original image coordinates
        """
        x_offset, y_offset = offset

        global_polygon = [
            [float(x + x_offset), float(y + y_offset)]
            for x, y in polygon
        ]

        return global_polygon

    def validate_global_coords(
        self,
        polygon: List[List[float]],
        original_shape: Tuple[int, int]
    ) -> bool:
        """
        Validate that transformed coordinates are within original image bounds.

        Args:
            polygon: Polygon in original coordinates
            original_shape: (height, width) of original image

        Returns:
            True if all points are valid
        """
        H, W = original_shape

        for x, y in polygon:
            if x < 0 or x >= W or y < 0 or y >= H:
                logger.warning(
                    f"Invalid coordinate ({x:.1f}, {y:.1f}) "
                    f"outside image bounds (0-{W}, 0-{H})"
                )
                return False

        return True


class CropInferenceEngine:
    """
    Orchestrate crop-based inference pipeline.

    Workflow:
    1. Crop image into tiles
    2. Run inference on each crop
    3. Transform coordinates to original image space
    4. Aggregate results
    """

    def __init__(
        self,
        yolo_model,
        cropper: ImageCropper,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None
    ):
        """
        Initialize inference engine.

        Args:
            yolo_model: Loaded YOLO model instance
            cropper: ImageCropper instance
            conf: Confidence threshold
            iou: IoU threshold for YOLO NMS
            imgsz: YOLO input size
            device: Device for inference
        """
        self.model = yolo_model
        self.cropper = cropper
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def process_image(
        self,
        image: np.ndarray
    ) -> Tuple[List[Dict], Dict]:
        """
        Process image with crop-based inference.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            (detections, metadata) where:
            - detections: List of detection dicts with global coordinates
            - metadata: Processing statistics
        """
        # Create crops
        crops = self.cropper.create_crops(image)

        all_detections = []
        crop_stats = []

        # Process each crop
        for crop_info in crops:
            crop_img = crop_info['image']
            offset = crop_info['offset']
            crop_id = crop_info['crop_id']
            original_shape = crop_info['original_shape']

            # Run YOLO inference
            results = self.model.predict(
                source=crop_img,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                max_det=300
            )

            if not results or len(results) == 0:
                logger.debug(f"Crop {crop_id}: No detections")
                crop_stats.append({'crop_id': crop_id, 'detections': 0})
                continue

            result = results[0]

            # Extract masks
            masks_data = getattr(result, 'masks', None)
            boxes = getattr(result, 'boxes', None)

            if masks_data is None or boxes is None or len(masks_data) == 0:
                logger.debug(f"Crop {crop_id}: No masks")
                crop_stats.append({'crop_id': crop_id, 'detections': 0})
                continue

            # Process detections
            mask_polygons = masks_data.xy
            class_ids = boxes.cls.cpu().numpy().astype(int)
            scores = boxes.conf.cpu().numpy()

            n_detections = 0
            for idx, local_polygon in enumerate(mask_polygons):
                if idx >= len(class_ids):
                    break

                # Convert to list format
                local_poly_list = [[float(x), float(y)] for x, y in local_polygon]

                if len(local_poly_list) < 3:
                    continue

                # Transform to global coordinates
                global_polygon = self.cropper.transform_to_global(
                    local_poly_list,
                    offset
                )

                # Validate coordinates
                if not self.cropper.validate_global_coords(
                    global_polygon,
                    original_shape
                ):
                    logger.warning(
                        f"Crop {crop_id} detection {idx}: "
                        f"Invalid global coordinates, skipping"
                    )
                    continue

                # Store detection with global coordinates
                all_detections.append({
                    'polygon': global_polygon,
                    'class_id': int(class_ids[idx]),
                    'score': float(scores[idx]),
                    'crop_id': crop_id,
                    'local_polygon': local_poly_list  # For debugging
                })

                n_detections += 1

            crop_stats.append({'crop_id': crop_id, 'detections': n_detections})
            logger.debug(f"Crop {crop_id}: {n_detections} detections")

        # Metadata
        metadata = {
            'n_crops': len(crops),
            'crop_stats': crop_stats,
            'total_detections': len(all_detections),
            'original_shape': crops[0]['original_shape'] if crops else (0, 0)
        }

        logger.info(
            f"Crop inference complete: {len(crops)} crops, "
            f"{len(all_detections)} total detections"
        )

        return all_detections, metadata


def visualize_crops(
    image: np.ndarray,
    crops: List[Dict],
    output_path: str
):
    """
    Visualize crop boundaries on original image.

    Useful for debugging and validation.
    """
    vis_img = image.copy()

    for crop_info in crops:
        x_offset, y_offset = crop_info['offset']
        crop_h, crop_w = crop_info['shape']
        crop_id = crop_info['crop_id']

        # Draw crop boundary
        pt1 = (x_offset, y_offset)
        pt2 = (x_offset + crop_w, y_offset + crop_h)

        cv2.rectangle(vis_img, pt1, pt2, (0, 255, 0), 3)

        # Draw crop ID
        label = f"[{crop_id[0]},{crop_id[1]}]"
        cv2.putText(
            vis_img, label,
            (x_offset + 10, y_offset + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_img)

    logger.info(f"Saved crop visualization: {output_path}")


if __name__ == '__main__':
    # Unit tests
    print("Testing ImageCropper...")

    # Test 1: Create crops
    cropper = ImageCropper(grid_size=(2, 2))
    test_img = np.zeros((2160, 3840, 3), dtype=np.uint8)
    crops = cropper.create_crops(test_img)

    assert len(crops) == 4, f"Expected 4 crops, got {len(crops)}"

    # Test 2: Coordinate transformation
    # Crop [1, 0]: offset should be (1920, 0)
    crop_10 = [c for c in crops if c['crop_id'] == (1, 0)][0]
    assert crop_10['offset'] == (0, 1080), f"Wrong offset: {crop_10['offset']}"

    local_polygon = [[100.0, 200.0], [150.0, 250.0]]
    global_polygon = cropper.transform_to_global(
        local_polygon,
        crop_10['offset']
    )

    expected = [[100.0, 1280.0], [150.0, 1330.0]]
    assert global_polygon == expected, f"Transform failed: {global_polygon}"

    # Test 3: Validation
    valid = cropper.validate_global_coords(global_polygon, (2160, 3840))
    assert valid, "Valid coordinates rejected"

    invalid_polygon = [[5000.0, 100.0]]
    invalid = cropper.validate_global_coords(invalid_polygon, (2160, 3840))
    assert not invalid, "Invalid coordinates accepted"

    print("✅ All tests passed!")
