"""YOLO segmentation inference utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _polygon_to_list(polygon: Iterable[Iterable[float]]) -> List[List[float]]:
    """Convert polygon coordinates to a JSON-serialisable list."""
    return [[float(x), float(y)] for x, y in polygon]


def visualize_detections(
    image_path: str,
    masks_data: List[dict],
    class_names: List[str],
    output_path: str,
    colors: Optional[dict] = None
) -> None:
    """
    Visualize YOLO detections with masks, bboxes, and labels.

    Args:
        image_path: Path to original RGB image
        masks_data: List of mask dictionaries with polygon, class, score
        class_names: List of class names
        output_path: Path to save visualization
        colors: Optional dict of class_name -> [B, G, R] color
    """
    # Load original image
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Failed to load image for visualization: {image_path}")
        return

    # Create overlay for masks
    overlay = img.copy()

    # Default colors (BGR format)
    if colors is None:
        default_colors = {
            'crack': [0, 0, 255],           # Red
            'spalling': [0, 255, 0],        # Green
            'efflorescence': [255, 0, 0],   # Blue
            'exposed_rebar': [0, 255, 255], # Yellow
            'corrosion': [255, 0, 255],     # Magenta
            'water_leakage': [255, 255, 0], # Cyan
            'honeycomb': [128, 0, 255],     # Purple
        }
        colors = default_colors

    for mask_info in masks_data:
        class_name = mask_info['class']
        score = mask_info['score']
        polygon = mask_info['polygon']

        # Get color for this class (BGR)
        color = colors.get(class_name, [255, 255, 255])  # Default white

        # Convert polygon to numpy array
        polygon_np = np.array(polygon, dtype=np.int32)

        # Draw filled polygon mask with transparency
        cv2.fillPoly(overlay, [polygon_np], color)

        # Draw polygon outline
        cv2.polylines(img, [polygon_np], isClosed=True, color=color, thickness=2)

        # Calculate bounding box for label placement
        x_coords = polygon_np[:, 0]
        y_coords = polygon_np[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        # Prepare label text
        label = f"{class_name}: {score:.2f}"

        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw label background
        label_y = y_min - 10 if y_min > 30 else y_max + 20
        cv2.rectangle(
            img,
            (x_min, label_y - text_height - 5),
            (x_min + text_width + 5, label_y + baseline),
            color,
            -1  # Filled
        )

        # Draw label text
        cv2.putText(
            img,
            label,
            (x_min + 2, label_y - 2),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness,
            cv2.LINE_AA
        )

    # Blend overlay with original image (30% transparency)
    alpha = 0.3
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Save visualization
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)

    logger.debug(f"Saved visualization to {output_path}")


class YOLOSegmenter:
    """Run YOLO segmentation and export masks as JSON polygons."""

    def __init__(
        self,
        weights_path: str,
        class_names: Sequence[str],
        conf: float = 0.25,
        iou: float = 0.45,
        img_size: int = 1024,
        device: Optional[str] = None,
        max_det: int = 300,
        visualization_dir: Optional[str] = None,
        colors: Optional[dict] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency import check
            raise ImportError(
                "ultralytics package is required for YOLO inference. "
                "Install it via `pip install ultralytics`."
            ) from exc

        self.model = YOLO(weights_path)
        self.class_names = list(class_names)
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.device = device
        self.max_det = max_det
        self.visualization_dir = Path(visualization_dir) if visualization_dir else None
        self.colors = colors

        logger.debug(
            "Initialized YOLOSegmenter with weights=%s, conf=%.2f, iou=%.2f, img_size=%d",
            weights_path,
            conf,
            iou,
            img_size,
        )

        if self.visualization_dir:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"YOLO visualizations will be saved to: {self.visualization_dir}")

    def process_image(self, image_path: str, output_path: str) -> None:
        """Run inference on an image and save mask polygons to JSON."""
        image_path = str(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=image_path,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )

        if not results:
            logger.warning("No YOLO results returned for %s", image_path)
            self._write_output(output_path, [], 0, 0)
            return

        result = results[0]
        height, width = result.orig_shape[:2]

        masks_data = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)

        if masks_data is None or boxes is None or len(masks_data) == 0:
            logger.info("No segmentation masks detected for %s", image_path)
            self._write_output(output_path, [], width, height)
            return

        mask_polygons = masks_data.xy
        class_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()

        masks: List[dict] = []
        for idx, polygon in enumerate(mask_polygons):
            if idx >= len(class_ids):
                break

            class_id = class_ids[idx]
            if class_id >= len(self.class_names):
                logger.debug(
                    "Skipping detection %d in %s due to class id %d outside configured range",
                    idx,
                    image_path,
                    class_id,
                )
                continue

            polygon_points = _polygon_to_list(polygon)
            if len(polygon_points) < 3:
                continue

            masks.append(
                {
                    "class": self.class_names[class_id],
                    "class_id": int(class_id),
                    "score": float(scores[idx]),
                    "polygon": polygon_points,
                    "instance_id": f"{Path(image_path).stem}_{idx:04d}",
                }
            )

        self._write_output(output_path, masks, width, height)

        # Save visualization if directory is specified
        if self.visualization_dir and len(masks) > 0:
            vis_filename = Path(image_path).stem + ".png"
            vis_path = self.visualization_dir / vis_filename
            visualize_detections(
                image_path,
                masks,
                self.class_names,
                str(vis_path),
                self.colors
            )
            logger.info(f"Saved visualization: {vis_path}")

    def _write_output(self, output_path: Path, masks: List[dict], width: int, height: int) -> None:
        """Persist the YOLO mask predictions to disk."""
        data = {
            "image_id": output_path.stem,
            "width": int(width),
            "height": int(height),
            "masks": masks,
        }

        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.debug("Saved %d masks to %s", len(masks), output_path)


class YOLOSegmenterWithCrop(YOLOSegmenter):
    """
    Extended YOLO segmenter with crop-based multi-scale inference support.

    Maintains full backward compatibility with YOLOSegmenter while adding
    optional crop-based processing for improved fine-scale defect detection.
    """

    def __init__(
        self,
        weights_path: str,
        class_names: Sequence[str],
        conf: float = 0.25,
        iou: float = 0.45,
        img_size: int = 1024,
        device: Optional[str] = None,
        max_det: int = 300,
        visualization_dir: Optional[str] = None,
        colors: Optional[dict] = None,
        # New parameters for crop mode
        use_crop: bool = False,
        crop_grid: Tuple[int, int] = (2, 2),
        merge_iou: float = 0.5,
        save_crop_viz: bool = False
    ) -> None:
        """
        Initialize segmenter with optional crop support.

        Args:
            weights_path: Path to YOLO weights
            class_names: List of class names
            conf: Confidence threshold
            iou: IoU threshold for YOLO NMS
            img_size: YOLO input size
            device: Device for inference
            max_det: Maximum detections
            visualization_dir: Optional visualization output directory
            colors: Optional class colors
            use_crop: Enable crop-based inference
            crop_grid: (rows, cols) grid for cropping
            merge_iou: IoU threshold for merging crop detections
            save_crop_viz: Save crop boundary visualizations
        """
        # Initialize parent class
        super().__init__(
            weights_path=weights_path,
            class_names=class_names,
            conf=conf,
            iou=iou,
            img_size=img_size,
            device=device,
            max_det=max_det,
            visualization_dir=visualization_dir,
            colors=colors
        )

        # Crop-specific settings
        self.use_crop = use_crop
        self.crop_grid = crop_grid
        self.merge_iou = merge_iou
        self.save_crop_viz = save_crop_viz

        if self.use_crop:
            # Lazy import to avoid dependency if not using crop
            from .crop_inference import ImageCropper, CropInferenceEngine
            from .mask_merge import merge_class_aware

            self.cropper = ImageCropper(grid_size=crop_grid)
            self.crop_engine = CropInferenceEngine(
                yolo_model=self.model,
                cropper=self.cropper,
                conf=conf,
                iou=iou,
                imgsz=img_size,
                device=device
            )
            self.merge_func = merge_class_aware

            logger.info(
                f"Crop mode enabled: grid={crop_grid}, merge_iou={merge_iou}"
            )
        else:
            logger.debug("Crop mode disabled (standard inference)")

    def process_image(self, image_path: str, output_path: str) -> None:
        """
        Run inference on an image with optional crop processing.

        Args:
            image_path: Path to input image
            output_path: Path to save masks JSON

        Note:
            Output format is identical to parent class regardless of crop mode,
            ensuring seamless integration with downstream pipeline phases.
        """
        image_path = str(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_crop:
            self._process_with_crop(image_path, output_path)
        else:
            # Fall back to standard processing
            super().process_image(image_path, output_path)

    def _process_with_crop(self, image_path: str, output_path: Path) -> None:
        """Process image using crop-based inference."""
        import cv2

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            self._write_output(output_path, [], 0, 0)
            return

        height, width = image.shape[:2]

        # Save crop visualization if requested
        if self.save_crop_viz and self.visualization_dir:
            from .crop_inference import visualize_crops
            crops = self.cropper.create_crops(image)
            crop_viz_path = self.visualization_dir / f"{Path(image_path).stem}_crops.png"
            visualize_crops(image, crops, str(crop_viz_path))

        # Run crop-based inference
        detections, metadata = self.crop_engine.process_image(image)

        logger.debug(
            f"Crop inference for {image_path}: "
            f"{metadata['n_crops']} crops, "
            f"{metadata['total_detections']} raw detections"
        )

        # Merge overlapping detections
        merged_detections, merge_stats = self.merge_func(
            detections,
            class_names=self.class_names,
            iou_threshold=self.merge_iou,
            image_shape=(height, width)
        )

        logger.info(
            f"{image_path}: {len(detections)} detections → "
            f"{len(merged_detections)} after NMS "
            f"({len(detections) - len(merged_detections)} suppressed)"
        )

        # Convert to output format (same as parent class)
        masks = []
        for idx, det in enumerate(merged_detections):
            class_id = det['class_id']

            if class_id >= len(self.class_names):
                logger.warning(
                    f"Skipping detection with class_id {class_id} "
                    f"(out of range for {len(self.class_names)} classes)"
                )
                continue

            masks.append({
                "class": self.class_names[class_id],
                "class_id": class_id,
                "score": det['score'],
                "polygon": det['polygon'],
                "instance_id": f"{Path(image_path).stem}_{idx:04d}",
            })

        # Save output (identical format to parent class)
        self._write_output(output_path, masks, width, height)

        # Save visualization if requested
        if self.visualization_dir and len(masks) > 0:
            vis_filename = Path(image_path).stem + ".png"
            vis_path = self.visualization_dir / vis_filename
            visualize_detections(
                image_path,
                masks,
                self.class_names,
                str(vis_path),
                self.colors
            )
            logger.info(f"Saved visualization: {vis_path}")


__all__ = ["YOLOSegmenter", "YOLOSegmenterWithCrop"]
