#!/usr/bin/env python3
"""
Unit tests for crop inference module.

Tests coordinate transformation, crop generation, and integration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest

from crop_inference import ImageCropper, CropInferenceEngine


class TestImageCropper:
    """Test ImageCropper functionality."""

    def test_create_crops_2x2(self):
        """Test 2x2 grid cropping."""
        cropper = ImageCropper(grid_size=(2, 2))

        # Test image: 3840 x 2160
        test_img = np.zeros((2160, 3840, 3), dtype=np.uint8)
        crops = cropper.create_crops(test_img)

        assert len(crops) == 4, f"Expected 4 crops, got {len(crops)}"

        # Check crop IDs
        crop_ids = [c['crop_id'] for c in crops]
        expected_ids = [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert crop_ids == expected_ids

        # Check crop sizes
        for crop in crops:
            h, w = crop['shape']
            assert h == 1080, f"Expected height 1080, got {h}"
            assert w == 1920, f"Expected width 1920, got {w}"

    def test_crop_offsets(self):
        """Test crop offset calculation."""
        cropper = ImageCropper(grid_size=(2, 2))
        test_img = np.zeros((2160, 3840, 3), dtype=np.uint8)
        crops = cropper.create_crops(test_img)

        # Expected offsets
        expected_offsets = {
            (0, 0): (0, 0),
            (0, 1): (1920, 0),
            (1, 0): (0, 1080),
            (1, 1): (1920, 1080)
        }

        for crop in crops:
            crop_id = crop['crop_id']
            offset = crop['offset']
            expected = expected_offsets[crop_id]
            assert offset == expected, \
                f"Crop {crop_id}: expected offset {expected}, got {offset}"

    def test_coordinate_transform(self):
        """Test local-to-global coordinate transformation."""
        cropper = ImageCropper(grid_size=(2, 2))

        # Crop [1, 0]: offset (0, 1080)
        offset = (0, 1080)

        # Local polygon in crop
        local_polygon = [[100.0, 200.0], [150.0, 250.0]]

        # Transform to global
        global_polygon = cropper.transform_to_global(local_polygon, offset)

        # Expected: add offset to each point
        expected = [[100.0, 1280.0], [150.0, 1330.0]]

        assert global_polygon == expected, \
            f"Transform failed: expected {expected}, got {global_polygon}"

    def test_coordinate_transform_crop_11(self):
        """Test transformation for crop [1,1] (bottom-right)."""
        cropper = ImageCropper(grid_size=(2, 2))

        # Crop [1, 1]: offset (1920, 1080)
        offset = (1920, 1080)

        # Local polygon
        local_polygon = [[100.0, 200.0]]

        # Transform
        global_polygon = cropper.transform_to_global(local_polygon, offset)

        # Expected
        expected = [[2020.0, 1280.0]]

        assert global_polygon == expected

    def test_validate_global_coords_valid(self):
        """Test validation of valid coordinates."""
        cropper = ImageCropper()

        polygon = [[100.0, 200.0], [500.0, 600.0]]
        original_shape = (2160, 3840)

        valid = cropper.validate_global_coords(polygon, original_shape)
        assert valid is True

    def test_validate_global_coords_invalid(self):
        """Test validation of invalid coordinates."""
        cropper = ImageCropper()

        # Coordinate outside bounds
        polygon = [[5000.0, 100.0], [100.0, 200.0]]
        original_shape = (2160, 3840)

        valid = cropper.validate_global_coords(polygon, original_shape)
        assert valid is False

    def test_min_crop_size(self):
        """Test fallback to full image if crop too small."""
        cropper = ImageCropper(grid_size=(10, 10), min_crop_size=500)

        # Small image
        test_img = np.zeros((800, 800, 3), dtype=np.uint8)
        crops = cropper.create_crops(test_img)

        # Should return single full image
        assert len(crops) == 1
        assert crops[0]['crop_id'] == (0, 0)
        assert crops[0]['offset'] == (0, 0)


class TestCoordinateConsistency:
    """Test coordinate consistency across pipeline."""

    def test_round_trip_transform(self):
        """Test that transformed coordinates stay within bounds."""
        cropper = ImageCropper(grid_size=(2, 2))
        test_img = np.zeros((2160, 3840, 3), dtype=np.uint8)
        crops = cropper.create_crops(test_img)

        for crop in crops:
            # Create a polygon in local coordinates
            crop_h, crop_w = crop['shape']
            local_polygon = [
                [10.0, 10.0],
                [crop_w - 10.0, 10.0],
                [crop_w - 10.0, crop_h - 10.0],
                [10.0, crop_h - 10.0]
            ]

            # Transform to global
            global_polygon = cropper.transform_to_global(
                local_polygon,
                crop['offset']
            )

            # Validate all points are within original image
            valid = cropper.validate_global_coords(
                global_polygon,
                crop['original_shape']
            )

            assert valid, \
                f"Crop {crop['crop_id']}: Transformed coordinates out of bounds"


class TestMaskMerge:
    """Test mask merging and NMS."""

    def test_nms_basic(self):
        """Test basic NMS suppression."""
        from mask_merge import merge_detections_nms

        # Two identical polygons, different scores
        poly1 = [[0, 0], [100, 0], [100, 100], [0, 100]]

        detections = [
            {'polygon': poly1, 'class_id': 0, 'score': 0.9},
            {'polygon': poly1, 'class_id': 0, 'score': 0.8},  # Should be suppressed
        ]

        merged = merge_detections_nms(
            detections,
            iou_threshold=0.8,
            use_shapely=False,
            image_shape=(200, 200)
        )

        assert len(merged) == 1, "Duplicate should be suppressed"
        assert merged[0]['score'] == 0.9, "Higher score should be kept"

    def test_nms_different_classes(self):
        """Test that different classes are not suppressed."""
        from mask_merge import merge_detections_nms

        poly = [[0, 0], [100, 0], [100, 100], [0, 100]]

        detections = [
            {'polygon': poly, 'class_id': 0, 'score': 0.9},
            {'polygon': poly, 'class_id': 1, 'score': 0.8},  # Different class
        ]

        merged = merge_detections_nms(
            detections,
            iou_threshold=0.8,
            use_shapely=False,
            image_shape=(200, 200)
        )

        assert len(merged) == 2, "Different classes should not be suppressed"


def run_all_tests():
    """Run all tests."""
    print("Running crop inference tests...")

    # Test ImageCropper
    print("\n[1/5] Testing ImageCropper...")
    test_cropper = TestImageCropper()
    test_cropper.test_create_crops_2x2()
    test_cropper.test_crop_offsets()
    test_cropper.test_coordinate_transform()
    test_cropper.test_coordinate_transform_crop_11()
    test_cropper.test_validate_global_coords_valid()
    test_cropper.test_validate_global_coords_invalid()
    test_cropper.test_min_crop_size()
    print("✅ ImageCropper tests passed")

    # Test coordinate consistency
    print("\n[2/5] Testing coordinate consistency...")
    test_consistency = TestCoordinateConsistency()
    test_consistency.test_round_trip_transform()
    print("✅ Coordinate consistency tests passed")

    # Test NMS
    print("\n[3/5] Testing NMS...")
    test_nms = TestMaskMerge()
    test_nms.test_nms_basic()
    test_nms.test_nms_different_classes()
    print("✅ NMS tests passed")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
