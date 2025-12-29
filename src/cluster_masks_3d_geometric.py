"""
3D Mask Clustering using Geometric Overlap (Enhanced)

Clusters 3D masks based on geometric properties:
- Principal axis (direction similarity) - prevents cross-crack merging
- Dynamic centroid distance - handles long cracks
- Gap detection - handles fragmented cracks
- BBox overlap - spatial proximity
- Point-to-point distance - fine-grained verification

Usage:
    python -m src.cluster_masks_3d_geometric \
        --masks-3d outputs/masks_3d.json \
        --output outputs/mask_clusters.json \
        --max-centroid-distance 0.5 \
        --proximity-threshold 0.05 \
        --overlap-threshold 0.3 \
        --min-angle-similarity 0.707
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_principal_axis(points_3d: np.ndarray) -> np.ndarray:
    """
    Compute principal axis (main direction) using PCA.

    Args:
        points_3d: (N, 3) 3D points

    Returns:
        axis: (3,) unit vector of principal direction
    """
    if len(points_3d) < 2:
        return np.array([1.0, 0.0, 0.0])  # Default

    # Center points
    centroid = np.mean(points_3d, axis=0)
    centered = points_3d - centroid

    # SVD
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # First principal component (largest variance direction)
        principal_axis = Vt[0]

        return principal_axis
    except np.linalg.LinAlgError:
        # Degenerate case
        return np.array([1.0, 0.0, 0.0])


def compute_angle_similarity(axis1: np.ndarray, axis2: np.ndarray) -> float:
    """
    Compute angle similarity between two direction vectors.

    Args:
        axis1, axis2: Direction vectors (unit vectors)

    Returns:
        similarity: Absolute cosine similarity (0~1)
                   1.0 = parallel, 0.0 = perpendicular
    """
    # Absolute cosine (direction-agnostic)
    cos_angle = abs(np.dot(axis1, axis2))

    # Clamp to [0, 1]
    cos_angle = np.clip(cos_angle, 0.0, 1.0)

    return cos_angle


class Cluster:
    """3D Mask Cluster with principal axis"""

    def __init__(self, initial_mask: Dict):
        """Initialize cluster with first mask"""
        self.masks = [initial_mask]
        self._update_metadata()

    def add_mask(self, mask: Dict):
        """Add mask to cluster"""
        self.masks.append(mask)
        self._update_metadata()

    def _update_metadata(self):
        """Update cluster metadata (centroid, bbox, points, principal axis)"""
        # Collect all points
        all_points = []
        for mask in self.masks:
            points = np.array(mask['points_3d'])
            all_points.append(points)

        self.all_points = np.vstack(all_points)

        # Centroid
        self.centroid_3d = np.mean(self.all_points, axis=0)

        # BBox
        self.bbox_min = np.min(self.all_points, axis=0)
        self.bbox_max = np.max(self.all_points, axis=0)

        self.bbox_3d = {
            'min': self.bbox_min,
            'max': self.bbox_max
        }

        # Principal axis
        self.principal_axis = compute_principal_axis(self.all_points)

    def get_sample_points(self, max_points: int = 500) -> np.ndarray:
        """Get sampled points (for performance)"""
        if len(self.all_points) <= max_points:
            return self.all_points

        # Random sampling
        indices = np.random.choice(len(self.all_points), max_points, replace=False)
        return self.all_points[indices]

    def to_dict(self, cluster_id: int) -> Dict:
        """Convert cluster to dict for JSON"""
        return {
            'cluster_id': cluster_id,
            'n_masks': len(self.masks),
            'n_points': len(self.all_points),
            'masks': [
                {
                    'image_id': m['image_id'],
                    'mask_id': m['mask_id'],
                    'confidence': m['confidence']
                }
                for m in self.masks
            ],
            'centroid_3d': self.centroid_3d.tolist(),
            'bbox_3d': {
                'min': self.bbox_min.tolist(),
                'max': self.bbox_max.tolist()
            },
            'principal_axis': self.principal_axis.tolist(),
            'mean_confidence': float(np.mean([m['confidence'] for m in self.masks]))
        }


def compute_bbox_iou_3d(bbox1: Dict, bbox2: Dict) -> float:
    """
    Compute 3D BBox IoU (Intersection over Union)

    Args:
        bbox1, bbox2: {'min': [x,y,z], 'max': [x,y,z]}

    Returns:
        iou: 0.0 ~ 1.0
    """
    min1 = np.array(bbox1['min'])
    max1 = np.array(bbox1['max'])
    min2 = np.array(bbox2['min'])
    max2 = np.array(bbox2['max'])

    # Intersection bbox
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    # Check if intersection exists
    if np.any(inter_min >= inter_max):
        return 0.0

    # Intersection volume
    inter_size = inter_max - inter_min
    inter_volume = np.prod(inter_size)

    # Union volume
    volume1 = np.prod(max1 - min1)
    volume2 = np.prod(max2 - min2)
    union_volume = volume1 + volume2 - inter_volume

    if union_volume <= 0:
        return 0.0

    # IoU
    iou = inter_volume / union_volume

    return iou


def get_dynamic_centroid_threshold(mask_3d: Dict, cluster: Cluster, config: Dict) -> float:
    """
    Compute dynamic centroid threshold based on BBox size.

    Larger cracks → larger threshold (handles long fragmented cracks)

    Args:
        mask_3d: Mask dict
        cluster: Cluster object
        config: Configuration dict

    Returns:
        dynamic_threshold: Adjusted centroid threshold (meters)
    """
    if not config.get('use_dynamic_threshold', True):
        return config.get('max_centroid_distance', 0.5)

    # Mask BBox size
    mask_bbox_min = np.array(mask_3d['bbox_3d']['min'])
    mask_bbox_max = np.array(mask_3d['bbox_3d']['max'])
    mask_bbox_size = np.linalg.norm(mask_bbox_max - mask_bbox_min)

    # Cluster BBox size
    cluster_bbox_size = np.linalg.norm(cluster.bbox_max - cluster.bbox_min)

    # Use larger BBox
    max_bbox_size = max(mask_bbox_size, cluster_bbox_size)

    # Dynamic threshold: 70% of BBox diagonal
    # (but don't exceed max if specified)
    dynamic_threshold = max_bbox_size * 0.7

    # Apply max limit if set
    max_limit = config.get('max_centroid_distance_limit', None)
    if max_limit is not None:
        dynamic_threshold = min(dynamic_threshold, max_limit)

    return dynamic_threshold


def compute_min_gap(mask_3d: Dict, cluster: Cluster) -> float:
    """
    Compute minimum gap (distance) between mask and cluster.

    Args:
        mask_3d: Mask dict
        cluster: Cluster object

    Returns:
        min_gap: Minimum distance (meters)
    """
    mask_points = np.array(mask_3d['points_3d'])
    cluster_points = cluster.get_sample_points(max_points=500)

    # Pairwise distance
    distances = np.linalg.norm(
        mask_points[:, None, :] - cluster_points[None, :, :],
        axis=2
    )

    # Minimum distance
    min_gap = distances.min()

    return min_gap


def check_axis_aligned(mask_3d: Dict, cluster: Cluster, config: Dict) -> bool:
    """
    Check if mask is aligned with cluster's principal axis.

    Useful for detecting long cracks that are far apart in centroid
    but aligned in direction.

    Args:
        mask_3d: Mask dict
        cluster: Cluster object
        config: Configuration dict

    Returns:
        is_aligned: True if aligned along principal axis
    """
    mask_centroid = np.array(mask_3d['centroid_3d'])

    # Vector from cluster to mask
    connection_vec = mask_centroid - cluster.centroid_3d
    connection_vec_norm = np.linalg.norm(connection_vec)

    if connection_vec_norm < 1e-6:
        return True  # Same location

    connection_vec /= connection_vec_norm

    # Check alignment with cluster's principal axis
    alignment = abs(np.dot(connection_vec, cluster.principal_axis))

    # Threshold: cos(30°) ≈ 0.866
    alignment_threshold = config.get('axis_alignment_threshold', np.cos(np.radians(30)))

    return alignment > alignment_threshold


def compute_close_ratio(mask_3d: Dict, cluster: Cluster, config: Dict) -> float:
    """
    Compute ratio of mask points close to cluster points.

    Args:
        mask_3d: Mask dict
        cluster: Cluster object
        config: Configuration dict

    Returns:
        close_ratio: Ratio of points within proximity threshold (0~1)
    """
    mask_points = np.array(mask_3d['points_3d'])

    # Sample for performance
    if len(mask_points) > 100:
        sample_indices = np.random.choice(len(mask_points), 100, replace=False)
        mask_sample = mask_points[sample_indices]
    else:
        mask_sample = mask_points

    cluster_sample = cluster.get_sample_points(max_points=500)

    # Pairwise distance (broadcasting)
    distances = np.linalg.norm(
        mask_sample[:, None, :] - cluster_sample[None, :, :],
        axis=2
    )

    # Minimum distance for each mask point
    min_distances = distances.min(axis=1)

    # Close ratio
    proximity_threshold = config.get('proximity_threshold', 0.05)
    close_ratio = np.mean(min_distances < proximity_threshold)

    return close_ratio


def compute_overlap_score(mask_3d: Dict, cluster: Cluster, config: Dict) -> float:
    """
    Compute enhanced geometric overlap score.

    Multi-criteria decision tree:
    1. Direction check (angle similarity) - prevents cross-crack merging
    2. Spatial distance (centroid + gap + axis-aligned)
    3. BBox overlap
    4. Point-level proximity

    Args:
        mask_3d: 3D mask dict
        cluster: Cluster object
        config: Configuration dict

    Returns:
        score: 0.0 ~ 1.0
    """
    mask_centroid = np.array(mask_3d['centroid_3d'])
    mask_bbox = mask_3d['bbox_3d']

    # === 1. PRINCIPAL AXIS (Direction Check) ===
    mask_points = np.array(mask_3d['points_3d'])
    mask_axis = compute_principal_axis(mask_points)
    cluster_axis = cluster.principal_axis

    angle_similarity = compute_angle_similarity(mask_axis, cluster_axis)

    # Early rejection: Different direction
    min_angle_similarity = config.get('min_angle_similarity', np.cos(np.radians(45)))

    if angle_similarity < min_angle_similarity:
        logger.debug(f"  Rejected: angle_similarity={angle_similarity:.3f} < {min_angle_similarity:.3f}")
        return 0.0  # Cross-crack or different direction

    # === 2. CENTROID DISTANCE (with dynamic threshold) ===
    dynamic_threshold = get_dynamic_centroid_threshold(mask_3d, cluster, config)
    centroid_dist = np.linalg.norm(mask_centroid - cluster.centroid_3d)

    logger.debug(f"  centroid_dist={centroid_dist:.3f}m, dynamic_threshold={dynamic_threshold:.3f}m")

    # === 3. SPATIAL CONNECTIVITY CHECK ===
    if centroid_dist > dynamic_threshold:
        # Centroid far, but check connectivity

        # 3a. Gap detection
        min_gap = compute_min_gap(mask_3d, cluster)
        gap_threshold = config.get('gap_threshold', 0.1)

        logger.debug(f"  min_gap={min_gap:.3f}m, gap_threshold={gap_threshold:.3f}m")

        # 3b. Axis alignment
        is_aligned = check_axis_aligned(mask_3d, cluster, config)

        logger.debug(f"  is_aligned={is_aligned}")

        # Reject if both fail
        if min_gap > gap_threshold and not is_aligned:
            logger.debug(f"  Rejected: centroid far, gap large, not aligned")
            return 0.0

        logger.debug(f"  Accepted by gap or alignment (centroid override)")

    # === 4. BBOX OVERLAP ===
    bbox_iou = compute_bbox_iou_3d(mask_bbox, cluster.bbox_3d)

    logger.debug(f"  bbox_iou={bbox_iou:.3f}")

    min_bbox_iou = config.get('min_bbox_iou', 0.05)

    if bbox_iou < min_bbox_iou:
        # BBox doesn't overlap, but check axis-aligned
        if not check_axis_aligned(mask_3d, cluster, config):
            logger.debug(f"  Rejected: bbox_iou low and not aligned")
            return 0.0

        logger.debug(f"  Accepted by alignment (bbox override)")

    # === 5. POINT-LEVEL PROXIMITY ===
    close_ratio = compute_close_ratio(mask_3d, cluster, config)

    logger.debug(f"  close_ratio={close_ratio:.3f}")

    # === 6. FINAL SCORE (Weighted combination) ===
    weights = config.get('score_weights', {
        'angle': 0.2,
        'bbox': 0.2,
        'point': 0.6
    })

    score = (
        weights['angle'] * angle_similarity +
        weights['bbox'] * bbox_iou +
        weights['point'] * close_ratio
    )

    logger.debug(f"  final_score={score:.3f}")

    return score


def cluster_masks_geometric(masks_3d: List[Dict], config: Optional[Dict] = None) -> List[Cluster]:
    """
    Cluster 3D masks using enhanced geometric overlap.

    Greedy clustering algorithm:
    - Process masks sequentially
    - For each mask, find best matching cluster
    - Merge if overlap score > threshold, else create new cluster

    Args:
        masks_3d: List of 3D mask dicts
        config: Configuration dict

    Returns:
        clusters: List of Cluster objects
    """
    if config is None:
        config = {
            # Direction
            'min_angle_similarity': np.cos(np.radians(45)),

            # Distance
            'max_centroid_distance': 0.5,
            'use_dynamic_threshold': True,
            'max_centroid_distance_limit': None,

            # Connectivity
            'gap_threshold': 0.1,
            'axis_alignment_threshold': np.cos(np.radians(30)),

            # BBox
            'min_bbox_iou': 0.05,

            # Point
            'proximity_threshold': 0.05,

            # Score
            'overlap_threshold': 0.3,
            'score_weights': {
                'angle': 0.2,
                'bbox': 0.2,
                'point': 0.6
            }
        }

    logger.info("Starting enhanced geometric clustering...")
    logger.info(f"  Config: {json.dumps(config, indent=2, default=str)}")

    clusters = []

    for mask_idx, mask_3d in enumerate(tqdm(masks_3d, desc="Clustering masks")):
        best_cluster = None
        best_score = 0.0

        logger.debug(f"\nMask {mask_idx}: {mask_3d['image_id']}/{mask_3d['mask_id']}")

        # Find best matching cluster
        for cluster_idx, cluster in enumerate(clusters):
            logger.debug(f"Checking cluster {cluster_idx}:")

            score = compute_overlap_score(mask_3d, cluster, config)

            if score > best_score:
                best_score = score
                best_cluster = cluster
                logger.debug(f"  → New best score: {score:.3f}")

        # Merge or create new
        overlap_threshold = config.get('overlap_threshold', 0.3)

        if best_score > overlap_threshold:
            # Merge to existing cluster
            logger.debug(f"→ Merged to cluster (score={best_score:.3f})")
            best_cluster.add_mask(mask_3d)
        else:
            # Create new cluster
            logger.debug(f"→ Created new cluster (best_score={best_score:.3f} < {overlap_threshold})")
            new_cluster = Cluster(mask_3d)
            clusters.append(new_cluster)

    logger.info(f"Clustering complete: {len(clusters)} clusters")

    return clusters


def run_clustering(
    masks_3d_json: str,
    output_json: str,
    config: Optional[Dict] = None
):
    """
    Run 3D mask clustering.

    Args:
        masks_3d_json: Input masks_3d JSON
        output_json: Output clusters JSON
        config: Configuration dict
    """
    if config is None:
        config = {
            'min_angle_similarity': np.cos(np.radians(45)),
            'max_centroid_distance': 0.5,
            'use_dynamic_threshold': True,
            'gap_threshold': 0.1,
            'min_bbox_iou': 0.05,
            'proximity_threshold': 0.05,
            'overlap_threshold': 0.3,
            'score_weights': {
                'angle': 0.2,
                'bbox': 0.2,
                'point': 0.6
            }
        }

    logger.info("=" * 80)
    logger.info("3D Mask Enhanced Geometric Clustering")
    logger.info("=" * 80)

    # Load 3D masks
    logger.info(f"Loading 3D masks: {masks_3d_json}")
    with open(masks_3d_json, 'r') as f:
        data = json.load(f)

    masks_3d = data['masks']
    logger.info(f"  Loaded {len(masks_3d)} 3D masks")

    # Cluster
    clusters = cluster_masks_geometric(masks_3d, config)

    # Statistics
    logger.info("=" * 80)
    logger.info("Clustering Statistics")
    logger.info("=" * 80)
    logger.info(f"  Input masks: {len(masks_3d)}")
    logger.info(f"  Output clusters: {len(clusters)}")
    logger.info(f"  Reduction: {(1 - len(clusters)/len(masks_3d))*100:.1f}%")

    # Cluster size distribution
    cluster_sizes = [len(c.masks) for c in clusters]
    logger.info(f"  Mean masks per cluster: {np.mean(cluster_sizes):.1f}")
    logger.info(f"  Median masks per cluster: {np.median(cluster_sizes):.0f}")
    logger.info(f"  Max masks per cluster: {np.max(cluster_sizes)}")

    single_view_count = sum(1 for size in cluster_sizes if size == 1)
    logger.info(f"  Single-view clusters: {single_view_count} ({single_view_count/len(clusters)*100:.1f}%)")

    # Save
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clusters_data = [c.to_dict(i) for i, c in enumerate(clusters)]

    result = {
        'metadata': {
            'input_masks': len(masks_3d),
            'output_clusters': len(clusters),
            'reduction_percent': (1 - len(clusters)/len(masks_3d)) * 100,
            'single_view_clusters': single_view_count,
            'config': config
        },
        'clusters': clusters_data
    }

    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Saved clusters: {output_json}")
    logger.info("=" * 80)

    return clusters


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='Enhanced 3D mask clustering with direction awareness')
    parser.add_argument('--masks-3d-json', '--masks-3d', required=True, dest='masks_3d',
                       help='Input masks_3d JSON')
    parser.add_argument('--output-json', '--output', required=True, dest='output',
                       help='Output clusters JSON')

    # Direction
    parser.add_argument('--min-angle-similarity', type=float, default=0.707,
                       help='Minimum angle similarity (cos, default: 0.707 = 45deg)')

    # Distance
    parser.add_argument('--max-centroid-distance', type=float, default=0.5,
                       help='Maximum centroid distance (meters, default: 0.5)')
    parser.add_argument('--use-dynamic-threshold', action='store_true', default=True,
                       help='Use dynamic threshold based on BBox size (default: True)')

    # Connectivity
    parser.add_argument('--max-gap-distance', '--gap-threshold', type=float, default=0.1, dest='gap_threshold',
                       help='Gap threshold for connectivity (meters, default: 0.1)')

    # Score
    parser.add_argument('--close-point-threshold', '--proximity-threshold', type=float, default=0.05, dest='proximity_threshold',
                       help='Point proximity threshold (meters, default: 0.05)')
    parser.add_argument('--min-close-ratio', '--overlap-threshold', type=float, default=0.3, dest='overlap_threshold',
                       help='Overlap score threshold (0-1, default: 0.3)')

    # BBox
    parser.add_argument('--min-bbox-iou', type=float, default=0.05,
                       help='Minimum BBox IoU for merging (0-1, default: 0.05)')

    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    config = {
        # Direction
        'min_angle_similarity': args.min_angle_similarity,

        # Distance
        'max_centroid_distance': args.max_centroid_distance,
        'use_dynamic_threshold': args.use_dynamic_threshold,
        'max_centroid_distance_limit': None,

        # Connectivity
        'gap_threshold': args.gap_threshold,
        'axis_alignment_threshold': np.cos(np.radians(30)),

        # BBox
        'min_bbox_iou': args.min_bbox_iou,

        # Point
        'proximity_threshold': args.proximity_threshold,

        # Score
        'overlap_threshold': args.overlap_threshold,
        'score_weights': {
            'angle': 0.2,
            'bbox': 0.2,
            'point': 0.6
        }
    }

    try:
        with open(args.masks_3d, 'r') as f:
            data = json.load(f)

        clusters = run_clustering(
            args.masks_3d,
            args.output,
            config
        )

        print(f"\n✅ Clustering complete!")
        print(f"   Input masks: {len(data['masks'])}")
        print(f"   Output clusters: {len(clusters)}")
        print(f"   Reduction: {(1 - len(clusters)/len(data['masks']))*100:.1f}%")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Clustering failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
