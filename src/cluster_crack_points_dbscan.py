"""
DBSCAN Clustering for Crack Points

Clusters 3D crack points using DBSCAN algorithm.
Uses mapping info to aggregate source masks per cluster.

Usage:
    python -m src.cluster_crack_points_dbscan \
        --input outputs/crack_points.json \
        --output outputs/crack_clusters.json \
        --eps 0.05 \
        --min-samples 10
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


def generate_cluster_colors(n_clusters: int) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for clusters using HSV color space.
    Uses varying saturation and value to avoid duplicates.

    Args:
        n_clusters: Number of clusters

    Returns:
        List of RGB tuples
    """
    if n_clusters == 0:
        return []

    colors = []

    # For more clusters, vary saturation and value as well
    # This creates more distinct colors
    for i in range(n_clusters):
        # Vary hue across the spectrum
        hue = (i * 0.618033988749895) % 1.0  # Golden ratio for better distribution

        # Vary saturation and value for more distinction
        sat_idx = (i // 12) % 3
        saturation = [1.0, 0.7, 0.85][sat_idx]
        value = [1.0, 0.9, 0.95][sat_idx]

        # HSV to RGB conversion
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))

    return colors


def save_ply_binary(filename: str, xyz: np.ndarray, rgb: np.ndarray):
    """Save point cloud as binary PLY."""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'wb') as f:
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

        for i in range(len(xyz)):
            f.write(xyz[i].astype(np.float32).tobytes())
            f.write(rgb[i].astype(np.uint8).tobytes())


def compute_principal_axis(points: np.ndarray) -> np.ndarray:
    """
    Compute principal axis (main direction) using PCA.

    Args:
        points: (N, 3) 3D points

    Returns:
        axis: (3,) unit vector of principal direction
    """
    if len(points) < 2:
        return np.array([1.0, 0.0, 0.0])

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        principal_axis = Vt[0]
        return principal_axis
    except np.linalg.LinAlgError:
        return np.array([1.0, 0.0, 0.0])


def angle_between_axes(axis1: np.ndarray, axis2: np.ndarray) -> float:
    """
    Compute angle between two axes (0 to 90 degrees).
    Since crack direction can be flipped, we use absolute cosine.
    """
    cos_angle = abs(np.dot(axis1, axis2))
    cos_angle = min(1.0, cos_angle)  # Numerical stability
    return np.degrees(np.arccos(cos_angle))


def split_cluster_by_direction(
    cluster: Dict,
    crack_points: List[Dict],
    split_angle_threshold: float = 45.0,
    min_points_for_split: int = 5
) -> List[Dict]:
    """
    Split a single cluster if it contains points with significantly different directions.

    Uses local direction analysis to identify distinct directional sub-groups.

    Args:
        cluster: Single cluster dict
        crack_points: Original crack points list
        split_angle_threshold: Angle difference to consider splitting (degrees)
        min_points_for_split: Minimum points required for a sub-cluster

    Returns:
        List of clusters (1 if no split, multiple if split occurred)
    """
    point_ids = cluster['point_ids']

    if len(point_ids) < min_points_for_split * 2:
        return [cluster]

    # Build point lookup
    point_lookup = {p['point_id']: p for p in crack_points}
    points_xyz = np.array([point_lookup[pid]['xyz'] for pid in point_ids if pid in point_lookup])
    valid_point_ids = [pid for pid in point_ids if pid in point_lookup]

    if len(points_xyz) < min_points_for_split * 2:
        return [cluster]

    # Compute local tangent direction for each point using k nearest neighbors
    from sklearn.neighbors import NearestNeighbors

    k = min(7, len(points_xyz) - 1)
    if k < 3:
        return [cluster]

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points_xyz)
    _, indices = nbrs.kneighbors(points_xyz)

    # Compute local direction for each point
    local_directions = []
    for i, neighbors in enumerate(indices):
        neighbor_points = points_xyz[neighbors]

        # PCA on local neighborhood
        centroid = np.mean(neighbor_points, axis=0)
        centered = neighbor_points - centroid

        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            direction = Vt[0]  # Principal direction
            # Ensure consistent direction (positive x component)
            if direction[0] < 0:
                direction = -direction
            local_directions.append(direction)
        except:
            local_directions.append(np.array([1.0, 0.0, 0.0]))

    local_directions = np.array(local_directions)

    # Use agglomerative clustering on directions
    from sklearn.cluster import AgglomerativeClustering

    # Convert directions to angle-based distance matrix
    # Use cosine similarity (absolute value since direction can be flipped)
    n_points = len(local_directions)

    # Try to find 2-3 directional groups
    best_split = None
    best_score = 0

    for n_dir_clusters in [2, 3]:
        if n_points < n_dir_clusters * min_points_for_split:
            continue

        try:
            dir_clustering = AgglomerativeClustering(
                n_clusters=n_dir_clusters,
                metric='cosine',
                linkage='average'
            ).fit(local_directions)

            dir_labels = dir_clustering.labels_

            # Check if sub-clusters have different directions
            sub_axes = []
            sub_counts = []
            valid_split = True

            for label in range(n_dir_clusters):
                mask = dir_labels == label
                count = np.sum(mask)

                if count < min_points_for_split:
                    valid_split = False
                    break

                sub_points = points_xyz[mask]
                sub_axis = compute_principal_axis(sub_points)
                sub_axes.append(sub_axis)
                sub_counts.append(count)

            if not valid_split:
                continue

            # Check angle between sub-cluster axes
            max_angle = 0
            for i in range(len(sub_axes)):
                for j in range(i + 1, len(sub_axes)):
                    angle = angle_between_axes(sub_axes[i], sub_axes[j])
                    max_angle = max(max_angle, angle)

            # Only split if angle difference is significant
            if max_angle >= split_angle_threshold:
                # Score based on angle difference and balance
                balance = min(sub_counts) / max(sub_counts)
                score = max_angle * balance

                if score > best_score:
                    best_score = score
                    best_split = (dir_labels, n_dir_clusters, max_angle)

        except Exception as e:
            continue

    if best_split is None:
        return [cluster]

    dir_labels, n_dir_clusters, split_angle = best_split
    logger.debug(f"  Splitting cluster {cluster['cluster_id']}: {n_dir_clusters} sub-clusters, angle={split_angle:.1f}°")

    # Create sub-clusters
    sub_clusters = []
    small_cluster_point_ids = []  # 포인트 누락 방지용

    for label in range(n_dir_clusters):
        mask = dir_labels == label
        sub_point_ids = [valid_point_ids[i] for i in range(len(valid_point_ids)) if mask[i]]
        sub_xyz = points_xyz[mask]

        if len(sub_point_ids) < min_points_for_split:
            # 작은 클러스터의 포인트들을 저장 (나중에 가장 큰 클러스터에 병합)
            small_cluster_point_ids.extend(sub_point_ids)
            continue

        # Get sub-cluster points
        sub_points = [point_lookup[pid] for pid in sub_point_ids if pid in point_lookup]

        # Aggregate source masks
        source_masks_map = defaultdict(lambda: {'n_points': 0, 'confidence_sum': 0.0})

        for point in sub_points:
            if point.get('is_synthetic', False):
                continue
            sources = point.get('sources', point.get('source_masks', []))
            for source in sources:
                key = (source['image_id'], source['mask_id'])
                source_masks_map[key]['n_points'] += 1
                source_masks_map[key]['confidence_sum'] += source.get('confidence', 1.0)

        source_masks = []
        for (image_id, mask_id), info in source_masks_map.items():
            source_masks.append({
                'image_id': image_id,
                'mask_id': mask_id,
                'n_points': info['n_points'],
                'avg_confidence': info['confidence_sum'] / info['n_points'] if info['n_points'] > 0 else 1.0
            })
        source_masks.sort(key=lambda x: x['n_points'], reverse=True)

        # Compute properties
        centroid = np.mean(sub_xyz, axis=0)
        bbox_min = np.min(sub_xyz, axis=0)
        bbox_max = np.max(sub_xyz, axis=0)
        principal_axis = compute_principal_axis(sub_xyz)

        # Confidence
        confidences = []
        for p in sub_points:
            if not p.get('is_synthetic', False):
                conf = p.get('avg_confidence', None)
                if conf is None:
                    sources = p.get('sources', p.get('source_masks', []))
                    if sources:
                        conf = np.mean([s.get('confidence', 1.0) for s in sources])
                if conf is not None:
                    confidences.append(conf)
        avg_confidence = np.mean(confidences) if confidences else 1.0

        sub_cluster = {
            'cluster_id': cluster['cluster_id'],  # Will be reassigned later
            'n_points': len(sub_point_ids),
            'point_ids': sub_point_ids,
            'centroid_3d': centroid.tolist(),
            'bbox_3d': {
                'min': bbox_min.tolist(),
                'max': bbox_max.tolist()
            },
            'principal_axis': principal_axis.tolist(),
            'source_masks': source_masks,
            'n_source_masks': len(source_masks),
            'n_views': len(set(s['image_id'] for s in source_masks)),
            'avg_confidence': float(avg_confidence),
            'split_from': cluster['cluster_id']
        }
        sub_clusters.append(sub_cluster)

    # 작은 클러스터의 포인트들을 가장 큰 클러스터에 병합
    if small_cluster_point_ids and sub_clusters:
        # 가장 큰 클러스터 찾기
        largest_idx = max(range(len(sub_clusters)), key=lambda i: sub_clusters[i]['n_points'])
        largest_cluster = sub_clusters[largest_idx]

        # 포인트 추가
        largest_cluster['point_ids'].extend(small_cluster_point_ids)
        largest_cluster['n_points'] = len(largest_cluster['point_ids'])

        # 속성 재계산
        all_xyz = np.array([point_lookup[pid]['xyz'] for pid in largest_cluster['point_ids'] if pid in point_lookup])
        largest_cluster['centroid_3d'] = np.mean(all_xyz, axis=0).tolist()
        largest_cluster['bbox_3d']['min'] = np.min(all_xyz, axis=0).tolist()
        largest_cluster['bbox_3d']['max'] = np.max(all_xyz, axis=0).tolist()
        largest_cluster['principal_axis'] = compute_principal_axis(all_xyz).tolist()

        logger.debug(f"    Merged {len(small_cluster_point_ids)} orphan points into largest sub-cluster")

    return sub_clusters if sub_clusters else [cluster]


def split_clusters_by_direction(
    clusters: List[Dict],
    crack_points: List[Dict],
    split_angle_threshold: float = 45.0,
    min_points_for_split: int = 5
) -> List[Dict]:
    """
    Stage 1.5: Split clusters containing multiple crack directions.

    Args:
        clusters: List of cluster dicts from DBSCAN
        crack_points: Original crack points list
        split_angle_threshold: Angle difference to consider splitting (degrees)
        min_points_for_split: Minimum points required for a sub-cluster

    Returns:
        List of clusters after splitting
    """
    split_clusters = []
    n_splits = 0

    for cluster in clusters:
        result = split_cluster_by_direction(
            cluster, crack_points, split_angle_threshold, min_points_for_split
        )
        split_clusters.extend(result)

        if len(result) > 1:
            n_splits += 1

    if n_splits > 0:
        logger.info(f"  Split {n_splits} clusters by direction")

    return split_clusters


def merge_clusters_by_direction(
    clusters: List[Dict],
    crack_points: List[Dict],
    merge_distance: float = 0.1,
    merge_angle_threshold: float = 30.0,
    centroid_overlap_threshold: float = 0.02
) -> List[Dict]:
    """
    Stage 2: Merge clusters with similar principal axis directions.

    Also force-merges clusters with overlapping centroids (same location problem fix).

    Args:
        clusters: List of cluster dicts from stage 1
        crack_points: Original crack points list
        merge_distance: Max centroid distance to consider merging (meters)
        merge_angle_threshold: Max angle difference to merge (degrees)
        centroid_overlap_threshold: Force merge if centroids closer than this (meters)

    Returns:
        List of merged clusters
    """
    if len(clusters) <= 1:
        return clusters

    # Build point lookup
    point_lookup = {p['point_id']: p for p in crack_points}

    # Extract centroids and axes
    n_clusters = len(clusters)
    centroids = np.array([c['centroid_3d'] for c in clusters])
    axes = np.array([c['principal_axis'] for c in clusters])

    # Union-Find for merging
    parent = list(range(n_clusters))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Find pairs to merge
    n_forced_merges = 0
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            # Check distance
            dist = np.linalg.norm(centroids[i] - centroids[j])

            # Force merge if centroids are very close (same location problem)
            if dist <= centroid_overlap_threshold:
                union(i, j)
                n_forced_merges += 1
                logger.debug(f"Force-merging overlapping clusters {i} and {j}: dist={dist:.3f}m")
                continue

            if dist > merge_distance:
                continue

            # Check angle
            angle = angle_between_axes(axes[i], axes[j])
            if angle <= merge_angle_threshold:
                union(i, j)
                logger.debug(f"Merging cluster {i} and {j}: dist={dist:.3f}m, angle={angle:.1f}°")

    if n_forced_merges > 0:
        logger.info(f"  Force-merged {n_forced_merges} overlapping cluster pairs")

    # Group clusters by their root
    groups = {}
    for i in range(n_clusters):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Create merged clusters
    merged_clusters = []
    for root, indices in groups.items():
        if len(indices) == 1:
            # No merge needed
            merged_clusters.append(clusters[indices[0]])
        else:
            # Merge multiple clusters
            merged_point_ids = []
            merged_source_masks_map = {}

            for idx in indices:
                c = clusters[idx]
                merged_point_ids.extend(c['point_ids'])

                for sm in c['source_masks']:
                    key = (sm['image_id'], sm['mask_id'])
                    if key not in merged_source_masks_map:
                        merged_source_masks_map[key] = {
                            'n_points': 0,
                            'confidence_sum': 0.0
                        }
                    merged_source_masks_map[key]['n_points'] += sm['n_points']
                    merged_source_masks_map[key]['confidence_sum'] += sm['avg_confidence'] * sm['n_points']

            # Rebuild cluster properties
            merged_xyz = np.array([point_lookup[pid]['xyz'] for pid in merged_point_ids if pid in point_lookup])
            merged_centroid = np.mean(merged_xyz, axis=0)
            merged_bbox_min = np.min(merged_xyz, axis=0)
            merged_bbox_max = np.max(merged_xyz, axis=0)
            merged_axis = compute_principal_axis(merged_xyz)

            # Rebuild source masks list
            source_masks = []
            for (image_id, mask_id), info in merged_source_masks_map.items():
                source_masks.append({
                    'image_id': image_id,
                    'mask_id': mask_id,
                    'n_points': info['n_points'],
                    'avg_confidence': info['confidence_sum'] / info['n_points']
                })
            source_masks.sort(key=lambda x: x['n_points'], reverse=True)

            # Compute average confidence (handle missing field and synthetic points)
            merged_points = [point_lookup[pid] for pid in merged_point_ids if pid in point_lookup]
            confidences = []
            for p in merged_points:
                if not p.get('is_synthetic', False):
                    conf = p.get('avg_confidence', None)
                    if conf is None:
                        sources = p.get('sources', p.get('source_masks', []))
                        if sources:
                            conf = np.mean([s.get('confidence', 1.0) for s in sources])
                    if conf is not None:
                        confidences.append(conf)
            avg_confidence = np.mean(confidences) if confidences else 1.0

            merged_cluster = {
                'cluster_id': root,  # Will be reassigned later
                'n_points': len(merged_point_ids),
                'point_ids': merged_point_ids,
                'centroid_3d': merged_centroid.tolist(),
                'bbox_3d': {
                    'min': merged_bbox_min.tolist(),
                    'max': merged_bbox_max.tolist()
                },
                'principal_axis': merged_axis.tolist(),
                'source_masks': source_masks,
                'n_source_masks': len(source_masks),
                'n_views': len(set(s['image_id'] for s in source_masks)),
                'avg_confidence': float(avg_confidence),
                'merged_from': indices
            }
            merged_clusters.append(merged_cluster)

    n_merged = n_clusters - len(merged_clusters)
    if n_merged > 0:
        logger.info(f"  Merged {n_merged} clusters (direction-aware)")

    return merged_clusters


def run_dbscan_clustering(
    input_json: str,
    output_json: str,
    eps: float = 0.05,
    min_samples: int = 10,
    output_ply: str = None,
    show_noise: bool = False,
    merge_distance: float = 0.1,
    merge_angle: float = 30.0,
    no_merge: bool = False,
    split_angle: float = 45.0,
    no_split: bool = False
):
    """
    Run DBSCAN clustering on crack points with direction-aware splitting and merging.

    Stage 1: DBSCAN clustering
    Stage 1.5: Split clusters by direction (if not disabled)
    Stage 2: Merge clusters with similar principal axis (if not disabled)

    Args:
        input_json: Input crack_points.json path
        output_json: Output crack_clusters.json path
        eps: DBSCAN epsilon (max distance between points in cluster)
        min_samples: DBSCAN minimum samples per cluster
        output_ply: Output PLY path for visualization (optional)
        show_noise: Whether to show noise points in PLY (gray)
        merge_distance: Max centroid distance to consider merging (meters)
        merge_angle: Max angle difference to merge clusters (degrees)
        no_merge: Disable direction-aware merging (Stage 2)
        split_angle: Angle threshold for splitting clusters by direction (degrees)
        no_split: Disable direction-aware splitting (Stage 1.5)
    """
    logger.info("=" * 80)
    logger.info("DBSCAN Clustering for Crack Points")
    logger.info("=" * 80)

    # Load crack points
    logger.info(f"Loading crack points: {input_json}")
    with open(input_json, 'r') as f:
        data = json.load(f)

    all_points = data['points']
    metadata = data['metadata']

    logger.info(f"  Loaded {len(all_points)} defect points")

    if len(all_points) == 0:
        logger.warning("No defect points to cluster!")
        return

    # Group points by class
    points_by_class = {}
    for point in all_points:
        cls = point.get('class', 'crack')  # Default to 'crack' for backward compatibility
        if cls not in points_by_class:
            points_by_class[cls] = []
        points_by_class[cls].append(point)

    logger.info(f"  Found {len(points_by_class)} defect classes:")
    for cls, pts in sorted(points_by_class.items()):
        logger.info(f"    {cls}: {len(pts)} points")

    # Process each class separately
    all_clusters = []
    total_noise = 0
    global_cluster_id = 0

    for cls, crack_points in sorted(points_by_class.items()):
        logger.info(f"\n{'='*80}")
        logger.info(f"Clustering class '{cls}' ({len(crack_points)} points)...")
        logger.info(f"{'='*80}")

        # Extract coordinates
        xyz = np.array([p['xyz'] for p in crack_points])

        # Run DBSCAN
        logger.info(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
        labels = clustering.labels_

        # Count clusters
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        total_noise += n_noise

        logger.info(f"  Found {n_clusters} clusters for class '{cls}'")
        logger.info(f"  Noise points: {n_noise} ({n_noise/len(crack_points)*100:.1f}%)")

        # Aggregate clusters
        logger.info("Aggregating cluster information...")
        clusters = []

        for cluster_id in sorted(unique_labels):
            if cluster_id == -1:
                continue  # Skip noise

            # Get points in this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_xyz = xyz[cluster_mask]

            # Get point data
            cluster_points = [crack_points[i] for i in cluster_indices]

            # Aggregate source masks (skip synthetic points)
            source_masks_map = defaultdict(lambda: {
                'n_points': 0,
                'confidence_sum': 0.0
            })

            for point in cluster_points:
                # Skip synthetic points - they don't have source information
                if point.get('is_synthetic', False):
                    continue
                # Handle both 'sources' and 'source_masks' field names
                sources = point.get('sources', point.get('source_masks', []))
                for source in sources:
                    key = (source['image_id'], source['mask_id'])
                    source_masks_map[key]['n_points'] += 1
                    source_masks_map[key]['confidence_sum'] += source.get('confidence', 1.0)

            # Convert to list
            source_masks = []
            for (image_id, mask_id), info in source_masks_map.items():
                source_masks.append({
                    'image_id': image_id,
                    'mask_id': mask_id,
                    'n_points': info['n_points'],
                    'avg_confidence': info['confidence_sum'] / info['n_points']
                })

            # Sort by n_points (most contributing masks first)
            source_masks.sort(key=lambda x: x['n_points'], reverse=True)

            # Compute cluster properties
            centroid = np.mean(cluster_xyz, axis=0)
            bbox_min = np.min(cluster_xyz, axis=0)
            bbox_max = np.max(cluster_xyz, axis=0)
            principal_axis = compute_principal_axis(cluster_xyz)

            # Average confidence (only from original points with confidence info)
            confidences = []
            for p in cluster_points:
                if not p.get('is_synthetic', False):
                    conf = p.get('avg_confidence', None)
                    if conf is None:
                        # Try to get confidence from sources
                        sources = p.get('sources', p.get('source_masks', []))
                        if sources:
                            conf = np.mean([s.get('confidence', 1.0) for s in sources])
                    if conf is not None:
                        confidences.append(conf)
            avg_confidence = np.mean(confidences) if confidences else 1.0

            # Create cluster entry
            cluster_entry = {
                'cluster_id': int(cluster_id),
                'class': cls,  # Add class information
                'n_points': len(cluster_points),
                'point_ids': [p['point_id'] for p in cluster_points],
                'centroid_3d': centroid.tolist(),
                'bbox_3d': {
                    'min': bbox_min.tolist(),
                    'max': bbox_max.tolist()
                },
                'principal_axis': principal_axis.tolist(),
                'source_masks': source_masks,
                'n_source_masks': len(source_masks),
                'n_views': len(set(s['image_id'] for s in source_masks)),
                'avg_confidence': float(avg_confidence)
            }

            clusters.append(cluster_entry)

            logger.debug(f"  Cluster {cluster_id}: {len(cluster_points)} points, "
                         f"{len(source_masks)} source masks, {cluster_entry['n_views']} views")

        # Stage 1.5: Direction-aware splitting (per class)
        if not no_split and len(clusters) > 0:
            logger.info(f"Stage 1.5: Direction-aware splitting for '{cls}' (angle={split_angle}°)...")
            clusters = split_clusters_by_direction(
                clusters, crack_points, split_angle
            )
            n_clusters = len(clusters)
            logger.info(f"  After splitting: {n_clusters} clusters")

        # Stage 2: Direction-aware merging (per class)
        if not no_merge and len(clusters) > 1:
            logger.info(f"Stage 2: Direction-aware merging for '{cls}' (distance={merge_distance}m, angle={merge_angle}°)...")
            clusters = merge_clusters_by_direction(
                clusters, crack_points, merge_distance, merge_angle
            )
            n_clusters = len(clusters)
            logger.info(f"  After merging: {n_clusters} clusters")

        # Add class clusters to all_clusters with temporary local IDs
        for cluster in clusters:
            cluster['class'] = cls  # Ensure class is set
            all_clusters.append(cluster)

        logger.info(f"Class '{cls}' complete: {len(clusters)} clusters")

    # Now all classes are processed, assign global cluster IDs
    logger.info(f"\n{'='*80}")
    logger.info(f"Assigning global cluster IDs across all classes...")
    logger.info(f"{'='*80}")

    # Sort all clusters by X coordinate (left to right)
    all_clusters.sort(key=lambda x: x['centroid_3d'][0])

    # Reassign cluster IDs globally
    for i, cluster in enumerate(all_clusters):
        cluster['cluster_id'] = i

    n_clusters = len(all_clusters)
    n_noise = total_noise

    # DEBUG: Check for duplicate point_ids across clusters
    all_point_ids = []
    for cluster in all_clusters:
        all_point_ids.extend(cluster['point_ids'])

    unique_point_ids = set(all_point_ids)
    if len(all_point_ids) != len(unique_point_ids):
        duplicate_count = len(all_point_ids) - len(unique_point_ids)
        logger.warning(f"⚠️ Found {duplicate_count} duplicate point_ids across clusters!")

        # Find which clusters share points
        from collections import Counter
        point_counter = Counter(all_point_ids)
        duplicates = {pid: count for pid, count in point_counter.items() if count > 1}
        logger.warning(f"  Duplicate point_ids: {list(duplicates.keys())[:10]}...")

        # Find clusters with overlapping centroids
        for i in range(len(all_clusters)):
            for j in range(i + 1, len(all_clusters)):
                dist = np.linalg.norm(
                    np.array(all_clusters[i]['centroid_3d']) - np.array(all_clusters[j]['centroid_3d'])
                )
                if dist < 0.05:  # 5cm
                    shared = set(all_clusters[i]['point_ids']) & set(all_clusters[j]['point_ids'])
                    logger.warning(f"  Clusters {i} and {j}: dist={dist:.3f}m, shared_points={len(shared)}")

    # Statistics
    logger.info("=" * 80)
    logger.info("Clustering Statistics (All Classes)")
    logger.info("=" * 80)
    logger.info(f"  Total clusters: {n_clusters}")
    logger.info(f"  Total noise points: {n_noise}")

    # Statistics per class
    if all_clusters:
        class_stats = {}
        for cluster in all_clusters:
            cls = cluster.get('class', 'unknown')
            if cls not in class_stats:
                class_stats[cls] = {'count': 0, 'points': 0}
            class_stats[cls]['count'] += 1
            class_stats[cls]['points'] += cluster['n_points']

        logger.info(f"  Clusters per class:")
        for cls, stats in sorted(class_stats.items()):
            logger.info(f"    {cls}: {stats['count']} clusters, {stats['points']} points")

        points_per_cluster = [c['n_points'] for c in all_clusters]
        logger.info(f"  Points per cluster: min={min(points_per_cluster)}, "
                    f"max={max(points_per_cluster)}, mean={np.mean(points_per_cluster):.1f}")

        views_per_cluster = [c['n_views'] for c in all_clusters]
        logger.info(f"  Views per cluster: min={min(views_per_cluster)}, "
                    f"max={max(views_per_cluster)}, mean={np.mean(views_per_cluster):.1f}")

    # Save output
    output_data = {
        'metadata': {
            'total_defect_points': len(all_points),
            'n_clusters': n_clusters,
            'noise_points': n_noise,
            'n_classes': len(points_by_class),
            'dbscan_eps': eps,
            'dbscan_min_samples': min_samples,
            'split_enabled': not no_split,
            'split_angle': split_angle,
            'merge_enabled': not no_merge,
            'merge_distance': merge_distance,
            'merge_angle': merge_angle,
            'sorted_by': 'x_coordinate',
            'input_metadata': metadata
        },
        'clusters': all_clusters
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nSaved clusters: {output_json}")

    # Generate PLY visualization if requested
    if output_ply:
        logger.info(f"\nGenerating PLY visualization...")

        # Build point_id to xyz lookup
        point_lookup = {p['point_id']: np.array(p['xyz']) for p in all_points}

        # Generate colors
        colors = generate_cluster_colors(len(all_clusters))

        # Collect all points with colors
        all_xyz = []
        all_rgb = []
        clustered_point_ids = set()

        for cluster_idx, cluster in enumerate(all_clusters):
            point_ids = cluster['point_ids']
            color = colors[cluster_idx % len(colors)] if colors else (255, 0, 0)

            for point_id in point_ids:
                if point_id in point_lookup:
                    all_xyz.append(point_lookup[point_id])
                    all_rgb.append(color)
                    clustered_point_ids.add(point_id)

        # Add noise points if requested
        if show_noise:
            noise_color = (128, 128, 128)
            noise_count = 0
            for point in all_points:
                if point['point_id'] not in clustered_point_ids:
                    all_xyz.append(np.array(point['xyz']))
                    all_rgb.append(noise_color)
                    noise_count += 1
            logger.info(f"  Added {noise_count} noise points (gray)")

        # Convert and save
        all_xyz = np.array(all_xyz)
        all_rgb = np.array(all_rgb, dtype=np.uint8)

        save_ply_binary(output_ply, all_xyz, all_rgb)
        logger.info(f"  Saved PLY: {output_ply} ({len(all_xyz)} points)")

    logger.info("=" * 80)

    return all_clusters


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(
        description='DBSCAN clustering for crack points with mapping info'
    )
    parser.add_argument('--input', required=True,
                       help='Input crack_points.json')
    parser.add_argument('--output', required=True,
                       help='Output crack_clusters.json')
    parser.add_argument('--eps', type=float, default=0.05,
                       help='DBSCAN epsilon - max distance between points (meters, default: 0.05)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='DBSCAN min samples per cluster (default: 10)')
    parser.add_argument('--output-ply', default=None,
                       help='Output PLY path for cluster visualization (optional)')
    parser.add_argument('--show-noise', action='store_true',
                       help='Show noise points in gray in PLY output')
    parser.add_argument('--merge-distance', type=float, default=0.1,
                       help='Max centroid distance for direction-aware merging (meters, default: 0.1)')
    parser.add_argument('--merge-angle', type=float, default=30.0,
                       help='Max angle difference for merging clusters (degrees, default: 30)')
    parser.add_argument('--no-merge', action='store_true',
                       help='Disable direction-aware merging (Stage 2)')
    parser.add_argument('--split-angle', type=float, default=45.0,
                       help='Angle threshold for splitting clusters by direction (degrees, default: 45)')
    parser.add_argument('--no-split', action='store_true',
                       help='Disable direction-aware splitting (Stage 1.5)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        clusters = run_dbscan_clustering(
            args.input,
            args.output,
            args.eps,
            args.min_samples,
            args.output_ply,
            args.show_noise,
            args.merge_distance,
            args.merge_angle,
            args.no_merge,
            args.split_angle,
            args.no_split
        )

        if clusters:
            print(f"\n✅ Clustering complete!")
            print(f"   Total clusters: {len(clusters)}")
            print(f"   Largest cluster: {clusters[0]['n_points']} points")
            print(f"   Output JSON: {args.output}")
            if args.output_ply:
                print(f"   Output PLY: {args.output_ply}")
        else:
            print(f"\n⚠️ No clusters found")

    except Exception as e:
        logger.error(f"Clustering failed: {e}", exc_info=True)
        import sys
        sys.exit(1)