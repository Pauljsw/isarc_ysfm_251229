#!/usr/bin/env python3
"""
Analyze pipeline statistics for multi-view defect detection.

Tracks redundancy elimination and false positive removal across pipeline stages:
- Stage 1: YOLO original detections
- Stage 2: Multi-view voting (2-stage)
- Stage 3: DBSCAN clustering

Generates statistics for paper tables and ablation studies.

Usage:
    python -m src.analyze_pipeline_statistics \
        --yolo-masks outputs/yolo_masks \
        --crack-points outputs/crack_points.json \
        --clusters outputs/crack_clusters.json \
        --output analysis/pipeline_statistics.json
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


def analyze_yolo_masks(masks_dir: Path) -> Dict:
    """
    Analyze original YOLO detections.

    Returns:
        Statistics dict with mask counts, class distribution, confidence distribution
    """
    logger.info("Analyzing YOLO masks...")

    mask_files = list(masks_dir.glob('*.json'))
    n_images = len(mask_files)

    total_masks = 0
    class_counts = defaultdict(int)
    confidence_values = []
    confidence_bins = {'low': 0, 'mid': 0, 'high': 0}

    # Per-class confidence
    class_confidences = defaultdict(list)

    for mask_file in mask_files:
        try:
            with open(mask_file) as f:
                data = json.load(f)

            # Handle different JSON formats
            masks = data.get('masks', []) if isinstance(data, dict) else data

            for mask in masks:
                total_masks += 1

                cls = mask.get('class', 'unknown')
                conf = mask.get('score', 0.0)

                class_counts[cls] += 1
                confidence_values.append(conf)
                class_confidences[cls].append(conf)

                # Bin by confidence
                if conf < 0.25:
                    confidence_bins['low'] += 1
                elif conf < 0.5:
                    confidence_bins['mid'] += 1
                else:
                    confidence_bins['high'] += 1

        except Exception as e:
            logger.warning(f"Failed to read {mask_file}: {e}")
            continue

    # Calculate statistics
    stats = {
        'stage': 'YOLO_Original_Detections',
        'n_images': n_images,
        'total_masks': total_masks,
        'class_distribution': dict(class_counts),
        'confidence_bins': confidence_bins,
        'avg_confidence': float(np.mean(confidence_values)) if confidence_values else 0.0,
        'std_confidence': float(np.std(confidence_values)) if confidence_values else 0.0,
        'class_confidences': {
            cls: {
                'mean': float(np.mean(confs)),
                'std': float(np.std(confs)),
                'min': float(np.min(confs)),
                'max': float(np.max(confs))
            } for cls, confs in class_confidences.items()
        }
    }

    logger.info(f"  Found {n_images} images with {total_masks} total masks")

    return stats


def analyze_crack_points(points_json: Path) -> Dict:
    """
    Analyze 3D defect points after 2-stage voting.

    Returns:
        Statistics about accepted points, redundancy, filtering
    """
    logger.info("Analyzing 3D defect points...")

    with open(points_json) as f:
        data = json.load(f)

    points = data['points']
    n_points = len(points)

    # Analyze redundancy (multi-view observations)
    redundancies = []
    class_counts = defaultdict(int)
    confidence_values = []
    detection_ratios = []
    class_ratios = []
    n_views_list = []

    # Track source masks
    total_source_masks = 0

    for point in points:
        cls = point.get('class', 'unknown')
        class_counts[cls] += 1

        # Source masks (how many YOLO masks contributed to this point)
        sources = point.get('source_masks', [])
        n_sources = len(sources)
        redundancies.append(n_sources)
        total_source_masks += n_sources

        # Voting ratios
        if 'detection_ratio' in point:
            detection_ratios.append(point['detection_ratio'])
        if 'class_ratio' in point:
            class_ratios.append(point['class_ratio'])

        # Confidence and views
        if 'avg_confidence' in point:
            confidence_values.append(point['avg_confidence'])
        if 'n_views' in point:
            n_views_list.append(point['n_views'])

    # Calculate statistics
    stats = {
        'stage': 'MultiView_Voting',
        'n_points': n_points,
        'total_source_masks': total_source_masks,
        'avg_redundancy': float(np.mean(redundancies)) if redundancies else 0.0,
        'max_redundancy': int(np.max(redundancies)) if redundancies else 0,
        'redundancy_distribution': {
            '1_view': sum(1 for r in redundancies if r == 1),
            '2_3_views': sum(1 for r in redundancies if 2 <= r <= 3),
            '4_5_views': sum(1 for r in redundancies if 4 <= r <= 5),
            '6+_views': sum(1 for r in redundancies if r >= 6)
        },
        'class_distribution': dict(class_counts),
        'avg_confidence': float(np.mean(confidence_values)) if confidence_values else 0.0,
        'avg_detection_ratio': float(np.mean(detection_ratios)) if detection_ratios else 0.0,
        'avg_class_ratio': float(np.mean(class_ratios)) if class_ratios else 0.0,
        'avg_n_views': float(np.mean(n_views_list)) if n_views_list else 0.0
    }

    logger.info(f"  Found {n_points} 3D points from {total_source_masks} source masks")
    logger.info(f"  Average redundancy: {stats['avg_redundancy']:.2f} masks/point")

    return stats


def analyze_clusters(clusters_json: Path, points_json: Path) -> Dict:
    """
    Analyze DBSCAN clustering results.

    Returns:
        Statistics about clusters, noise, consolidation
    """
    logger.info("Analyzing clusters...")

    with open(clusters_json) as f:
        clusters_data = json.load(f)

    with open(points_json) as f:
        points_data = json.load(f)

    clusters = clusters_data['clusters']
    n_clusters = len(clusters)
    n_points_total = len(points_data['points'])

    # Collect clustered points
    clustered_point_ids = set()
    cluster_sizes = []
    class_counts = defaultdict(int)

    for cluster in clusters:
        point_ids = cluster['point_ids']
        cluster_sizes.append(len(point_ids))
        clustered_point_ids.update(point_ids)

        cls = cluster.get('class', 'unknown')
        class_counts[cls] += 1

    n_clustered = len(clustered_point_ids)
    n_noise = n_points_total - n_clustered

    # Cluster size distribution
    size_bins = {
        'tiny_2_5': sum(1 for s in cluster_sizes if 2 <= s <= 5),
        'small_6_10': sum(1 for s in cluster_sizes if 6 <= s <= 10),
        'medium_11_20': sum(1 for s in cluster_sizes if 11 <= s <= 20),
        'large_21_50': sum(1 for s in cluster_sizes if 21 <= s <= 50),
        'huge_51+': sum(1 for s in cluster_sizes if s > 50)
    }

    stats = {
        'stage': 'DBSCAN_Clustering',
        'n_clusters': n_clusters,
        'n_clustered_points': n_clustered,
        'n_noise_points': n_noise,
        'noise_ratio': float(n_noise / n_points_total) if n_points_total > 0 else 0.0,
        'avg_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        'std_cluster_size': float(np.std(cluster_sizes)) if cluster_sizes else 0.0,
        'min_cluster_size': int(np.min(cluster_sizes)) if cluster_sizes else 0,
        'max_cluster_size': int(np.max(cluster_sizes)) if cluster_sizes else 0,
        'size_distribution': size_bins,
        'class_distribution': dict(class_counts)
    }

    logger.info(f"  Found {n_clusters} clusters from {n_points_total} points")
    logger.info(f"  Noise points: {n_noise} ({stats['noise_ratio']*100:.1f}%)")

    return stats


def compute_pipeline_summary(yolo_stats: Dict, points_stats: Dict, cluster_stats: Dict) -> Dict:
    """
    Compute overall pipeline statistics and reduction ratios.
    """
    logger.info("Computing pipeline summary...")

    # Extract key numbers
    n_masks = yolo_stats['total_masks']
    n_points = points_stats['n_points']
    n_clusters = cluster_stats['n_clusters']

    total_source_masks = points_stats['total_source_masks']

    # Redundancy elimination
    redundancy_eliminated = total_source_masks - n_points
    redundancy_ratio = total_source_masks / n_points if n_points > 0 else 0

    # False positive estimation (masks not contributing to any point)
    # This is approximate: masks that never passed voting
    false_positives_estimated = n_masks - total_source_masks

    # Overall reduction
    overall_reduction = n_masks / n_clusters if n_clusters > 0 else 0

    # Point consolidation (points merged into clusters)
    points_consolidated = n_points - n_clusters
    consolidation_ratio = n_points / n_clusters if n_clusters > 0 else 0

    summary = {
        'pipeline_flow': {
            'yolo_masks': n_masks,
            'voting_points': n_points,
            'final_clusters': n_clusters
        },
        'redundancy_elimination': {
            'source_masks_used': total_source_masks,
            'points_created': n_points,
            'eliminated': redundancy_eliminated,
            'avg_redundancy': redundancy_ratio
        },
        'false_positive_removal': {
            'original_masks': n_masks,
            'masks_used': total_source_masks,
            'estimated_false_positives': false_positives_estimated,
            'removal_rate': false_positives_estimated / n_masks if n_masks > 0 else 0
        },
        'point_consolidation': {
            'accepted_points': n_points,
            'final_clusters': n_clusters,
            'consolidated_points': points_consolidated,
            'consolidation_ratio': consolidation_ratio,
            'noise_removed': cluster_stats['n_noise_points']
        },
        'overall_efficiency': {
            'original_masks': n_masks,
            'final_clusters': n_clusters,
            'reduction_ratio': overall_reduction,
            'percentage_reduction': (1 - n_clusters/n_masks) * 100 if n_masks > 0 else 0
        }
    }

    return summary


def generate_paper_tables(stats: Dict, output_dir: Path):
    """
    Generate formatted tables for paper.
    """
    logger.info("Generating paper tables...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: Pipeline Statistics Summary
    table1_path = output_dir / 'table1_pipeline_summary.txt'
    with open(table1_path, 'w') as f:
        f.write("Table 1: Multi-view Pipeline Statistics\n")
        f.write("=" * 80 + "\n\n")

        flow = stats['summary']['pipeline_flow']
        redun = stats['summary']['redundancy_elimination']
        fp = stats['summary']['false_positive_removal']
        consol = stats['summary']['point_consolidation']
        overall = stats['summary']['overall_efficiency']

        f.write("Stage                    | Input    | Output   | Removed  | Ratio\n")
        f.write("-" * 80 + "\n")
        f.write(f"YOLO Detection           | -        | {flow['yolo_masks']:<8} | -        | -\n")
        f.write(f"Multi-view Voting        | {flow['yolo_masks']:<8} | {flow['voting_points']:<8} | {redun['eliminated']:<8} | {redun['avg_redundancy']:.1f}:1\n")
        f.write(f"False Positive Filter    | {redun['source_masks_used']:<8} | {flow['voting_points']:<8} | {fp['estimated_false_positives']:<8} | {fp['removal_rate']*100:.1f}%\n")
        f.write(f"DBSCAN Clustering        | {flow['voting_points']:<8} | {flow['final_clusters']:<8} | {consol['consolidated_points']:<8} | {consol['consolidation_ratio']:.1f}:1\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall                  | {flow['yolo_masks']:<8} | {flow['final_clusters']:<8} | {flow['yolo_masks']-flow['final_clusters']:<8} | {overall['reduction_ratio']:.1f}:1\n")
        f.write("\n")
        f.write(f"Overall reduction: {overall['percentage_reduction']:.1f}%\n")

    logger.info(f"  Saved Table 1: {table1_path}")

    # Table 2: Class Distribution
    table2_path = output_dir / 'table2_class_distribution.txt'
    with open(table2_path, 'w') as f:
        f.write("Table 2: Defect Class Distribution Across Pipeline Stages\n")
        f.write("=" * 80 + "\n\n")

        yolo_classes = stats['yolo']['class_distribution']
        points_classes = stats['points']['class_distribution']
        cluster_classes = stats['clusters']['class_distribution']

        # Get all classes
        all_classes = set(yolo_classes.keys()) | set(points_classes.keys()) | set(cluster_classes.keys())

        f.write("Class                | YOLO Masks | 3D Points  | Clusters   | Retention\n")
        f.write("-" * 80 + "\n")

        for cls in sorted(all_classes):
            yolo_count = yolo_classes.get(cls, 0)
            points_count = points_classes.get(cls, 0)
            cluster_count = cluster_classes.get(cls, 0)
            retention = (cluster_count / yolo_count * 100) if yolo_count > 0 else 0

            f.write(f"{cls:<20} | {yolo_count:<10} | {points_count:<10} | {cluster_count:<10} | {retention:.1f}%\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'TOTAL':<20} | {stats['yolo']['total_masks']:<10} | {stats['points']['n_points']:<10} | {stats['clusters']['n_clusters']:<10} | -\n")

    logger.info(f"  Saved Table 2: {table2_path}")

    # Table 3: Redundancy Analysis
    table3_path = output_dir / 'table3_redundancy_analysis.txt'
    with open(table3_path, 'w') as f:
        f.write("Table 3: Multi-view Redundancy Analysis\n")
        f.write("=" * 80 + "\n\n")

        redun_dist = stats['points']['redundancy_distribution']

        f.write("View Count           | 3D Points  | Percentage | Cumulative\n")
        f.write("-" * 80 + "\n")

        total_points = stats['points']['n_points']
        cumulative = 0

        for key, count in redun_dist.items():
            pct = (count / total_points * 100) if total_points > 0 else 0
            cumulative += pct
            f.write(f"{key:<20} | {count:<10} | {pct:>6.1f}%    | {cumulative:>6.1f}%\n")

        f.write("-" * 80 + "\n")
        f.write(f"Average redundancy: {stats['points']['avg_redundancy']:.2f} masks/point\n")
        f.write(f"Maximum redundancy: {stats['points']['max_redundancy']} masks/point\n")

    logger.info(f"  Saved Table 3: {table3_path}")

    # Table 4: Cluster Size Distribution
    table4_path = output_dir / 'table4_cluster_sizes.txt'
    with open(table4_path, 'w') as f:
        f.write("Table 4: Cluster Size Distribution\n")
        f.write("=" * 80 + "\n\n")

        size_dist = stats['clusters']['size_distribution']

        f.write("Size Range           | Clusters   | Percentage\n")
        f.write("-" * 80 + "\n")

        total_clusters = stats['clusters']['n_clusters']

        for key, count in size_dist.items():
            pct = (count / total_clusters * 100) if total_clusters > 0 else 0
            f.write(f"{key:<20} | {count:<10} | {pct:>6.1f}%\n")

        f.write("-" * 80 + "\n")
        f.write(f"Average size: {stats['clusters']['avg_cluster_size']:.1f} points\n")
        f.write(f"Std deviation: {stats['clusters']['std_cluster_size']:.1f} points\n")
        f.write(f"Range: {stats['clusters']['min_cluster_size']}-{stats['clusters']['max_cluster_size']} points\n")

    logger.info(f"  Saved Table 4: {table4_path}")


def run_analysis(
    yolo_masks_dir: str,
    crack_points_json: str,
    clusters_json: str,
    output_json: str,
    output_tables_dir: str = None
):
    """
    Run complete pipeline statistics analysis.
    """
    logger.info("=" * 80)
    logger.info("Pipeline Statistics Analysis")
    logger.info("=" * 80)

    # Convert to Path
    yolo_masks_path = Path(yolo_masks_dir)
    points_path = Path(crack_points_json)
    clusters_path = Path(clusters_json)
    output_path = Path(output_json)

    # Stage 1: Analyze YOLO masks
    yolo_stats = analyze_yolo_masks(yolo_masks_path)

    # Stage 2: Analyze 3D points
    points_stats = analyze_crack_points(points_path)

    # Stage 3: Analyze clusters
    cluster_stats = analyze_clusters(clusters_path, points_path)

    # Compute summary
    summary = compute_pipeline_summary(yolo_stats, points_stats, cluster_stats)

    # Combine all statistics
    all_stats = {
        'yolo': yolo_stats,
        'points': points_stats,
        'clusters': cluster_stats,
        'summary': summary
    }

    # Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    logger.info(f"\nSaved statistics JSON: {output_path}")

    # Generate paper tables
    if output_tables_dir:
        generate_paper_tables(all_stats, Path(output_tables_dir))

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"YOLO masks:        {yolo_stats['total_masks']}")
    logger.info(f"3D points:         {points_stats['n_points']}")
    logger.info(f"Final clusters:    {cluster_stats['n_clusters']}")
    logger.info(f"Reduction ratio:   {summary['overall_efficiency']['reduction_ratio']:.1f}:1")
    logger.info(f"Redundancy avg:    {points_stats['avg_redundancy']:.2f} masks/point")
    logger.info(f"False positives:   {summary['false_positive_removal']['estimated_false_positives']} ({summary['false_positive_removal']['removal_rate']*100:.1f}%)")
    logger.info(f"Noise removed:     {cluster_stats['n_noise_points']} ({cluster_stats['noise_ratio']*100:.1f}%)")
    logger.info("=" * 80)

    return all_stats


if __name__ == '__main__':
    import argparse
    from src.utils import setup_logging

    parser = argparse.ArgumentParser(
        description='Analyze multi-view pipeline statistics for paper tables'
    )
    parser.add_argument('--yolo-masks', required=True,
                       help='Directory containing YOLO mask JSON files')
    parser.add_argument('--crack-points', required=True,
                       help='Path to crack_points.json')
    parser.add_argument('--clusters', required=True,
                       help='Path to crack_clusters.json')
    parser.add_argument('--output', required=True,
                       help='Output JSON path for statistics')
    parser.add_argument('--output-tables', default=None,
                       help='Output directory for paper tables (optional)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        stats = run_analysis(
            args.yolo_masks,
            args.crack_points,
            args.clusters,
            args.output,
            args.output_tables
        )

        print(f"\n✅ Analysis complete!")
        print(f"   Statistics JSON: {args.output}")
        if args.output_tables:
            print(f"   Paper tables: {args.output_tables}/")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
