#!/bin/bash
# 3D Crack Detection Pipeline Runner
# Usage: ./run.sh [1-8|all]
#
# Environment Variables:
#   USE_CROP=1    Enable crop-based multi-scale inference (Phase 2)
#
# Examples:
#   ./run.sh 2              # Standard YOLO inference
#   USE_CROP=1 ./run.sh 2   # Crop-based inference

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_phase() {
    echo ""
    echo "=========================================="
    echo -e "${YELLOW}$1${NC}"
    echo "=========================================="
}

# Function to run a phase
run_phase() {
    local phase_num=$1
    local phase_name=$2
    local command=$3

    print_phase "[$phase_num/8] $phase_name"
    print_info "Command: $command"
    echo ""

    local start_time=$(date +%s)

    if eval "$command"; then
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo ""
        print_success "Complete (${elapsed}s)"
        return 0
    else
        echo ""
        print_error "Failed: $phase_name"
        return 1
    fi
}

# Phase definitions
phase1() {
    run_phase 1 "Phase 0: SFM (Structure from Motion)" \
        "python -m src.pipeline sfm --config configs/simple.yaml"
}

phase2() {
    # Build command with optional crop flag
    local cmd="python -m src.pipeline infer --config configs/simple.yaml"

    if [ "${USE_CROP}" = "1" ]; then
        cmd="$cmd --use-crop"
        print_info "🔬 Crop-based multi-scale inference ENABLED"
    else
        print_info "📷 Standard inference mode"
    fi

    run_phase 2 "Phase 1: YOLO Inference" "$cmd"
}

phase3() {
    run_phase 3 "Phase 2: Depth-to-Color Alignment + Scale Maps" \
        "python -m src.d2c_and_pixel_scale \
            --depth-dir data/depth \
            --rgb-calib calib/rgb_camera_info.json \
            --depth-calib calib/depth_camera_info.json \
            --extrinsic calib/extrinsic_depth_to_color.json \
            --output-dir outputs/d2c_pixel_scale"
}

phase4() {
    run_phase 4 "Phase 3: Point Cloud Overlay (2-Stage Voting)" \
        "python -m src.point_cloud_overlay \
            --config configs/simple.yaml \
            --sparse-dir data/sfm/sparse/0 \
            --masks-dir data/yolo_masks \
            --output outputs/sfm_masked_cloud.ply \
            --output-json outputs/crack_points.json \
            --crack-color 255 0 0 \
            --min-track-length 2 \
            --vote-threshold 0.5 \
            --class-vote-threshold 0.5 \
            --min-confidence 0.25 \
            --log-level INFO"
}

phase5() {
    run_phase 5 "Phase 4: Upsampling" \
        "python -m src.upsample_crack_points \
            --input outputs/crack_points.json \
            --output outputs/crack_points_upsampled.json \
            --method density \
            --k-neighbors 5 \
            --max-distance 0.1 \
            --min-spacing 0.005"
}

phase6() {
    run_phase 6 "Phase 5: DBSCAN Clustering" \
        "python -m src.cluster_crack_points_dbscan \
            --input outputs/crack_points_upsampled.json \
            --output outputs/crack_clusters.json \
            --output-ply outputs/clustered_cracks.ply \
            --eps 0.15 \
            --min-samples 5 \
            --split-angle 90 \
            --merge-distance 0.2 \
            --merge-angle 30 \
            --log-level DEBUG"
}

phase7() {
    run_phase 7 "Phase 6: Crack Measurement" \
        "python -m src.measure_clusters \
            --clusters outputs/crack_clusters.json \
            --crack-points outputs/crack_points.json \
            --masks-dir data/yolo_masks \
            --scale-maps-dir outputs/d2c_pixel_scale \
            --rgb-dir data/rgb \
            --output outputs/cluster_measurements.json \
            --image-width 3840 \
            --image-height 2160 \
            --n-segments 100 \
            --detection-method gradient \
            --gradient-percentile 80 \
            --min-component-ratio 0.3 \
            --max-width-filter 1.0 \
            --sample-interval 5 \
            --log-level INFO \
            --viz-dir outputs/visualizations_measurements"
}

phase8() {
    run_phase 8 "Phase 7: Inspection Report Generation" \
        "python src/generate_inspection_report.py \
            --ply outputs/clustered_cracks.ply \
            --crack-points outputs/crack_points.json \
            --measurements outputs/cluster_measurements.json \
            --flip-y"
}

# Run all phases
run_all() {
    local pipeline_start=$(date +%s)

    echo "============================================================================"
    echo "🚀 3D Crack Detection Pipeline - Full Run"
    echo "============================================================================"

    phase1 || exit 1
    phase2 || exit 1
    phase3 || exit 1
    phase4 || exit 1
    phase5 || exit 1
    phase6 || exit 1
    phase7 || exit 1
    phase8 || exit 1

    local total_time=$(($(date +%s) - pipeline_start))
    local minutes=$((total_time / 60))
    local seconds=$((total_time % 60))

    echo ""
    echo "============================================================================"
    print_success "Pipeline Complete!"
    echo "============================================================================"
    echo "Total time: ${minutes}m ${seconds}s"
    echo ""
    echo "📊 Output files:"
    echo "   - outputs/cluster_measurements.json"
    echo "   - outputs/inspection_report.png"
    echo "   - outputs/inspection_diagram.png"
    echo "   - outputs/measurement_table.csv"
    echo "   - outputs/clustered_cracks.ply"
    echo "   - outputs/visualizations_measurements/"
    echo ""
}

# Show usage
show_usage() {
    echo "============================================================================"
    echo "3D Crack Detection Pipeline Runner"
    echo "============================================================================"
    echo ""
    echo "Usage: ./run.sh [PHASE|all]"
    echo ""
    echo "Phases:"
    echo "  1     Phase 0: SFM (Structure from Motion)"
    echo "  2     Phase 1: YOLO Inference"
    echo "  3     Phase 2: Depth-to-Color Alignment + Scale Maps"
    echo "  4     Phase 3: Point Cloud Overlay"
    echo "  5     Phase 4: Upsampling"
    echo "  6     Phase 5: DBSCAN Clustering"
    echo "  7     Phase 6: Crack Measurement"
    echo "  8     Phase 7: Inspection Report Generation"
    echo "  all   Run all phases sequentially"
    echo ""
    echo "Environment Variables:"
    echo "  USE_CROP=1    Enable crop-based multi-scale inference (Phase 2)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh 1                # Run only Phase 0 (SFM)"
    echo "  ./run.sh 2                # Run Phase 1 (standard YOLO)"
    echo "  USE_CROP=1 ./run.sh 2     # Run Phase 1 (crop-based YOLO)"
    echo "  ./run.sh 5                # Run only Phase 4 (Upsampling)"
    echo "  ./run.sh all              # Run complete pipeline"
    echo "  USE_CROP=1 ./run.sh all   # Full pipeline with crop mode"
    echo ""
}

# Main logic
case "$1" in
    1)
        phase1
        ;;
    2)
        phase2
        ;;
    3)
        phase3
        ;;
    4)
        phase4
        ;;
    5)
        phase5
        ;;
    6)
        phase6
        ;;
    7)
        phase7
        ;;
    8)
        phase8
        ;;
    all)
        run_all
        ;;
    *)
        show_usage
        exit 0
        ;;
esac
