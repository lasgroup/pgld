#!/bin/bash
# Master launcher for paper experiments
# Usage: bash launch_all.sh [landscape] [experiment_type] [--skip-existing|--overwrite]
#   landscape: tf_binding | mhc | riboswitch | all (default: all)
#   experiment_type: template | lambda | all (default: all)
#   --skip-existing: skip experiments that already have results
#   --overwrite: wipe existing results without prompting

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HOME}/pgld/.env.local"
source "${SCRIPT_DIR}/../common_basic_setup.sh" 2>/dev/null || source "${HOME}/pgld/.env.local"

LANDSCAPE=${1:-all}
EXPERIMENT=${2:-all}
MODE_FLAG=${3:-}

echo "=========================================="
echo "Paper Experiments Launcher"
echo "Landscape: $LANDSCAPE"
echo "Experiment: $EXPERIMENT"
echo "=========================================="

# Handle mode flag
SWEEP_MODE=""
case $MODE_FLAG in
    --skip-existing)
        export PGLD_SKIP_EXISTING=1
        SWEEP_MODE="--skip-existing"
        echo "Mode: Skip existing experiments"
        ;;
    --overwrite)
        export PGLD_OVERWRITE=1
        SWEEP_MODE="--overwrite"
        echo "Mode: Overwrite existing experiments"
        ;;
    "")
        ;;
    *)
        echo "Unknown flag: $MODE_FLAG"
        echo "Use --skip-existing or --overwrite"
        exit 1
        ;;
esac

# Pre-flight check for existing outputs
preflight_check() {
    local base_dir=$1
    local sweep_name=$2
    local full_path="${base_dir}/${sweep_name}"

    if [ ! -d "$full_path" ]; then
        return 0
    fi

    # Count experiment dirs (those with .hydra subdir)
    local exp_count=$(find "$full_path" -type d -name ".hydra" 2>/dev/null | wc -l)

    if [ "$exp_count" -eq 0 ]; then
        return 0
    fi

    if [ -n "$PGLD_SKIP_EXISTING" ]; then
        echo "  [skip-existing] $exp_count experiments exist in ${sweep_name}, will skip"
        return 0
    fi

    if [ -n "$PGLD_OVERWRITE" ]; then
        echo "  [overwrite] Wiping ${full_path}..."
        rm -rf "$full_path"
        return 0
    fi

    echo ""
    echo "WARNING: ${sweep_name} already has $exp_count completed experiments"
    echo "  Path: $full_path"
    echo ""
    echo "Options: [w]ipe all, [s]kip existing, [a]bort"
    read -p "Choose [w/s/a]: " choice

    case $choice in
        w|W)
            echo "Wiping..."
            rm -rf "$full_path"
            return 0
            ;;
        s|S)
            export PGLD_SKIP_EXISTING=1
            echo "Will skip existing experiments"
            return 0
            ;;
        *)
            echo "Aborted"
            return 1
            ;;
    esac
}

launch_tf_binding() {
    local exp_type=$1
    local base="${PGLD_OUTPUT_DIR}/optimizer_exps/tf_binding"
    echo "Launching TF-binding experiments ($exp_type)..."

    if [ "$exp_type" == "template" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "template_sweep" || return 1
        echo "  -> Template sweep"
        sbatch ${SCRIPT_DIR}/tf_binding_sweep.sh template $SWEEP_MODE
    fi

    if [ "$exp_type" == "lambda" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "lambda_sweep" || return 1
        echo "  -> Lambda sweep"
        sbatch ${SCRIPT_DIR}/tf_binding_sweep.sh lambda $SWEEP_MODE
    fi

    if [ "$exp_type" == "arbitrary" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "arbitrary" || return 1
        echo "  -> Arbitrary policy"
        sbatch ${SCRIPT_DIR}/tf_binding_sweep.sh arbitrary $SWEEP_MODE
    fi
}

launch_mhc() {
    local exp_type=$1
    local base="${PGLD_OUTPUT_DIR}/optimizer_exps/mhc_flurry"
    echo "Launching MHC experiments ($exp_type)..."

    if [ "$exp_type" == "template" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "template_sweep" || return 1
        echo "  -> Template sweep"
        sbatch ${SCRIPT_DIR}/mhc_sweep.sh template $SWEEP_MODE
    fi

    if [ "$exp_type" == "lambda" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "lambda_sweep" || return 1
        echo "  -> Lambda sweep"
        sbatch ${SCRIPT_DIR}/mhc_sweep.sh lambda $SWEEP_MODE
    fi

    if [ "$exp_type" == "arbitrary" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "arbitrary" || return 1
        echo "  -> Arbitrary policy"
        sbatch ${SCRIPT_DIR}/mhc_sweep.sh arbitrary $SWEEP_MODE
    fi
}

launch_riboswitch() {
    local exp_type=$1
    local base="${PGLD_OUTPUT_DIR}/optimizer_exps/riboswitch"
    echo "Launching Riboswitch experiments ($exp_type)..."

    if [ "$exp_type" == "template" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "template_sweep" || return 1
        echo "  -> Template sweep"
        sbatch ${SCRIPT_DIR}/riboswitch_sweep.sh template $SWEEP_MODE
    fi

    if [ "$exp_type" == "lambda" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "lambda_sweep" || return 1
        echo "  -> Lambda sweep"
        sbatch ${SCRIPT_DIR}/riboswitch_sweep.sh lambda $SWEEP_MODE
    fi

    if [ "$exp_type" == "arbitrary" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "arbitrary" || return 1
        echo "  -> Arbitrary policy"
        sbatch ${SCRIPT_DIR}/riboswitch_sweep.sh arbitrary $SWEEP_MODE
    fi
}

launch_gb1() {
    local exp_type=$1
    local base="${PGLD_OUTPUT_DIR}/optimizer_exps/gb1"
    echo "Launching GB1 experiments ($exp_type)..."

    if [ "$exp_type" == "template" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "template_sweep" || return 1
        echo "  -> Template sweep"
        sbatch ${SCRIPT_DIR}/gb1_sweep.sh template $SWEEP_MODE
    fi

    if [ "$exp_type" == "lambda" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "lambda_sweep" || return 1
        echo "  -> Lambda sweep"
        sbatch ${SCRIPT_DIR}/gb1_sweep.sh lambda $SWEEP_MODE
    fi

    if [ "$exp_type" == "arbitrary" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "arbitrary" || return 1
        echo "  -> Arbitrary policy"
        sbatch ${SCRIPT_DIR}/gb1_sweep.sh arbitrary $SWEEP_MODE
    fi
}

launch_trpb() {
    local exp_type=$1
    local base="${PGLD_OUTPUT_DIR}/optimizer_exps/trpb"
    echo "Launching TrpB experiments ($exp_type)..."

    if [ "$exp_type" == "template" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "template_sweep" || return 1
        echo "  -> Template sweep"
        sbatch ${SCRIPT_DIR}/trpb_sweep.sh template $SWEEP_MODE
    fi

    if [ "$exp_type" == "lambda" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "lambda_sweep" || return 1
        echo "  -> Lambda sweep"
        sbatch ${SCRIPT_DIR}/trpb_sweep.sh lambda $SWEEP_MODE
    fi

    if [ "$exp_type" == "arbitrary" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "arbitrary" || return 1
        echo "  -> Arbitrary policy"
        sbatch ${SCRIPT_DIR}/trpb_sweep.sh arbitrary $SWEEP_MODE
    fi
}

launch_phoq() {
    local exp_type=$1
    local base="${PGLD_OUTPUT_DIR}/optimizer_exps/phoq"
    echo "Launching PhoQ experiments ($exp_type)..."

    if [ "$exp_type" == "template" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "template_sweep" || return 1
        echo "  -> Template sweep"
        sbatch ${SCRIPT_DIR}/phoq_sweep.sh template $SWEEP_MODE
    fi

    if [ "$exp_type" == "lambda" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "lambda_sweep" || return 1
        echo "  -> Lambda sweep"
        sbatch ${SCRIPT_DIR}/phoq_sweep.sh lambda $SWEEP_MODE
    fi

    if [ "$exp_type" == "arbitrary" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "arbitrary" || return 1
        echo "  -> Arbitrary policy"
        sbatch ${SCRIPT_DIR}/phoq_sweep.sh arbitrary $SWEEP_MODE
    fi
}

launch_dhfr() {
    local exp_type=$1
    local base="${PGLD_OUTPUT_DIR}/optimizer_exps/dhfr"
    echo "Launching DHFR experiments ($exp_type)..."

    if [ "$exp_type" == "template" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "template_sweep" || return 1
        echo "  -> Template sweep"
        sbatch ${SCRIPT_DIR}/dhfr_sweep.sh template $SWEEP_MODE
    fi

    if [ "$exp_type" == "lambda" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "lambda_sweep" || return 1
        echo "  -> Lambda sweep"
        sbatch ${SCRIPT_DIR}/dhfr_sweep.sh lambda $SWEEP_MODE
    fi

    if [ "$exp_type" == "arbitrary" ] || [ "$exp_type" == "all" ]; then
        preflight_check "$base" "arbitrary" || return 1
        echo "  -> Arbitrary policy"
        sbatch ${SCRIPT_DIR}/dhfr_sweep.sh arbitrary $SWEEP_MODE
    fi
}

# Launch based on arguments
case $LANDSCAPE in
    tf_binding)
        launch_tf_binding $EXPERIMENT
        ;;
    mhc)
        launch_mhc $EXPERIMENT
        ;;
    riboswitch)
        launch_riboswitch $EXPERIMENT
        ;;
    gb1)
        launch_gb1 $EXPERIMENT
        ;;
    trpb)
        launch_trpb $EXPERIMENT
        ;;
    phoq)
        launch_phoq $EXPERIMENT
        ;;
    dhfr)
        launch_dhfr $EXPERIMENT
        ;;
    all)
        launch_tf_binding $EXPERIMENT
        sleep 1
        launch_mhc $EXPERIMENT
        sleep 1
        launch_riboswitch $EXPERIMENT
        sleep 1
        launch_gb1 $EXPERIMENT
        sleep 1
        launch_trpb $EXPERIMENT
        sleep 1
        launch_phoq $EXPERIMENT
        sleep 1
        launch_dhfr $EXPERIMENT
        ;;
    *)
        echo "Unknown landscape: $LANDSCAPE"
        echo "Usage: bash launch_all.sh [tf_binding|mhc|riboswitch|gb1|trpb|phoq|dhfr|all] [template|lambda|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Jobs submitted! Monitor with: squeue -u \$USER"
echo "=========================================="
echo ""
echo "After experiments complete, generate plots with:"
echo "  python ${PGLD_PKG_ROOT}/scripts/aggregate_paper_results.py"
echo ""
