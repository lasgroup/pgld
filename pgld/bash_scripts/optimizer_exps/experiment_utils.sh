#!/bin/bash
# Utility functions for paper experiments

# Check if an experiment output directory already has results
# Usage: check_experiment_dir <dir_path> [--skip|--fail|--overwrite]
#   --skip: skip this experiment, return 1
#   --fail: exit with error (default)
#   --overwrite: wipe and continue
# Returns: 0 to proceed, 1 to skip
check_experiment_dir() {
    local dir_path="$1"
    local mode="${2:---fail}"

    # Overwrite mode: always delete if exists and proceed
    if [ "$mode" == "--overwrite" ]; then
        if [ -d "$dir_path" ]; then
            echo "OVERWRITE: $dir_path"
            rm -rf "$dir_path"
        fi
        return 0
    fi

    # Directory doesn't exist: proceed
    if [ ! -d "$dir_path" ]; then
        return 0
    fi

    # Check for completed results (.hydra dir or result files)
    local has_results=0
    if [ -d "$dir_path/.hydra" ]; then
        has_results=1
    else
        if find "$dir_path" -type f \( -name "*.yaml" -o -name "*.pt" -o -name "*.json" -o -name "*.csv" \) 2>/dev/null | head -1 | grep -q .; then
            has_results=1
        fi
    fi

    # No results found: proceed
    if [ "$has_results" -eq 0 ]; then
        return 0
    fi

    # Has results: handle based on mode
    case "$mode" in
        --skip)
            echo "SKIP: $dir_path (already has results)"
            return 1
            ;;
        --fail|*)
            echo ""
            echo "ERROR: Output already exists: $dir_path"
            echo "  Delete it or use --skip-existing / --overwrite"
            echo ""
            exit 1
            ;;
    esac
}
