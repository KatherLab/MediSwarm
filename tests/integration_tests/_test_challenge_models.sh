#!/bin/bash

##############################################################################
# Integration test for ODELIA challenge models
# Tests preflight_check and local_training for all 5 challenge models
# Can be called as part of the standard test suite
##############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
ODELIA_CUSTOM_DIR="${PROJECT_ROOT}/application/jobs/ODELIA_ternary_classification/app/custom"

MODELS=("1DivideAndConquer" "2BCN_AIM" "3agaldran" "4LME_ABMIL" "5Pimed")
FAILED_TESTS=()

echo "============================================"
echo "ODELIA Challenge Models Integration Tests"
echo "============================================"
echo ""

# Test a single model with given mode
test_model() {
    local model=$1
    local mode=$2
    
    export TRAINING_MODE="$mode"
    export SITE_NAME="TEST_SITE"
    export MODEL_NAME="challenge_$model"
    export MODEL_VARIANT="$model"
    export NUM_EPOCHS="1"
    
    echo -n "Testing $model - $mode ... "
    
    cd "$ODELIA_CUSTOM_DIR"
    
    if timeout 600 python3 main.py > /tmp/test_${model}_${mode}.log 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED_TESTS+=("$model-$mode")
        
        # Print last few lines of error
        echo "  Error details:"
        tail -5 "/tmp/test_${model}_${mode}.log" | sed 's/^/    /'
        return 1
    fi
}

# Run all tests
for model in "${MODELS[@]}"; do
    echo "========== $model =========="
    
    # Test preflight_check
    test_model "$model" "preflight_check" || true
    
    # Test local_training
    test_model "$model" "local_training" || true
    
    echo ""
done

# Summary
echo "=========================================="
if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo -e "${GREEN}All integration tests PASSED${NC}"
    echo "=========================================="
    exit 0
else
    echo -e "${RED}Failed tests:${NC}"
    printf '  - %s\n' "${FAILED_TESTS[@]}"
    echo "=========================================="
    exit 1
fi
