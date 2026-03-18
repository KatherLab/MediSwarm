#!/bin/bash

##############################################################################
# Complete Testing & Build Automation for Challenge Models
# 
# This script automates the full workflow:
# 1. Tests model loading and training for all 5 challenge models
# 2. Generates startup kits (config is now static with dynamic model factory)
# 3. Runs preflight checks and local training tests
#
# The NVFlare config now uses a dynamic model factory that reads MODEL_VARIANT
# from environment variables, eliminating the need to update configs per model.
#
# Usage: ./test_and_build_all_models.sh [--no-push] [--models "model1,model2"] [--skip-build]
##############################################################################

set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
python_env="/home/swarm/Documents/ODELIA/.venv"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ODELIA_APP_DIR="${PROJECT_ROOT}/application/jobs/ODELIA_ternary_classification/app"
PROVISION_FILE="${PROJECT_ROOT}/application/provision/project_Challenge_test.yml"
DOCKER_DIR="${PROJECT_ROOT}/workspace/odelia_challenge_model_test/prod_00/UKA_1/startup/"

CONFIG_PATH="${ODELIA_APP_DIR}/config/config_fed_client.conf"
CHALLENGE_CONFIG_PATH="${ODELIA_APP_DIR}/custom/models/challenge/challenge_models_config.sh"

TEST_SCRIPT="${PROJECT_ROOT}/tests/unit_tests/test_challenge_models.py"
UPDATE_CONFIG_SCRIPT="${PROJECT_ROOT}/scripts/update_config_fed_client.py"

source "$python_env/bin/activate"

# Parse arguments
NO_PUSH=false
SKIP_BUILD=false
MODELS_TO_TEST="1DivideAndConquer"  #,2BCN_AIM,3agaldran,4LME_ABMIL,5Pimed"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-push)
            NO_PUSH=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --models)
            MODELS_TO_TEST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine which models to test
if [ -z "$MODELS_TO_TEST" ]; then
    MODELS=("${CHALLENGE_MODELS[@]}")
else
    IFS=',' read -ra MODELS <<< "$MODELS_TO_TEST"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Challenge Models Testing & Build Automation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Project Root:     $PROJECT_ROOT"
echo "ODELIA App:       $ODELIA_APP_DIR"
echo "Models to test:   ${MODELS[*]}"
echo "Push changes:     $([ "$NO_PUSH" = true ] && echo 'NO' || echo 'YES')"
echo "Build startup:    $([ "$SKIP_BUILD" = true ] && echo 'SKIP' || echo 'YES')"
echo ""

# Function to print section header
print_section() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" -eq 0 ]; then
        echo -e "${GREEN}✓ $message${NC}"
    else
        echo -e "${RED}✗ $message${NC}"
    fi
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to run test for a model
run_model_test() {
    local model=$1 # model name (e.g. "3agaldran")
    local mode=$2  # preflight_check or local_training
    
    echo -e "${YELLOW}Testing $model - $mode mode...${NC}"
    
    # Set environment variables
    export SITE_NAME=UKA_1
    export DATA_DIR=/home/swarm/Documents/ODELIA/LocalSL/ChallengeData/
    export SCRATCH_DIR="/home/swarm/Documents/MediSwarmChallenge/MediSwarm/tests/results/${mode}_uka_${model}"
    mkdir -p $SCRATCH_DIR

    export TRAINING_MODE="$mode"
    export MODEL_NAME="challenge_$model"
    export MODEL_VARIANT="challenge_$model"
    export NUM_EPOCHS="1"
    echo "Environment: TRAINING_MODE=$TRAINING_MODE, SITE_NAME=$SITE_NAME, MODEL_NAME=$MODEL_NAME, MODEL_VARIANT=$MODEL_VARIANT, NUM_EPOCHS=$NUM_EPOCHS"
    
    cd "${ODELIA_APP_DIR}/custom"

    # Run with timeout
    timeout 600 $python_env/bin/python3 main.py > "/tmp/test_${model}_${mode}.log" 2>&1
    
    cd $DOCKER_DIR
    docker rm -f "odelia_swarm_client_${SITE_NAME}_$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)" 2>/dev/null || true  # remove any existing container before next test
    if [ "$mode" = "local_training" ]; then
        ./docker.sh --scratch_dir $SCRATCH_DIR --GPU device=0 --dummy_training 2>&1 | tee $SCRATCH_DIR/dummy_training_console_output.txt
    fi
    if [ "$mode" = "preflight_check" ]; then
        ./docker.sh --data_dir $DATA_DIR --scratch_dir $SCRATCH_DIR --GPU device=0 --preflight_check  2>&1 | tee $SCRATCH_DIR/preflight_check_console_output.txt
    fi

    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_status 0 "$model - $mode"
        return 0
    elif [ $exit_code -eq 124 ]; then
        print_status 1 "$model - $mode (TIMEOUT)"
        return 1
    else
        print_status 1 "$model - $mode"
        echo "  Last 20 lines of log:"
        tail -20 "/tmp/test_${model}_${mode}.log" | sed 's/^/    /'
        return 1
    fi
}

# Function to commit and push changes
commit_and_push() {
    # No config changes to commit anymore
    print_status 0 "Git commit skipped (no config changes)"
    return 0
}

# Function to build startup kits
build_startup_kits() {
    print_section "Building Startup Kits"
    
    cd "$PROJECT_ROOT"
    echo "Current directory: $PROJECT_ROOT"
    
    # Check if build script exists
    if [ ! -f "./buildDockerImageAndStartupKits2.sh" ]; then
        print_warning "Build script not found"
        return 1
    fi
    
    echo -e "${YELLOW}Running build script...${NC}"
    
    timeout 1800 ./buildDockerImageAndStartupKits2.sh -p  $PROVISION_FILE > /tmp/build.log 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_status 0 "Startup kits built successfully"
        return 0
    elif [ $exit_code -eq 124 ]; then
        print_warning "Build script timed out (>30 min)"
        return 1
    else
        print_status 1 "Build script failed"
        echo "  Last 20 lines of build log:"
        tail -20 /tmp/build.log | sed 's/^/    /'
        return 1
    fi
}

# Main workflow
main() {
    local failed_models=()
    local test_results=()
    
    # Section 1: Test model loading
    print_section "Phase 1: Testing Model Loading"
    
    for model in "${MODELS[@]}"; do
        echo ""
        echo -e "${BLUE}Testing model: $model${NC}"
        
        all_passed=true
        
        # Test preflight_check
        if ! run_model_test "$model" "preflight_check"; then
            all_passed=false
        fi
        
        # Test local_training
        if ! run_model_test "$model" "local_training"; then
            all_passed=false
        fi
        
        if [ "$all_passed" = true ]; then
            test_results+=("$model:PASS")
            echo -e "${GREEN}✓ All tests passed for $model${NC}"
        else
            test_results+=("$model:FAIL")
            failed_models+=("$model")
            echo -e "${RED}✗ Some tests failed for $model${NC}"
        fi
    done
    
    # Section 2: Build startup kits 
    if [ "$SKIP_BUILD" = false ]; then
        print_section "Phase 2: Building Startup Kits"
        build_startup_kits
    else
        print_section "Phase 2: Build Skipped (--skip-build)"
        echo -e "${YELLOW}Startup kit build skipped as requested${NC}"
    fi
    
    # Final summary
    print_section "Test Summary"
    
    for result in "${test_results[@]}"; do
        IFS=':' read -r model status <<< "$result"
        if [ "$status" = "PASS" ]; then
            echo -e "${GREEN}✓ $model: PASSED${NC}"
        else
            echo -e "${RED}✗ $model: FAILED${NC}"
        fi
    done
    
    if [ ${#failed_models[@]} -eq 0 ]; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}ALL TESTS PASSED ✓${NC}"
        echo -e "${GREEN}========================================${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}FAILED MODELS: ${failed_models[*]}${NC}"
        echo -e "${RED}========================================${NC}"
        return 1
    fi
}

# Run main
main
exit_code=$?

echo ""
echo -e "${BLUE}Logs available at:${NC}"
echo "  /tmp/test_*_{preflight_check,local_training}.log"
echo "  /tmp/build.log"
echo ""

exit $exit_code
