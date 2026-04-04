#!/usr/bin/env bash
# =============================================================================
# MediSwarm GPU Health Check
# =============================================================================
# Verifies that NVIDIA GPUs are healthy and accessible. Intended as:
#   - A pre-training check (run before launching FL client)
#   - A Docker HEALTHCHECK command
#   - A periodic cron/systemd health probe
#
# Exit codes:
#   0 — healthy: driver loaded, GPU(s) responsive, temperature OK
#   1 — warning: driver loaded but temperature high or minor issue
#   2 — critical: driver not responding or no GPUs found
#
# Usage:
#   ./gpu_health_check.sh              # basic check
#   ./gpu_health_check.sh --json       # output JSON (for monitoring integration)
#   ./gpu_health_check.sh --docker     # minimal output for Docker HEALTHCHECK
#   ./gpu_health_check.sh --reset      # attempt nvidia-smi -r if driver is stuck
#
# Environment variables (optional):
#   TEMP_WARNING_C    temperature warning threshold (default: 85)
#   TEMP_CRITICAL_C   temperature critical threshold (default: 95)
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEMP_WARNING_C="${TEMP_WARNING_C:-85}"
TEMP_CRITICAL_C="${TEMP_CRITICAL_C:-95}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log() {
    local level="$1"; shift
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $*"
}

# ---------------------------------------------------------------------------
# Check nvidia-smi responsiveness
# ---------------------------------------------------------------------------
check_driver() {
    if ! command -v nvidia-smi &>/dev/null; then
        log CRITICAL "nvidia-smi not found in PATH"
        return 2
    fi

    # Timeout after 10 seconds — a hung driver will block indefinitely
    if ! timeout 10 nvidia-smi &>/dev/null; then
        log CRITICAL "nvidia-smi did not respond within 10 seconds — driver may be hung"
        return 2
    fi

    return 0
}

# ---------------------------------------------------------------------------
# Query GPU details
# ---------------------------------------------------------------------------
query_gpus() {
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits 2>/dev/null
}

# ---------------------------------------------------------------------------
# Evaluate GPU health
# ---------------------------------------------------------------------------
evaluate_health() {
    local worst_status=0
    local gpu_count=0

    while IFS=', ' read -r idx name temp util mem_used mem_total power; do
        gpu_count=$((gpu_count + 1))

        # Trim whitespace
        temp="${temp// /}"
        util="${util// /}"
        mem_used="${mem_used// /}"
        mem_total="${mem_total// /}"
        power="${power// /}"
        name="${name// /}"

        local status="OK"
        local exit_code=0

        if (( temp >= TEMP_CRITICAL_C )); then
            status="CRITICAL (${temp}°C >= ${TEMP_CRITICAL_C}°C)"
            exit_code=2
        elif (( temp >= TEMP_WARNING_C )); then
            status="WARNING (${temp}°C >= ${TEMP_WARNING_C}°C)"
            exit_code=1
        fi

        if [[ "$OUTPUT_MODE" == "json" ]]; then
            echo "  {\"index\": $idx, \"name\": \"$name\", \"temp_c\": $temp, \"util_pct\": $util, \"mem_used_mib\": $mem_used, \"mem_total_mib\": $mem_total, \"power_w\": $power, \"status\": \"$status\"}"
        elif [[ "$OUTPUT_MODE" != "docker" ]]; then
            log INFO "GPU $idx ($name): ${temp}°C, ${util}% util, ${mem_used}/${mem_total} MiB, ${power}W — $status"
        fi

        if (( exit_code > worst_status )); then
            worst_status=$exit_code
        fi
    done < <(query_gpus)

    if (( gpu_count == 0 )); then
        log CRITICAL "No GPUs found by nvidia-smi"
        return 2
    fi

    if [[ "$OUTPUT_MODE" == "docker" ]]; then
        if (( worst_status == 0 )); then
            echo "healthy ($gpu_count GPU(s))"
        else
            echo "unhealthy"
        fi
    fi

    return $worst_status
}

# ---------------------------------------------------------------------------
# Attempt GPU reset
# ---------------------------------------------------------------------------
attempt_reset() {
    log WARNING "Attempting GPU reset via nvidia-smi -r ..."
    if sudo nvidia-smi -r 2>/dev/null; then
        log INFO "GPU reset command sent successfully. Waiting 10s for driver to recover..."
        sleep 10
        if check_driver; then
            log INFO "Driver recovered after reset"
            return 0
        else
            log CRITICAL "Driver still unresponsive after reset"
            return 2
        fi
    else
        log CRITICAL "GPU reset failed — manual intervention may be required"
        return 2
    fi
}

# ---------------------------------------------------------------------------
# Docker HEALTHCHECK mode (for use in --health-cmd)
# ---------------------------------------------------------------------------
docker_healthcheck() {
    if ! timeout 10 nvidia-smi &>/dev/null; then
        echo "unhealthy"
        exit 1
    fi
    echo "healthy"
    exit 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
OUTPUT_MODE="text"
DO_RESET=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --json)    OUTPUT_MODE="json" ;;
        --docker)  OUTPUT_MODE="docker" ;;
        --reset)   DO_RESET=true ;;
        -h|--help)
            echo "Usage: $0 [--json|--docker|--reset|-h]"
            echo "  --json     JSON output for monitoring systems"
            echo "  --docker   minimal output for Docker HEALTHCHECK"
            echo "  --reset    attempt nvidia-smi -r if driver is unresponsive"
            echo "  -h         show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Fast path for Docker health checks
if [[ "$OUTPUT_MODE" == "docker" ]]; then
    docker_healthcheck
fi

# Full check
if [[ "$OUTPUT_MODE" == "json" ]]; then
    echo "{"
    echo "  \"timestamp\": \"$(date -Iseconds)\","
fi

if ! check_driver; then
    if $DO_RESET; then
        attempt_reset
        exit $?
    fi
    exit 2
fi

if [[ "$OUTPUT_MODE" == "json" ]]; then
    echo "  \"driver\": \"ok\","
    echo "  \"gpus\": ["
fi

evaluate_health
status=$?

if [[ "$OUTPUT_MODE" == "json" ]]; then
    echo "  ],"
    local_status="healthy"
    (( status == 1 )) && local_status="warning"
    (( status == 2 )) && local_status="critical"
    echo "  \"overall\": \"$local_status\""
    echo "}"
fi

exit $status
