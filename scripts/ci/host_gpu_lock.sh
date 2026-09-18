#!/usr/bin/env bash
# ============================================================================
# host_gpu_lock.sh — host-wide mutex for GPU/Docker-heavy MediSwarm work (#448)
#
# The self-hosted CI runner and our DECADE image builds / deploy tests are the
# SAME machine. They share one GPU, one Docker daemon, and ports 8002/8003.
# #445 made CI *tolerant* of that contention (longer liveness/poll timeouts); it
# did not remove it.
#
# CI already serialises itself through the `mediswarm-self-hosted-gpu-validation`
# concurrency group. The gap #448 names is the other side: image builds and
# deploy tests are started BY HAND and belong to no group, so nothing stops them
# landing on top of a running CI job. "Check the runner is idle first" is exactly
# the discipline that lapses under time pressure -- it lapsed during the 1.6.0
# rollout, which is why this exists.
#
# Two entry points:
#
#   source scripts/ci/host_gpu_lock.sh && acquire_host_lock "what I am"
#       Takes the lock for the REST OF THE CALLING PROCESS (released on exit).
#       Used by the manual build / deploy-test scripts.
#
#   scripts/ci/host_gpu_lock.sh wait [timeout_s]
#       Blocks until the lock is free, then returns WITHOUT holding it. Used as a
#       CI step so a job defers to a manual build already in flight instead of
#       racing it.
#
# Scope, honestly: this removes the "manual run lands on top of CI" collision and
# makes CI yield to an in-flight manual run. It does NOT make the two fully
# mutually exclusive for a whole CI job, because each GitHub Actions step is its
# own process and cannot hold a file descriptor across steps. Closing #448
# properly still wants a dedicated runner (option 1 in the issue).
#
# Escape hatch: MEDISWARM_SKIP_HOST_LOCK=1 (expect flaky CI if you use it).
# ============================================================================

MEDISWARM_HOST_LOCK="${MEDISWARM_HOST_LOCK:-/tmp/mediswarm-host-gpu.lock}"

# Wait up to this long for a peer to finish. A DECADE image build + push is ~10
# min and a 2-node deploy test can run an hour, so the default is generous: we
# would rather queue than interleave and produce a false CI failure.
MEDISWARM_LOCK_WAIT="${MEDISWARM_LOCK_WAIT:-3600}"

_host_lock_available() {
    if [ -n "${MEDISWARM_SKIP_HOST_LOCK:-}" ]; then
        echo "[host-lock] disabled via MEDISWARM_SKIP_HOST_LOCK" >&2
        return 1
    fi
    if ! command -v flock >/dev/null 2>&1; then
        # Non-Linux dev boxes: degrade to a no-op rather than block a build.
        echo "[host-lock] flock not available; continuing unguarded" >&2
        return 1
    fi
    return 0
}

# acquire_host_lock <description> [timeout_s]
# Holds the lock until the calling process exits. Returns non-zero on timeout so
# the caller can abort rather than silently interleave.
acquire_host_lock() {
    local what="${1:-MediSwarm job}"
    local wait_s="${2:-$MEDISWARM_LOCK_WAIT}"

    _host_lock_available || return 0

    # NB: never attach a redirection to this `exec`. `exec` with redirections and no
    # command applies them to the CURRENT SHELL, permanently -- an `exec ... 2>/dev/null`
    # here silently sent the sourcing script's stderr to /dev/null for the rest of its
    # run, swallowing every info()/ok()/warn()/err() in the deploy orchestrators and the
    # build script. Failures then looked like silence. Let a genuine failure print.
    if ! exec {MEDISWARM_LOCK_FD}>"$MEDISWARM_HOST_LOCK"; then
        echo "[host-lock] cannot open $MEDISWARM_HOST_LOCK; continuing unguarded" >&2
        return 0
    fi

    if flock -n "$MEDISWARM_LOCK_FD"; then
        echo "[host-lock] acquired for: $what"
        return 0
    fi

    echo "[host-lock] another GPU/Docker job holds the host; waiting up to ${wait_s}s for: $what" >&2
    if flock -w "$wait_s" "$MEDISWARM_LOCK_FD"; then
        echo "[host-lock] acquired for: $what"
        return 0
    fi

    echo "[host-lock] TIMEOUT after ${wait_s}s. Refusing to run '$what' concurrently with" >&2
    echo "            another GPU/Docker job on this host -- that is the #388 flake (#448)." >&2
    echo "            Override with MEDISWARM_SKIP_HOST_LOCK=1 if you accept the risk." >&2
    return 1
}

# Blocks until the lock is free, then releases immediately. For CI, which cannot
# hold a descriptor across steps but can at least refuse to start on top of a
# manual build that is already running.
wait_for_host_lock() {
    local wait_s="${1:-$MEDISWARM_LOCK_WAIT}"
    _host_lock_available || return 0
    echo "[host-lock] waiting for any in-flight GPU/Docker job (up to ${wait_s}s)..."
    if flock -w "$wait_s" "$MEDISWARM_HOST_LOCK" true; then
        echo "[host-lock] host is free; proceeding"
        return 0
    fi
    echo "[host-lock] still busy after ${wait_s}s; proceeding anyway (CI must not hang forever)" >&2
    return 0
}

# Allow direct invocation: `host_gpu_lock.sh wait [timeout]`
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        wait) wait_for_host_lock "${2:-}" ;;
        *) echo "Usage: $0 wait [timeout_seconds]" >&2; exit 2 ;;
    esac
fi
