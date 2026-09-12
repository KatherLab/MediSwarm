"""Bound DataLoader worker count by what the training set can feed (#574).

Deliberately a module of its own with no imports: env_config pulls in the
dataset package (and through it torch), so anything that wants to be unit-tested
without the training image cannot live there.
"""


def cap_loader_workers(configured: int, num_train_samples, min_samples_per_worker: int = 4) -> int:
    """Never run more DataLoader workers than the training set can keep busy.

    Sixteen workers on a 38-volume training set do nothing for throughput and
    maximise shared-memory churn at every epoch boundary: with batch size 1 and
    the default prefetch, the whole epoch sits in /dev/shm at once, and the
    ``file_system`` sharing strategy's cleanup race then kills the pin-memory
    thread ("unable to open shared memory object", #574). Seen on the 1.8.0 MVP
    deploy test; the fault-tolerant controller retried, but on a two-client run
    with min_clients=2 a retry is a whole round.

    Each worker is guaranteed at least ``min_samples_per_worker`` samples per
    epoch. On a real site (thousands of volumes) the cap never binds.
    ``configured == 0`` (in-process loading) is left alone.
    """
    configured = max(0, int(configured))
    try:
        n = int(num_train_samples)
    except (TypeError, ValueError):
        return configured
    if n <= 0:
        return configured
    return min(configured, max(1, n // max(1, int(min_samples_per_worker))))
