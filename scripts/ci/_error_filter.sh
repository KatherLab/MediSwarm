#!/usr/bin/env bash
# Shared error-detection helper for the integration tests.
#
# The tests used to assert success with a blanket `! grep -qi "error"`, so ANY
# line containing "error" failed the build. NVFlare's audit logger races with
# shutdown and intermittently emits a harmless traceback *after* training has
# already finished:
#
#     root - ERROR - Traceback (most recent call last):
#       File ".../nvflare/fuel/sec/audit.py", line 61, in add_event
#         self.audit_file.write(line + "\n")
#     ValueError: I/O operation on closed file.
#
# That made the same commit pass or fail purely on timing (#434). This narrows
# the filter -- it does not disable it. Any other line containing "error" is
# still fatal.

_benign_error_noise() {
    grep -viE "I/O operation on closed file|fuel/sec/audit\.py|hci/server/(reg|audit)\.py|self\.audit_file\.write|root - ERROR - Traceback"
}

# has_real_error <output>  -> exit 0 if a non-benign "error" line is present
has_real_error() {
    grep -i "error" <<< "$1" | _benign_error_noise | grep -q .
}
