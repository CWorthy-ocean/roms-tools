"""Lightweight timing/memory instrumentation for roms-tools internals.

Vendored from cstar-forge's ``cstar_forge/utils.py`` (kept self-contained here
so roms-tools has no dependency on forge). See ``mem_log`` for usage.
"""

import gc
import logging
import os
import sys
import time
from contextlib import contextmanager

log = logging.getLogger(__name__)

_PLATFORM = sys.platform  # 'linux', 'darwin', 'win32', ...

# --- instantaneous RSS -----------------------------------------------------
try:
    import psutil

    _proc = psutil.Process(os.getpid())

    def rss_mb():
        return _proc.memory_info().rss / 1024**2
except ImportError:
    if _PLATFORM == "linux":

        def rss_mb():
            with open("/proc/self/statm") as f:
                pages = int(f.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / 1024**2
    else:
        import resource

        def rss_mb():
            # no cheap instantaneous RSS without psutil on macOS;
            # fall back to peak as an approximation
            return _peak_rss_mb()


# --- peak RSS ----------------------------------------------------------
if _PLATFORM == "linux":

    def _peak_rss_mb():
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
        return float("nan")

    def _reset_peak():
        # kernel 3.11+; resets VmHWM to current RSS. RHEL 8 (Anvil) is fine.
        try:
            with open("/proc/self/clear_refs", "w") as f:
                f.write("5")
            return True
        except (OSError, PermissionError):
            return False

else:  # macOS (darwin), and a reasonable default elsewhere
    import resource

    def _peak_rss_mb():
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kB. We're in the non-linux branch,
        # so assume bytes on darwin, kB otherwise.
        divisor = 1024**2 if _PLATFORM == "darwin" else 1024
        return maxrss / divisor

    def _reset_peak():
        return False  # no way to reset the high-water mark here


# --- context manager -------------------------------------------------------
@contextmanager
def mem_log(name, level=logging.INFO, collect=True, enabled=True):
    """Time a block and log RSS before/after, plus peak if resettable.

    On Linux (incl. Anvil) reports a true per-block peak.
    On macOS reports process-lifetime peak (cannot be reset).

    If ``enabled`` is False, this is a no-op: no timing, no RSS reads, no
    ``gc.collect()``, no log line. Use this to keep the fast path free of
    instrumentation overhead when verbose diagnostics aren't requested.
    """
    if not enabled:
        yield
        return
    log.debug("%s: starting", name)
    resettable = _reset_peak()
    before = rss_mb()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if collect:
            gc.collect()
        after = rss_mb()
        peak = _peak_rss_mb()
        delta = after - before
        if resettable:
            log.log(
                level,
                "%s: %.2fs, %.0f -> %.0f MB, peak %.0f (delta %+.0f)",
                name,
                elapsed,
                before,
                after,
                peak,
                delta,
            )
        else:
            log.log(
                level,
                "%s: %.2fs, %.0f -> %.0f MB (delta %+.0f); peak %.0f is process-lifetime max",
                name,
                elapsed,
                before,
                after,
                delta,
                peak,
            )
