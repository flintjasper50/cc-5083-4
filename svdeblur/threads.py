"""Locked numerical thread configuration for the release package.

The restoration is deterministic only if BLAS, OpenMP and FFT libraries use a
fixed number of workers.  Environment variables are exported here *before*
numpy/scipy are imported (``svdeblur/__init__.py`` calls :func:`lock_threads`
first), and the start scripts export the same values in the shell so a
subprocess inherits them too.

This module intentionally imports only the standard library so it is safe to
import prior to any numerical package.
"""

from __future__ import annotations

import os

RELEASE_THREADS = 1

THREAD_ENV = {
    "OMP_NUM_THREADS": str(RELEASE_THREADS),
    "OPENBLAS_NUM_THREADS": str(RELEASE_THREADS),
    "MKL_NUM_THREADS": str(RELEASE_THREADS),
    "NUMEXPR_NUM_THREADS": str(RELEASE_THREADS),
    "VECLIB_MAXIMUM_THREADS": str(RELEASE_THREADS),
    "BLIS_NUM_THREADS": str(RELEASE_THREADS),
}


def lock_threads() -> dict[str, str]:
    """Force every known threading backend to one worker.

    The assignment is forced rather than defaulted: the release promises
    reproducible numbers, so a caller cannot silently run with another
    setting.  Return the applied environment.
    """
    for key, value in THREAD_ENV.items():
        os.environ[key] = value
    return dict(THREAD_ENV)


def thread_config() -> dict[str, object]:
    """Observable record of the thread policy used for a run."""
    return {
        "policy": "one worker per BLAS/OpenMP/FFT backend",
        "release_threads": RELEASE_THREADS,
        "env": {key: os.environ.get(key) for key in THREAD_ENV},
        "fft": "scipy.signal.fftconvolve (pocketfft, workers=1)",
    }


def verify_thread_config() -> tuple[bool, list[str]]:
    """Check the environment actually carries the locked values."""
    problems = []
    for key, value in THREAD_ENV.items():
        if os.environ.get(key) != value:
            problems.append(f"{key}={os.environ.get(key)!r}, expected {value!r}")
    return (not problems), problems
