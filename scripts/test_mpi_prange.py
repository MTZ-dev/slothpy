#!/usr/bin/env python3
"""
test_numba_mpi_prange.py

Experiments for numba-mpi inside/outside Numba prange loops.

Run for normal tests (use ``--bind-to none`` so each rank sees all node CPUs;
SlothPy session jobs add this automatically via :mod:`slothpy.core.mpi_launch`):

    mpiexec -n 2 --bind-to none python scripts/test_mpi_prange.py --threads 4 --n 32

Numba + MPI threading:

    Numba caps ``set_num_threads(n)`` at ``NUMBA_NUM_THREADS``, which defaults to
    ``len(os.sched_getaffinity(0))`` (CPUs this rank may use). If the launcher
    binds each rank to a small CPU set, use ``--bind-to none`` (see above) or set
    ``NUMBA_NUM_THREADS`` to match the affinity mask.

Run also the dangerous concurrent-MPI-in-prange test:

    mpiexec -n 2 --bind-to none python scripts/test_mpi_prange.py --threads 4 --n 32 --dangerous

Run a deliberately nondeterministic same-tag test:

    mpiexec -n 2 --bind-to none python scripts/test_mpi_prange.py --threads 4 --n 32 --dangerous --chaos

Recommended:
    Use exactly 2 MPI ranks for the communication tests.

Notes:
    - This assumes the common numba-mpi API:
          import numba_mpi as mpi
          mpi.rank()
          mpi.size()
          mpi.send(array, dest=..., tag=...)
          mpi.recv(array, source=..., tag=...)
          mpi.barrier()

    - If your installed numba-mpi version uses slightly different argument names,
      adjust the calls in the small JIT functions below.

    - The script requests MPI_THREAD_MULTIPLE through mpi4py before importing
      numba_mpi. If the provided level is lower, the script still runs the safe
      tests, but refuses the dangerous concurrent test unless you force it.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# Ask MPI for THREAD_MULTIPLE before importing numba_mpi.
# ---------------------------------------------------------------------------
import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

from mpi4py import MPI

if not MPI.Is_initialized():
    provided = MPI.Init_thread(required=MPI.THREAD_MULTIPLE)
else:
    provided = MPI.Query_thread()

THREAD_LEVEL_NAMES = {
    MPI.THREAD_SINGLE: "MPI_THREAD_SINGLE",
    MPI.THREAD_FUNNELED: "MPI_THREAD_FUNNELED",
    MPI.THREAD_SERIALIZED: "MPI_THREAD_SERIALIZED",
    MPI.THREAD_MULTIPLE: "MPI_THREAD_MULTIPLE",
}

comm = MPI.COMM_WORLD
py_rank = comm.Get_rank()
py_size = comm.Get_size()


_CLI_DEFAULT_THREADS = 4


def _threads_from_argv(argv: list[str]) -> int | None:
    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])

        if arg.startswith("--threads="):
            return int(arg.partition("=")[2])

        i += 1

    return None


# Import after MPI has been explicitly initialized.
import numba as nb
import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads

try:
    import numba_mpi as mpi
except Exception as exc:
    if py_rank == 0:
        print("Could not import numba_mpi.")
        print("Install it first, for example with your usual environment manager.")
        print(f"Original error: {exc!r}")
    MPI.Finalize()
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Pure Numba tests: prange thread mapping and memory races.
# ---------------------------------------------------------------------------


@njit(parallel=True)
def prange_thread_map(n: int) -> np.ndarray:
    """
    Return which Numba worker thread executed each prange iteration.
    """
    out = np.empty(n, dtype=np.int64)

    print(f"Rank {mpi.rank()}: Number of threads: {get_num_threads()}")

    for i in prange(n):
        out[i] = get_thread_id()
    return out


@njit(parallel=True)
def unsafe_shared_counter(n: int) -> int:
    """
    Deliberately wrong.

    Multiple prange iterations update the same array element.
    This is a race and often returns less than n.
    """
    x = np.zeros(1, dtype=np.int64)

    for _ in prange(n):
        x[0] += 1

    return int(x[0])


@njit(parallel=True)
def safe_per_thread_counter(n: int) -> np.ndarray:
    """
    Safe version: each Numba thread writes only to its own slot.
    """
    nthreads = get_num_threads()
    partial = np.zeros(nthreads, dtype=np.int64)

    for _ in prange(n):
        tid = get_thread_id()
        partial[tid] += 1

    return partial


# ---------------------------------------------------------------------------
# numba-mpi tests.
# ---------------------------------------------------------------------------


@njit
def mpi_pingpong_outside_prange() -> int:
    """
    Safe baseline: one MPI call sequence, no prange.

    rank 0 sends 123 to rank 1.
    rank 1 receives it, adds 1, sends it back.
    rank 0 receives 124.
    """
    rank = mpi.rank()
    size = mpi.size()

    if size < 2:
        return -999

    buf = np.empty(1, dtype=np.int64)

    if rank == 0:
        buf[0] = 123
        mpi.send(buf, dest=1, tag=10)
        mpi.recv(buf, source=1, tag=11)
        return int(buf[0])

    elif rank == 1:
        mpi.recv(buf, source=0, tag=10)
        buf[0] += 1
        mpi.send(buf, dest=0, tag=11)
        return int(buf[0])

    else:
        return -1


@njit(parallel=True)
def mpi_single_call_site_inside_prange(n: int) -> tuple[int, np.ndarray]:
    """
    MPI communication from exactly one prange iteration.

    This tests which Numba thread executed the MPI-calling iteration.
    It is not concurrent MPI, but it may still not be strictly FUNNELED
    if the MPI-calling iteration is not executed by the main thread.
    """
    rank = mpi.rank()
    size = mpi.size()

    thread_seen = np.full(n, -1, dtype=np.int64)
    result = np.zeros(1, dtype=np.int64)
    mpi_calling_thread = np.full(1, -1, dtype=np.int64)

    for i in prange(n):
        tid = get_thread_id()
        thread_seen[i] = tid

        if i == 0 and size >= 2:
            mpi_calling_thread[0] = tid

            buf = np.empty(1, dtype=np.int64)

            if rank == 0:
                buf[0] = 2000 + tid
                mpi.send(buf, dest=1, tag=20)
                mpi.recv(buf, source=1, tag=21)
                result[0] = buf[0]

            elif rank == 1:
                mpi.recv(buf, source=0, tag=20)
                buf[0] += 1
                mpi.send(buf, dest=0, tag=21)
                result[0] = buf[0]

    return int(result[0]), thread_seen


@njit(parallel=True)
def mpi_concurrent_distinct_tags_in_prange(n: int) -> np.ndarray:
    """
    Potentially dangerous.

    Every prange iteration performs an MPI call.

    rank 0:
        sends one small message per iteration to rank 1, using tag 1000 + i.

    rank 1:
        receives one message per iteration from rank 0, using tag 1000 + i.

    This requires real concurrent MPI support if several Numba threads enter
    MPI at the same time.
    """
    rank = mpi.rank()
    size = mpi.size()

    received_sender_threads = np.full(n, -1, dtype=np.int64)

    if size < 2:
        return received_sender_threads

    for i in prange(n):
        tid = get_thread_id()
        buf = np.empty(2, dtype=np.int64)

        if rank == 0:
            buf[0] = i
            buf[1] = tid
            mpi.send(buf, dest=1, tag=1000 + i)

        elif rank == 1:
            mpi.recv(buf, source=0, tag=1000 + i)
            received_sender_threads[i] = buf[1]

    return received_sender_threads


@njit(parallel=True)
def mpi_concurrent_same_tag_chaos(n: int) -> np.ndarray:
    """
    Very deliberately nondeterministic test.

    All prange iterations use the same tag. Messages from different threads
    may arrive in an order that is not the same as iteration order.

    This is useful to show that "it ran" does not mean "ordering is what I
    mentally expected".
    """
    rank = mpi.rank()
    size = mpi.size()

    values = np.full(n, -1, dtype=np.int64)

    if size < 2:
        return values

    for i in prange(n):
        tid = get_thread_id()
        buf = np.empty(2, dtype=np.int64)

        if rank == 0:
            buf[0] = i
            buf[1] = tid
            mpi.send(buf, dest=1, tag=3000)

        elif rank == 1:
            mpi.recv(buf, source=0, tag=3000)
            values[i] = buf[0]

    return values


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def root_print(*args: object, **kwargs: object) -> None:
    if py_rank == 0:
        print(*args, **kwargs)


def summarize_threads(arr: np.ndarray) -> str:
    counts = Counter(int(x) for x in arr.tolist())
    return ", ".join(f"thread {k}: {v}" for k, v in sorted(counts.items()))


def barrier(label: str | None = None) -> None:
    comm.Barrier()
    if label is not None:
        root_print(f"\n--- {label} ---")
    comm.Barrier()


def print_by_rank(title: str, value: object) -> None:
    comm.Barrier()
    for r in range(py_size):
        comm.Barrier()
        if py_rank == r:
            print(f"[rank {py_rank}] {title}: {value}", flush=True)
    comm.Barrier()


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=_CLI_DEFAULT_THREADS)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument(
        "--race-n",
        type=int,
        default=2_000_000,
        help="Number of increments for the pure prange race test.",
    )
    parser.add_argument(
        "--dangerous",
        action="store_true",
        help="Run concurrent MPI calls from inside prange.",
    )
    parser.add_argument(
        "--chaos",
        action="store_true",
        help="Run same-tag concurrent MPI test. Requires --dangerous.",
    )
    parser.add_argument(
        "--force-dangerous-without-thread-multiple",
        action="store_true",
        help="Do not refuse dangerous tests even if MPI_THREAD_MULTIPLE was not provided.",
    )
    args = parser.parse_args()

    set_num_threads(args.threads)

    barrier("environment")

    root_print(f"Python executable: {sys.executable}")
    root_print(f"MPI ranks:         {py_size}")
    root_print(f"CPUs in affinity:  {len(os.sched_getaffinity(0))}")
    root_print(f"Requested threads:{args.threads}")
    root_print(f"Numba threads:    {get_num_threads()}")
    root_print(f"MPI thread level: {THREAD_LEVEL_NAMES.get(provided, provided)}")

    try:
        root_print(f"Numba threading layer before compilation: {nb.threading_layer()}")
    except ValueError:
        root_print("Numba threading layer before compilation: not initialized yet")

    root_print("")
    root_print("Relevant environment variables:")
    for key in [
        "NUMBA_NUM_THREADS",
        "NUMBA_THREADING_LAYER",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ]:
        root_print(f"  {key}={os.environ.get(key)}")

    if py_size != 2:
        root_print("")
        root_print(
            "WARNING: The MPI communication tests are written for exactly 2 ranks."
        )
        root_print("         Extra ranks will mostly stay idle.")
        root_print(
            "         Recommended command: mpiexec -n 2 python test_numba_mpi_prange.py"
        )

    # ------------------------------------------------------------------
    # Pure prange behavior.
    # ------------------------------------------------------------------

    barrier("pure Numba prange thread mapping")

    thread_map = prange_thread_map(args.n)
    print_by_rank("prange iteration -> Numba thread", thread_map)
    print_by_rank("thread histogram", summarize_threads(thread_map))

    try:
        root_print(f"Numba threading layer after compilation: {nb.threading_layer()}")
    except ValueError:
        pass

    barrier("pure prange race condition")

    wrong = unsafe_shared_counter(args.race_n)
    safe_partial = safe_per_thread_counter(args.race_n)
    safe_total = int(safe_partial.sum())

    print_by_rank("unsafe_shared_counter result", f"{wrong} expected {args.race_n}")
    print_by_rank("safe_per_thread_counter partial", safe_partial)
    print_by_rank(
        "safe_per_thread_counter total", f"{safe_total} expected {args.race_n}"
    )

    # ------------------------------------------------------------------
    # numba-mpi outside prange.
    # ------------------------------------------------------------------

    barrier("numba-mpi ping-pong outside prange")

    out = mpi_pingpong_outside_prange()
    print_by_rank("mpi_pingpong_outside_prange result", out)

    # ------------------------------------------------------------------
    # numba-mpi from one prange iteration.
    # ------------------------------------------------------------------

    barrier("numba-mpi single call site inside prange")

    result, seen = mpi_single_call_site_inside_prange(args.n)
    print_by_rank("single-call-site result", result)
    print_by_rank("single-call-site prange thread histogram", summarize_threads(seen))
    print_by_rank(
        "thread that likely executed iteration i=0",
        int(seen[0]) if len(seen) else None,
    )

    root_print("")
    root_print(
        "Interpretation: this test has only one MPI-calling prange iteration. "
        "So it is not concurrent MPI, but the MPI call may be made by a Numba "
        "worker thread rather than the process main thread."
    )

    # ------------------------------------------------------------------
    # Dangerous concurrent MPI tests.
    # ------------------------------------------------------------------

    if args.dangerous:
        if (
            provided != MPI.THREAD_MULTIPLE
            and not args.force_dangerous_without_thread_multiple
        ):
            barrier("dangerous concurrent MPI test skipped")
            root_print(
                "Skipping dangerous test because MPI did not provide "
                "MPI_THREAD_MULTIPLE."
            )
            root_print(
                "To force it anyway, rerun with "
                "--force-dangerous-without-thread-multiple."
            )
        else:
            barrier("DANGEROUS: concurrent MPI calls inside prange, distinct tags")

            recv_threads = mpi_concurrent_distinct_tags_in_prange(args.n)
            print_by_rank("received sender-thread ids", recv_threads)

            if py_rank == 1:
                missing = int(np.sum(recv_threads < 0))
                print(
                    f"[rank {py_rank}] missing receives: {missing}",
                    flush=True,
                )

            root_print("")
            root_print(
                "If this works, it only shows that this particular MPI build, "
                "numba-mpi version, and message pattern survived this test. "
                "It does not prove that arbitrary MPI-in-prange patterns are safe."
            )

            if args.chaos:
                barrier("VERY DANGEROUS: concurrent MPI calls inside prange, same tag")

                chaos_values = mpi_concurrent_same_tag_chaos(args.n)
                print_by_rank("same-tag received values", chaos_values)

                if py_rank == 1:
                    expected = np.arange(args.n, dtype=np.int64)
                    same_order = bool(np.array_equal(chaos_values, expected))
                    print(
                        f"[rank {py_rank}] same order as 0..n-1? {same_order}",
                        flush=True,
                    )

                root_print("")
                root_print(
                    "The same-tag test is expected to be nondeterministic. "
                    "It is meant to demonstrate that message ordering can differ "
                    "from loop iteration order."
                )
    else:
        barrier("dangerous tests not requested")
        root_print(
            "Skipped concurrent MPI-in-prange tests. "
            "Rerun with --dangerous to test them."
        )

    barrier("done")

    if not MPI.Is_finalized():
        MPI.Finalize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
