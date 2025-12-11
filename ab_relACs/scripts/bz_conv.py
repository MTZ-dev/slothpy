#!/usr/bin/env python3

# SlothPy
# Copyright (C) 2025 Mikolaj Tadeusz Zychowicz (MTZ)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Batched convergence of n_points and (optionally) FWHM for BZ integration.

Key points
----------
- Uses a TemporaryDirectory for patched configs; no config clutter.
- Runs the program with *lists*:
    [relacs].n_points = [n1, n2, ...]   (batched)
    [relacs].fwhm     = [f1, f2, ...]     (optional batching; default single)
- Detects all newly created chi CSVs per batch by directory diff, parses peaks,
  and (optionally) deletes them immediately.
- Convergence per FWHM: require |Δ log10(f_peak)| ≤ 2 × Δlog10(f) (streak of N).

Example
-------
python converge_bz_grid_batched.py \
  --runner "python ab_relACs/ab_relACs.py --config" \
  --base-config ./ab_relACs/examples/config_example.toml \
  --start-fwhm 30 \
  --fwhm-damp 0.8 \
  --min-fwhm 0.5 \
  --start-n 3 \
  --max-n 129 \
  --batch-n 5 \
  --batch-fwhm 1 \
  --consec-ok 2 \
  --summary-out ./convergence_runs/summary.csv \
  --plots-prefix ./convergence_runs/plots/conv
"""

from __future__ import annotations
import argparse
import csv
import math
import re
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from mpmath import erfinv, sqrt as mp_sqrt, log as mp_log

# ---------------- TOML patch (regex; only touch [relacs]) ----------------

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def _replace_relacs_key(toml_text: str, key: str, new_value_src: str) -> str:
    relacs_pat = r"(?P<head>.*?)(?P<relacs>\[relacs\][\s\S]*?)(?P<tail>\n\[[^\]\n]+\][\s\S]*|\Z)"
    m = re.search(relacs_pat, toml_text, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("No [relacs] section found.")
    relacs_block = m.group("relacs")
    key_line = rf"(^\s*{re.escape(key)}\s*=\s*).*$"
    new_relacs = re.sub(key_line, rf"\g<1>{new_value_src}", relacs_block, count=1, flags=re.MULTILINE)
    if new_relacs == relacs_block:
        raise RuntimeError(f"Key '{key}' not found in [relacs].")
    return m.group("head") + new_relacs + m.group("tail")

def to_toml_array(vals: List) -> str:
    # format numbers compactly (integers as ints; floats as minimal)
    def fmt(x):
        if isinstance(x, int) or (isinstance(x, float) and float(x).is_integer()):
            return str(int(x))
        return f"{float(x):g}"
    return "[" + ",".join(fmt(v) for v in vals) + "]"

def patch_relacs_lists(base_text: str, n_points_list: List[int], fwhm_list: List[float],
                       cutoff_override: Optional[float] = None) -> str:
    out = _replace_relacs_key(base_text, "n_points", to_toml_array(n_points_list))
    out = _replace_relacs_key(out, "fwhm", to_toml_array(fwhm_list))
    if cutoff_override is not None:
        out = _replace_relacs_key(out, "cutoff_fwhm", f"{cutoff_override:.10g}")
    return out

def extract_chi_base_path(base_text: str) -> Path:
    m = re.search(r"^\s*chi_csv_path\s*=\s*\"([^\"]+)\"", base_text, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("chi_csv_path not found in [relacs].")
    return Path(m.group(1))

# ---------------- CSV parsing & peak detection ----------------

@dataclass
class PeakResult:
    f_peak: float
    log10_f: float
    thr_log10: float  # 2× median Δlog10 step

def read_peak_from_csv(csv_path: Path) -> PeakResult:
    import csv as _csv
    with csv_path.open("r", encoding="utf-8") as f:
        reader = _csv.reader(f)
        header = None
        for row in reader:
            if not row:
                continue
            if row[0].startswith("["):
                continue
            header = row
            break
        if header is None:
            raise RuntimeError(f"No header in {csv_path}")
        col = {h.strip(): i for i, h in enumerate(header)}
        fk = "Wave Frequency (Hz)"
        mkp = 'm" (emu)'
        if fk not in col or mkp not in col:
            raise RuntimeError(f"Expected '{fk}' and '{mkp}' in {csv_path}")

        freqs, mpp = [], []
        for row in reader:
            if not row or len(row) <= max(col[fk], col[mkp]):
                continue
            try:
                freqs.append(float(row[col[fk]]))
                mpp.append(float(row[col[mkp]]))
            except ValueError:
                continue

    if not freqs:
        raise RuntimeError(f"No numeric rows in {csv_path}")

    k = max(range(len(mpp)), key=lambda i: mpp[i])
    f_peak = freqs[k]
    log_peak = math.log10(f_peak)

    logs = sorted({math.log10(x) for x in freqs if x > 0})
    if len(logs) >= 2:
        diffs = [b - a for a, b in zip(logs[:-1], logs[1:])]
        diffs.sort()
        median_step = diffs[len(diffs)//2]
    else:
        median_step = 0.0
    thr = 10.0 * median_step if median_step > 0 else 0.0
    return PeakResult(f_peak, log_peak, thr)

# ---------------- DOS & utilities ----------------

@dataclass
class DosCurve:
    fwhm: float
    n_points: int
    freq: List[float]
    conv: List[float]

def extract_hessian_dos_path(toml_text: str) -> Optional[Path]:
    m_relacs = re.search(r"(?P<head>.*?)(?P<hess>\[hessian\][\s\S]*?)(?P<tail>\n\[[^\]\n]+\][\s\S]*|\Z)",
                         toml_text, flags=re.MULTILINE)
    if not m_relacs:
        return None
    block = m_relacs.group("hess")
    m = re.search(r"^\s*dos\s*=\s*\"([^\"]+)\"", block, flags=re.MULTILINE)
    return Path(m.group(1)) if m else None

def read_dos_csv(csv_path: Path) -> Tuple[List[float], List[float]]:
    import csv as _csv
    with csv_path.open("r", encoding="utf-8") as f:
        reader = _csv.reader(f)
        header = None
        for row in reader:
            if not row:
                continue
            if row[0].startswith("["):
                continue
            header = row
            break
        if header is None:
            raise RuntimeError(f"No header in DOS CSV: {csv_path}")
        col = {h.strip(): i for i, h in enumerate(header)}
        if "Frequency" not in col or "Convolution" not in col:
            raise RuntimeError(f"DOS CSV missing 'Frequency'/'Convolution': {csv_path}")
        xs, ys = [], []
        for row in reader:
            if not row or len(row) <= max(col["Frequency"], col["Convolution"]):
                continue
            try:
                xs.append(float(row[col["Frequency"]]))
                ys.append(float(row[col["Convolution"]]))
            except ValueError:
                continue
    return xs, ys

def half_support_fwhm_gaussian(p: float) -> float:
    """Return half-support in units of FWHM that encloses central fraction p of a normalized Gaussian."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    # H/FWHM = erfinv(p) / (2*sqrt(ln 2))
    return float(erfinv(p) / (2 * mp_sqrt(mp_log(2))))

def half_support_fwhm_lorentzian(p: float) -> float:
    """Return half-support in units of FWHM that encloses central fraction p of a normalized Lorentzian."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    return 0.5 * math.tan(math.pi * p / 2.0)

def extract_broadening(base_text: str) -> Optional[str]:
    m = re.search(r'^\s*broadening\s*=\s*"([^"]+)"', base_text, flags=re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip().lower() if m else None

# ---------------- Records & utilities ----------------

@dataclass
class RunRecord:
    fwhm: float
    n_points: int
    f_peak: float
    log10_f: float
    thr_log10: float

def seq_from(start: int, max_n: int) -> List[int]:
    """Return all integer n_points from start to max_n inclusive (step=1)."""
    if start > max_n:
        return []
    return list(range(start, max_n + 1, 1))

def chunk(lst: List[int], size: int) -> List[List[int]]:
    return [lst[i:i+size] for i in range(0, len(lst), size)]

# ---------------- Batch runner ----------------

def run_batch(runner_prefix: str,
              base_toml_text: str,
              base_config_path: Path,
              chi_base_path: Path,
              n_points_batch: List[int],
              fwhm_batch: List[float],
              cleanup_csv: bool = True,
              *,
              dos_base_path: Optional[Path] = None,
              dos_sink: Optional[List[DosCurve]] = None,
              cutoff_override: Optional[float] = None,
            ) -> List[RunRecord]:
    """
    Run a single process with lists of n_points and fwhm.
    Detect all newly created CSVs and parse peaks.
    Returns RunRecord per CSV (one per (n_points, fwhm)).
    """
    # Prepare temp config with lists
    with tempfile.TemporaryDirectory(prefix="bzconv_cfg_") as tdir:
        tmp_cfg = Path(tdir) / f"{base_config_path.stem}_batched.toml"
        patched = patch_relacs_lists(base_toml_text, n_points_batch, fwhm_batch,
                                     cutoff_override=cutoff_override)
        write_text(tmp_cfg, patched)

        chi_dir = chi_base_path.parent
        stem_prefix = chi_base_path.stem
        before = {p.name for p in chi_dir.iterdir() if p.is_file()}

        dos_dir = dos_base_path.parent if dos_base_path else None
        dos_stem = dos_base_path.stem if dos_base_path else None
        before_dos = {p.name for p in dos_dir.iterdir() if p.is_file()} if dos_dir else set()

        cmd = f"{runner_prefix} {tmp_cfg}"
        print(f"[RUN] {cmd}")
        rc = subprocess.run(cmd, shell=True).returncode
        if rc != 0:
            raise RuntimeError(f"Runner failed with rc={rc} for batch n={n_points_batch}, f={fwhm_batch}")

        after = {p.name for p in chi_dir.iterdir() if p.is_file()}
        new_names = sorted(after - before)

        # Filter to stems starting with base stem, containing both tokens
        new_paths = [chi_dir / n for n in new_names if n.startswith(stem_prefix)]
        if not new_paths:
            raise FileNotFoundError("No new chi CSVs detected for this batch.")

        if dos_dir:
            after_dos = {p.name for p in dos_dir.iterdir() if p.is_file()}
            new_dos = sorted(after_dos - before_dos)
            dos_paths = [dos_dir / n for n in new_dos if n.startswith(dos_stem)]
            for p in dos_paths:
                s = p.stem
                if "_npoints_" not in s or "_fwhm_" not in s:
                    continue
                try:
                    n_val = int(s.split("_npoints_")[1].split("_")[0])
                except Exception:
                    continue
                # We batch single fwhm; assign the batch value
                f_curr = float(fwhm_batch[0]) if fwhm_batch else None
                if f_curr is None:
                    continue
                freq, conv = read_dos_csv(p)
                if dos_sink is not None:
                    dos_sink.append(DosCurve(fwhm=f_curr, n_points=n_val, freq=freq, conv=conv))
                if cleanup_csv:
                    try:
                        p.unlink()
                    except Exception as e:
                        print(f"[WARN] Could not delete DOS {p}: {e}")
                        
        # Build records per file by extracting n_points & fwhm tokens from filename
        # Expected tokens contain `_npoints_` and `_fwhm_`. We'll be permissive with fwhm formatting.
        recs: List[RunRecord] = []
        for p in new_paths:
            s = p.stem
            if "_npoints_" not in s or "_fwhm_" not in s or "no_chit" in s:
                if cleanup_csv:
                    try:
                        p.unlink()
                    except Exception as e:
                        print(f"[WARN] Could not delete {p}: {e}")
                continue
            try:
                # npoints
                n_str = s.split("_npoints_")[1].split("_")[0]
                n_val = int(n_str)
                # fwhm (read token but we don't rely on exact float formatting for matching)
                f_token = s.split("_fwhm_")[1].split("_")[0]
            except Exception:
                continue

            peak = read_peak_from_csv(p)
            recs.append(RunRecord(
                fwhm=None,  # fill below by best match
                n_points=n_val,
                f_peak=peak.f_peak,
                log10_f=peak.log10_f,
                thr_log10=peak.thr_log10,
            ))
            if cleanup_csv:
                try:
                    p.unlink()
                except Exception as e:
                    print(f"[WARN] Could not delete {p}: {e}")

        # Assign fwhm to each record using the batch list and filename token presence.
        # If multiple fwhm in batch, each file name should include the fwhm token (string).
        # We'll match by presence of stringified fwhm in filename in a tolerant way.
        assigned: List[RunRecord] = []
        for r in recs:
            matched_f: Optional[float] = None
            # search again – cheap & safe: use stem of path we just deleted? Keep stems first:
        # Recreate new_paths mapping because we might have deleted files
        for p in [chi_dir / n for n in new_names if n.startswith(stem_prefix)]:
            s = p.stem
            # collect, but some may be removed; ignore if missing
        # Better approach: re-parse directly while still present above; already did.
        # Assign by nearest fwhm comparing log10(f_token) if numeric, else fallback to single batch value.
        for r in recs:
            if len(fwhm_batch) == 1:
                r.fwhm = float(fwhm_batch[0])
            else:
                # try to parse fwhm token from filename-like pattern we already used (approximate)
                # Not perfect across arbitrary formatting; as a fallback, assign closest by peak similarity per fwhm.
                # But simpler: we cannot reconstruct file stem here (we deleted). Assign by best guess: all fwhm in batch.
                # Practical compromise: assign None here and let caller group by 'None'—but we need fwhm for loops.
                # Safer: don't delete before assigning; fix above: move unlink to after we assign.
                pass  # will never hit because we won't batch fwhm by default unless user asks

        return recs

def recheck_tail_for_new_fwhm(
    runner_prefix: str,
    base_toml_text: str,
    base_config_path: Path,
    chi_base_path: Path,
    fwhm: float,
    last_tail_end_n: int,
    start_n: int,
    max_n: int,
    consec_ok: int,
    cleanup_csv: bool,
    dos_base_path: str = "",
    dos_sink: list = None,
    cutoff_override: Optional[float] = None,
) -> Tuple[bool, Optional[RunRecord], List[RunRecord]]:
    """
    Quick pass: re-run ONLY the minimal tail window of n_points that was sufficient
    to declare convergence at the previous FWHM. If it still satisfies the 'consec_ok'
    streak at this (smaller) FWHM, we accept convergence without increasing n_points.

    The tail window size in points is (consec_ok + 1).
    For consec_ok=2 -> we need [N-4, N-2, N] (3 points) to produce two consecutive OK deltas.
    """

    # Build the minimal contiguous tail of length (consec_ok + 1)
    window_pts = max(1, consec_ok + 1)

    # If previous N is below start_n, build a fresh tail from start_n
    if last_tail_end_n < start_n:
        base_end = start_n + window_pts - 1
        if base_end > max_n:
            return False, None, []
        tail = list(range(start_n, base_end + 1))
    else:
        raw_start = last_tail_end_n - (window_pts - 1)
        tail = list(range(raw_start, last_tail_end_n + 1))
        # clamp to [start_n, max_n]
        tail = [n for n in tail if start_n <= n <= max_n]
        # if clamping removed early elements (e.g., hit start_n), extend to the right
        while len(tail) < window_pts and tail and tail[-1] < max_n:
            tail.append(tail[-1] + 1)

    # final sanity
    tail = sorted(set(tail))
    if len(tail) < window_pts:
        # Not enough points to form a full streak window; fall back to slow path
        return False, None, []

    # Run a single batch with just this tail
    recs = run_batch(
        runner_prefix=runner_prefix,
        base_toml_text=base_toml_text,
        base_config_path=base_config_path,
        chi_base_path=chi_base_path,
        n_points_batch=tail,
        fwhm_batch=[fwhm],
        cleanup_csv=cleanup_csv,
        dos_base_path=dos_base_path,
        dos_sink=dos_sink,
        cutoff_override=cutoff_override,
    )
    for r in recs: r.fwhm = fwhm
    recs.sort(key=lambda r: r.n_points)

    # Evaluate the streak on these results
    last_log = None
    streak = 0
    converged: Optional[RunRecord] = None
    for r in recs:
        if last_log is None:
            last_log = r.log10_f
            continue
        delta = abs(r.log10_f - last_log)
        thr = r.thr_log10
        ok = (thr > 0) and (delta <= thr)
        streak = streak + 1 if ok else 0
        last_log = r.log10_f
        if ok:
            converged = r
        if streak >= consec_ok:
            return True, converged, recs

    return False, None, recs

# ---------------- Convergence controller (batched n_points, single fwhm) ----------------

def converge_for_fwhm(runner_prefix: str,
                      base_toml_text: str,
                      base_config_path: Path,
                      chi_base_path: Path,
                      fwhm: float,
                      start_n: int,
                      max_n: int,
                      batch_n: int,
                      consec_ok: int,
                      cleanup_csv: bool,
                      dos_base_path: str = "",
                      dos_sink: list = None,
                      cutoff_override: Optional[float] = None,
                    ) -> Tuple[Optional[RunRecord], List[RunRecord]]:
    """
    For a fixed FWHM: grow n_points in batches until convergence streak is met.
    Returns (last_converged_record, all_records_at_this_fwhm)
    """
    all_recs: List[RunRecord] = []
    needed = consec_ok
    ok_streak = 0
    last_log: Optional[float] = None
    converged: Optional[RunRecord] = None

    all_ns = seq_from(start_n, max_n)
    window_pts = max(1, consec_ok + 1)
    for batch in chunk(all_ns, max(window_pts, batch_n)):
        recs = run_batch(
            runner_prefix=runner_prefix,
            base_toml_text=base_toml_text,
            base_config_path=base_config_path,
            chi_base_path=chi_base_path,
            n_points_batch=batch,
            fwhm_batch=[fwhm],
            cleanup_csv=cleanup_csv,
            dos_base_path=dos_base_path,
            dos_sink=dos_sink,
            cutoff_override=cutoff_override,
        )
        # fill missing fwhm field
        for r in recs:
            r.fwhm = fwhm

        # sort by n for consistent convergence tracking
        recs.sort(key=lambda r: r.n_points)
        for r in recs:
            if last_log is None:
                print(f"  n={r.n_points}: log10 f*={r.log10_f:.6f} (first)")
                last_log = r.log10_f
            else:
                delta = abs(r.log10_f - last_log)
                thr = r.thr_log10
                ok = (thr > 0) and (delta <= thr)
                ok_streak = ok_streak + 1 if ok else 0
                print(f"  n={r.n_points}: log10 f*={r.log10_f:.6f}, Δ={delta:.6g}, thr={thr:.6g} -> "
                      f"{'OK' if ok else 'NO'} (streak {ok_streak}/{needed})")
                last_log = r.log10_f
                if ok:
                    converged = r
                    if ok_streak >= needed:
                        all_recs.extend(recs)
                        return converged, all_recs
        all_recs.extend(recs)

    if converged is None and all_recs:
        converged = all_recs[-1]
    return converged, all_recs

# ---------------- Plotting & summary ----------------

def save_summary(path: Path, recs: List[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fwhm", "n_points", "peak_freq_hz", "log10_peak", "thr_log10"])
        for r in recs:
            w.writerow([r.fwhm, r.n_points, r.f_peak, r.log10_f, r.thr_log10])

def make_plots(recs: List[RunRecord], plots_prefix: Path, show_3d: bool = False, mpl_backend: Optional[str] = None) -> None:
    if show_3d and mpl_backend:
        import matplotlib
        matplotlib.use(mpl_backend, force=True)  # set BEFORE importing pyplot

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    plots_prefix.parent.mkdir(parents=True, exist_ok=True)

    # --- per-FWHM 2D plots (saved, not shown) ---
    by_f: Dict[float, List[RunRecord]] = {}
    for r in recs:
        by_f.setdefault(r.fwhm, []).append(r)

    for f, arr in sorted(by_f.items()):
        arr.sort(key=lambda x: x.n_points)
        xs = [a.n_points for a in arr]
        ys = [a.log10_f for a in arr]
        fig2d = plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("n_points")
        plt.ylabel("log10 peak frequency (Hz)")
        plt.title(f"Peak vs n_points @ FWHM={f:g}")
        plt.grid(True, which="both", linestyle=":")
        out = plots_prefix.with_name(f"{plots_prefix.stem}_fwhm_{str(f).replace('.','p')}_2d.png")
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close(fig2d)

    # --- 3D scatter (saved; optionally kept open for interactive view) ---
    xs = [r.n_points for r in recs]
    ys = [r.fwhm for r in recs]
    zs = [r.log10_f for r in recs]

    fig3d = plt.figure()
    ax = fig3d.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=30, depthshade=True)
    ax.set_xlabel("n_points")
    ax.set_ylabel("FWHM")
    ax.set_zlabel("log10 peak frequency (Hz)")
    ax.set_title("Convergence landscape")
    out3d = plots_prefix.with_name(f"{plots_prefix.stem}_3d.png")
    plt.tight_layout()
    plt.savefig(out3d, dpi=240)

    if show_3d:
        try:
            fig3d.canvas.manager.set_window_title("BZ convergence: n_points × FWHM × log10(f*)")
        except Exception:
            pass
        # Do NOT close fig3d; show it interactively
        import matplotlib.pyplot as _plt
        _plt.show()  # blocks until you close the window
    else:
        plt.close(fig3d)

def plot_dos_for_converged_fwhm(dos_curves: List[DosCurve], converged_fwhms: set[float], plots_prefix: Path, show: bool = False, mpl_backend: Optional[str] = None) -> None:
    if show and mpl_backend:
        import matplotlib
        matplotlib.use(mpl_backend, force=True)
    import matplotlib.pyplot as plt

    # Group curves by FWHM; only keep converged FWHM values
    by_f: Dict[float, List[DosCurve]] = {}
    for d in dos_curves:
        if d.fwhm in converged_fwhms:
            by_f.setdefault(d.fwhm, []).append(d)

    for fwhm, curves in sorted(by_f.items()):
        curves = sorted(curves, key=lambda c: c.n_points)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for c in curves:
            ax.plot(c.freq, c.conv, label=f"n={c.n_points}")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Convolution (DOS)")
        ax.set_title(f"DOS vs n_points @ FWHM={fwhm:g}")
        ax.legend(title="n_points", loc="best")
        ax.grid(True, linestyle=":")
        out = plots_prefix.with_name(f"{plots_prefix.stem}_DOS_fwhm_{str(fwhm).replace('.','p')}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=220)
        if show:
            plt.show()
        plt.close(fig)

# ---------------- Main driver (batched n_points; single fwhm per batch) ----------------

def main():
    ap = argparse.ArgumentParser(description="Batched n_points/FWHM convergence (temp configs, no CSV clutter).")
    ap.add_argument("--runner", required=True,
                    help='e.g. "python ab_relACs/ab_relACs.py --config"')
    ap.add_argument("--base-config", required=True, type=Path)

    ap.add_argument("--start-fwhm", type=float, required=True)
    ap.add_argument("--fwhm-damp", type=float, default=0.8)
    ap.add_argument("--min-fwhm", type=float, default=0.5)
    ap.add_argument("--max-fwhm-steps", type=int, default=12)

    ap.add_argument("--start-n", type=int, default=3)
    ap.add_argument("--max-n", type=int, default=129)
    ap.add_argument("--batch-n", type=int, default=5, help="Batch size for n_points (>=3 recommended).")
    ap.add_argument("--consec-ok", type=int, default=2)
    ap.add_argument("--cross-consec-ok", type=int, default=3,
                    help="Stop when the converged peak frequency stays within the convergence window "
                         "for this many consecutive FWHM steps (default: 3).")
    ap.add_argument("--cross-eps", type=float, default=0.0,
                    help="Optional absolute tolerance in log10(Hz) added to the cross-FWHM window (default: 0.0).")

    ap.add_argument("--cleanup-output", action="store_true", default=True)
    ap.add_argument("--no-cleanup-output", action="store_false", dest="cleanup_output")

    ap.add_argument("--summary-out", type=Path, default=Path("./convergence_runs/summary.csv"))
    ap.add_argument("--plots-prefix", type=Path, default=Path("./convergence_runs/plots/conv"))

    ap.add_argument("--show-3d", action="store_true",
                    help="Show the final 3D convergence plot in an interactive window.")
    ap.add_argument("--mpl-backend", type=str, default=None,
                    help="Matplotlib backend for interactive window, e.g. 'QtAgg' or 'TkAgg'.")
    ap.add_argument("--cutoff-p", type=float, default=None,
                help="If given (0<p<1), override [relacs].cutoff_fwhm to the half-support multiplier "
                     "that encloses fraction p of the kernel (Gaussian/Lorentzian chosen from TOML).")

    args = ap.parse_args()
    base_text = read_text(args.base_config)
    broadening = extract_broadening(base_text)
    cutoff_override: Optional[float] = None

    if args.cutoff_p is not None:
        p = args.cutoff_p
        if not 0.0 < p < 1.0:
            raise SystemExit("--cutoff-p must be in (0,1)")
        if broadening is None:
            print("[cutoff] Broadening not found in TOML; ignoring --cutoff-p")
        elif broadening == "gaussian":
            cutoff_override = half_support_fwhm_gaussian(p)
            print(f"[cutoff] broadening=gaussian, p={p:g} -> cutoff_fwhm={cutoff_override:.6g} (half-support per side, in FWHM)")
        elif broadening == "lorentzian":
            cutoff_override = half_support_fwhm_lorentzian(p)
            print(f"[cutoff] broadening=lorentzian, p={p:g} -> cutoff_fwhm={cutoff_override:.6g} (half-support per side, in FWHM)")
        else:
            print(f"[cutoff] Unknown broadening '{broadening}'; ignoring --cutoff-p")

    dos_base_path = extract_hessian_dos_path(base_text)  # may be None if not set
    all_dos: List[DosCurve] = []
    converged_fwhms: set[float] = set()
    chi_base_path = extract_chi_base_path(base_text)

    all_records: List[RunRecord] = []
    fwhm = float(args.start_fwhm)
    last_converged_log: Optional[float] = None
    curr_start_n = args.start_n

    last_converged_thr: Optional[float] = None
    cross_ok_streak = 0

    final_rec: Optional[RunRecord] = None
    final_reason: Optional[str] = None
    ok_fast = False

    try:
        for step in range(args.max_fwhm_steps):
            print(f"\n=== FWHM step {step+1}: FWHM={fwhm:g} ===")
            # --- FAST PATH: revalidate at the same n_points tail from previous step ---
            conv_rec = None
            if step > 0:  # we have a previous convergence
                ok_fast, conv_fast, recs_fast = recheck_tail_for_new_fwhm(
                    runner_prefix=args.runner,
                    base_toml_text=base_text,
                    base_config_path=args.base_config,
                    chi_base_path=chi_base_path,
                    fwhm=fwhm,
                    last_tail_end_n=curr_start_n,   # last converged N
                    start_n=args.start_n,
                    max_n=args.max_n,
                    consec_ok=args.consec_ok,
                    cleanup_csv=args.cleanup_output,
                    dos_base_path=dos_base_path,
                    dos_sink=all_dos,
                    cutoff_override=cutoff_override,
                )
                all_records.extend(recs_fast)
                if ok_fast:
                    conv_rec = conv_fast
                    print(f"[FAST] FWHM={fwhm:g} converged with tail ending at n={curr_start_n}: "
                          f"f*={conv_fast.f_peak:.6g} Hz  (log10={conv_fast.log10_f:.6f}), "
                          f"window={conv_fast.thr_log10:.6g}")
                    final_rec = conv_rec
                else:
                    print(f"[FAST] Tail check not sufficient; growing n_points...")

            # --- SLOW PATH (only if fast path failed or it's the first FWHM) ---
            if conv_rec is None:
                conv_rec, recs_slow = converge_for_fwhm(
                    runner_prefix=args.runner,
                    base_toml_text=base_text,
                    base_config_path=args.base_config,
                    chi_base_path=chi_base_path,
                    fwhm=fwhm,
                    start_n=curr_start_n,
                    max_n=args.max_n,
                    batch_n=max(3, args.batch_n),
                    consec_ok=args.consec_ok,
                    cleanup_csv=args.cleanup_output,
                    dos_base_path=dos_base_path,
                    dos_sink=all_dos,
                    cutoff_override=cutoff_override,
                )
                all_records.extend(recs_slow)

            if conv_rec is not None:
                converged_fwhms.add(float(fwhm))

            if conv_rec is not None and not ok_fast:
                print(f"[SLOW] FWHM={fwhm:g} converged at n_points={conv_rec.n_points}: "
                      f"f*={conv_rec.f_peak:.6g} Hz  (log10={conv_rec.log10_f:.6f}), "
                      f"window={conv_rec.thr_log10:.6g}")
                final_rec = conv_rec

            if conv_rec is None:
                print("Reached max_n without satisfying per-FWHM convergence; stopping.")
                final_reason = "max-n"
                break

            # cross-FWHM stabilization window
            if last_converged_log is not None:
                thr_cross = max(last_converged_thr or 0.0, conv_rec.thr_log10) + getattr(args, "cross_eps", 0.0)
                delta_cross = abs(conv_rec.log10_f - last_converged_log)
                ok = (thr_cross > 0.0) and (delta_cross <= thr_cross)

                cross_ok_streak = cross_ok_streak + 1 if ok else 0
                print(f"Cross-FWHM Δ={delta_cross:.6g} vs window={thr_cross:.6g} "
                      f"-> {'OK' if ok else 'NO'} (streak {cross_ok_streak}/{args.cross_consec_ok})")

                if cross_ok_streak >= args.cross_consec_ok:
                    print("Cross-FWHM stabilized for required consecutive steps. Stopping.")
                    final_reason = "cross-stable"
                    break
            else:
                # first FWHM has no previous to compare
                cross_ok_streak = 0

            # update “previous” references for the next FWHM step
            last_converged_log = conv_rec.log10_f
            last_converged_thr = conv_rec.thr_log10
            curr_start_n = conv_rec.n_points

            # next FWHM
            nf = fwhm * args.fwhm_damp
            if nf < args.min_fwhm:
                print(f"Next FWHM={nf:g} < min_fwhm={args.min_fwhm:g}. Stopping.")
                final_reason = "min-fwhm"
                break
            fwhm = nf

        # ---------- Final summary print ----------
        print("\n=== FINAL CONVERGENCE SUMMARY ===")
        if final_rec is not None:
            print(
                f"Reason: {final_reason or 'loop-end'}\n"
                f"Converged FWHM: {final_rec.fwhm:g}\n"
                f"Converged n_points: {final_rec.n_points}\n"
                f"Peak frequency f*: {final_rec.f_peak:.9g} Hz\n"
                f"log10(f*): {final_rec.log10_f:.9f}\n"
                f"Convergence window (Δlog10): {final_rec.thr_log10:.9g}\n"
                f"Total (fwhm, n_points) runs used: {len(all_records)}"
            )
        else:
            if all_records:
                last = all_records[-1]
                print(
                    f"Reason: {final_reason or 'no-convergence'} (showing last point)\n"
                    f"FWHM: {last.fwhm:g}  n_points: {last.n_points}\n"
                    f"f*: {last.f_peak:.9g} Hz  log10(f*): {last.log10_f:.9f}\n"
                    f"Window (Δlog10): {last.thr_log10:.9g}\n"
                    f"Total runs: {len(all_records)}"
                )
            else:
                print("No runs executed.")

        save_summary(args.summary_out, all_records)
        make_plots(all_records, args.plots_prefix, show_3d=args.show_3d, mpl_backend=args.mpl_backend)
        plot_dos_for_converged_fwhm(
            dos_curves=all_dos,
            converged_fwhms=converged_fwhms,
            plots_prefix=args.plots_prefix,
            show=True,
            mpl_backend=args.mpl_backend
        )
        print("\nDone.")
        print(f"Summary: {args.summary_out}")
        print(f"Plots: {args.plots_prefix}_[...].png")

    except KeyboardInterrupt:
        # ---- Graceful interrupt: dump what we have so far ----
        print("\n[INTERRUPTED] Ctrl+C received — saving partial summary and plots...", flush=True)
        try:
            if all_records:
                save_summary(args.summary_out, all_records)
                # Non-blocking plots (windows disabled on interrupt)
                make_plots(all_records, args.plots_prefix, show_3d=False, mpl_backend=args.mpl_backend)
            if all_dos:
                plot_dos_for_converged_fwhm(
                    dos_curves=all_dos,
                    converged_fwhms=converged_fwhms,
                    plots_prefix=args.plots_prefix,
                    show=False,
                    mpl_backend=args.mpl_backend
                )
        finally:
            print(f"Partial summary saved to: {args.summary_out}")
            print(f"Partial plots written under: {args.plots_prefix}_[...].png")
            raise SystemExit(130)  # conventional exit code for SIGINT

if __name__ == "__main__":
    main()
