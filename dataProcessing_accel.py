"""
Accelerated data processing script.

Adds:
  • Multiprocessing with sharded output (--num_workers, --shard_size)
  • Fast interval-indexed blacklist filtering (precomputed starts for bisect)
  • Optional batch bigWig reads per chromosome region (--batch_bw_per_chrom, --max_batch_bp)

Input: same as dataProcessing_updated.py (cells.tsv variable BW columns, optional blacklist BED(s)).
Output: sharded NPZ files per split (train/val/test) under <out_dir>/<cell>/npz_shards/
"""
import os
import re
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional, Iterable
from bisect import bisect_left
from multiprocessing import get_context

try:
    import pyBigWig  # type: ignore
except Exception as e:
    raise SystemExit("Please install pyBigWig: pip install pyBigWig") from e

try:
    from pyfaidx import Fasta  # type: ignore
except Exception as e:
    raise SystemExit("Please install pyfaidx: pip install pyfaidx") from e


# ------------------------------
# DNA one-hot encoding (fast)
# ------------------------------
# byte->row lookup: A,C,G,T (case-insensitive), others -> -1
_LUT = np.full(256, -1, dtype=np.int16)
_LUT[ord('A')] = _LUT[ord('a')] = 0
_LUT[ord('C')] = _LUT[ord('c')] = 1
_LUT[ord('G')] = _LUT[ord('g')] = 2
_LUT[ord('T')] = _LUT[ord('t')] = 3


def one_hot_dna(seq: str) -> np.ndarray:
    b = np.frombuffer(seq.encode('ascii', 'ignore'), dtype=np.uint8)
    rows = _LUT[b]  # [-1..3]
    L = rows.shape[0]
    x = np.zeros((4, L), dtype=np.float32)
    for r in (0, 1, 2, 3):
        m = (rows == r)
        if m.any():
            x[r, m] = 1.0
    return x


# ------------------------------
# IO helpers
# ------------------------------
def _sanitize_npz_key(s: str) -> str:
    s2 = re.sub(r'[^0-9A-Za-z_]+', '_', s.strip())
    s2 = re.sub(r'_+', '_', s2).strip('_')
    return s2 if s2 else "bw"


def read_cells_tsv(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, 'r') as f:
        header = f.readline().strip().split('\t')
        if not header or len(header) < 2:
            raise ValueError("cells.tsv must have at least columns: cell, narrowpeak")
        cols = {name: i for i, name in enumerate(header)}
        for r in ('cell', 'narrowpeak'):
            if r not in cols:
                raise ValueError(f"cells.tsv missing required column: {r}")
        input_cols = [h for h in header if h not in ('cell', 'narrowpeak')]

        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.rstrip().split('\t')
            cell = parts[cols['cell']]
            npeaks = parts[cols['narrowpeak']]

            input_bw_cols: List[str] = []
            input_bw_items: List[Tuple[str, str]] = []
            input_bw_keys: List[str] = []
            for c in input_cols:
                idx = cols[c]
                bw_path = parts[idx] if idx < len(parts) else ''
                if bw_path is None or str(bw_path).strip() == '':
                    continue
                key = "x_" + _sanitize_npz_key(c)
                input_bw_cols.append(c)
                input_bw_keys.append(key)
                input_bw_items.append((key, bw_path))

            rows.append({
                'cell': cell,
                'narrowpeak': npeaks,
                'input_bw_cols': input_bw_cols,
                'input_bw_items': input_bw_items,
                'input_bw_keys': input_bw_keys,
            })
    return rows


def read_tracks_tsv(path: str) -> Dict[str, List[Tuple[str, str]]]:
    with open(path, 'r') as f:
        header = f.readline().strip().split('\t')
        cols = {name: i for i, name in enumerate(header)}
        result: Dict[str, List[Tuple[str, str]]] = {}
        if 'cell' in cols and 'name' in cols and 'path' in cols:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.rstrip().split('\t')
                cell = parts[cols['cell']]
                name = parts[cols['name']]
                pathp = parts[cols['path']]
                result.setdefault(cell, []).append((name, pathp))
        else:
            name_idx = cols.get('name', 0)
            path_idx = cols.get('path', 1 if 'path' in cols else 1)
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.rstrip().split('\t')
                name = parts[name_idx]
                pathp = parts[path_idx]
                result.setdefault('*', []).append((name, pathp))
    return result


# ------------------------------
# narrowPeak parsing & peak selection
# ------------------------------
class Peaks:
    def __init__(self):
        self.by_chrom: Dict[str, List[Tuple[int, int, int]]] = {}  # (start,end,center)

    def load_narrowpeak(self, path: str):
        with open(path, 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.rstrip().split('\t')
                if len(parts) < 3:
                    continue
                chrom = parts[0]
                try:
                    start = int(parts[1]); end = int(parts[2])
                except ValueError:
                    continue
                if end <= start:
                    continue
                summit_center = None
                if len(parts) >= 10 and parts[9] not in ('', '.', 'NA'):
                    try:
                        summit_center = start + int(float(parts[9]))
                    except ValueError:
                        summit_center = None
                if summit_center is None:
                    summit_center = (start + end) // 2
                self.by_chrom.setdefault(chrom, []).append((start, end, int(summit_center)))
        for chrom, lst in self.by_chrom.items():
            lst.sort(key=lambda t: t[2])

    def dedupe_min_distance(self, min_dist: int):
        if min_dist <= 0:
            return
        for chrom, lst in self.by_chrom.items():
            kept: List[Tuple[int, int, int]] = []
            last_center = None
            for s, e, c in lst:
                if last_center is None or (c - last_center) >= min_dist:
                    kept.append((s, e, c))
                    last_center = c
            self.by_chrom[chrom] = kept


# ------------------------------
# Blacklist helpers (fast interval index)
# ------------------------------
class Blacklist:
    def __init__(self):
        self.by_chrom: Dict[str, List[Tuple[int, int]]] = {}
        self.starts_by_chrom: Dict[str, np.ndarray] = {}

    def load_bed(self, path: str):
        tmp: Dict[str, List[Tuple[int, int]]] = {}
        with open(path, 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                    continue
                parts = line.rstrip().split('\t')
                if len(parts) < 3:
                    continue
                chrom = parts[0]
                try:
                    start = int(parts[1]); end = int(parts[2])
                except ValueError:
                    continue
                if end <= start:
                    continue
                tmp.setdefault(chrom, []).append((start, end))

        for chrom, ivs in tmp.items():
            ivs.sort()
            merged: List[Tuple[int, int]] = []
            for s, e in ivs:
                if not merged or s > merged[-1][1]:
                    merged.append((s, e))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            self.by_chrom.setdefault(chrom, [])
            # merge across multiple BEDs
            self.by_chrom[chrom].extend(merged)

        # final merge per chrom
        for chrom, ivs in self.by_chrom.items():
            ivs.sort()
            merged: List[Tuple[int, int]] = []
            for s, e in ivs:
                if not merged or s > merged[-1][1]:
                    merged.append((s, e))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            self.by_chrom[chrom] = merged
        self._finalize()

    def _finalize(self):
        for chrom, ivs in self.by_chrom.items():
            self.starts_by_chrom[chrom] = np.array([s for s, _ in ivs], dtype=np.int64)

    def overlaps_any(self, chrom: str, start: int, end: int) -> bool:
        ivs = self.by_chrom.get(chrom)
        if not ivs:
            return False
        starts = self.starts_by_chrom[chrom]
        i = int(bisect_left(starts, start))
        # check i-1 and forward until start>=end
        j = i - 1 if i > 0 else i
        while j < len(ivs) and ivs[j][0] < end:
            s, e = ivs[j]
            if s < end and e > start:
                return True
            j += 1
        return False


def load_blacklists(paths: List[str]) -> Optional[Blacklist]:
    if not paths:
        return None
    bl = Blacklist()
    for p in paths:
        if p and str(p).strip():
            bl.load_bed(p)
    return bl


# ------------------------------
# BigWig helpers
# ------------------------------
class BW:
    def __init__(self, path: str):
        self.path = path
        self.bw = pyBigWig.open(path)
        if not self.bw.isBigWig():
            raise ValueError(f"Not a bigWig: {path}")

    def close(self):
        self.bw.close()

    def bin(self, chrom: str, start: int, end: int, n_bins: int, agg: str = 'sum') -> np.ndarray:
        vals = self.bw.stats(chrom, start, end, nBins=n_bins, type=agg)
        if vals is None:
            vals = [0.0] * n_bins
        arr = np.array([0.0 if v is None else float(v) for v in vals], dtype=np.float32)
        return arr

    def values(self, chrom: str, start: int, end: int) -> np.ndarray:
        vals = self.bw.values(chrom, start, end)
        if vals is None:
            return np.zeros(end - start, dtype=np.float32)
        arr = np.array(vals, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0)


class TrackSet:
    def __init__(self, items: List[Tuple[str, str]]):  # (name_or_key, path)
        self.names = [n for n, _ in items]
        self.bws = [BW(p) for _, p in items]

    def close(self):
        for b in self.bws:
            b.close()

    def bin_all(self, chrom: str, start: int, end: int, n_bins: int, agg: str = 'sum') -> np.ndarray:
        arrs = [b.bin(chrom, start, end, n_bins, agg=agg) for b in self.bws]
        return np.stack(arrs, axis=0)  # [C, n_bins]


# ------------------------------
# Chrom sizes
# ------------------------------
def chromosome_sizes(fa) -> Dict[str, int]:
    return {name: len(fa[name]) for name in fa.keys()}


# ------------------------------
# Window planning (parent process)
# ------------------------------
def plan_windows_for_cell(
    cell: str,
    fasta_path: str,
    narrowpeak_path: str,
    sequence_length: int,
    min_peak_distance_bp: int,
    allow_regex: str,
    train_chroms: List[str],
    val_chroms: List[str],
    test_chroms: List[str],
    limit_per_split: int,
    blacklist: Optional[Blacklist],
) -> Tuple[Dict[str, List[Tuple[str, int, int, int, int]]], Dict[str, int]]:
    """
    Returns:
      windows_by_split[split] = list of (chrom, start, end, center, peak_idx)
    """
    fa = Fasta(fasta_path, as_raw=True, one_based_attributes=False)
    sizes = chromosome_sizes(fa)
    fa.close()
    allow_re = re.compile(allow_regex)

    peaks = Peaks()
    peaks.load_narrowpeak(narrowpeak_path)
    peaks.dedupe_min_distance(min_peak_distance_bp)

    L = sequence_length
    half = L // 2

    def split_of(chrom: str) -> str:
        if chrom in test_chroms:
            return 'test'
        if chrom in val_chroms:
            return 'val'
        return 'train'

    windows_by_split: Dict[str, List[Tuple[str, int, int, int, int]]] = {'train': [], 'val': [], 'test': []}
    stats = {'peaks_loaded': 0, 'skipped_bad_contig': 0, 'skipped_oob': 0, 'skipped_blacklist': 0}

    counters = {'train': 0, 'val': 0, 'test': 0}

    for chrom, peaks_list in peaks.by_chrom.items():
        if chrom not in sizes or not allow_re.match(chrom):
            stats['skipped_bad_contig'] += len(peaks_list)
            continue
        clen = sizes[chrom]
        split = split_of(chrom)
        for i, (_ps, _pe, center) in enumerate(peaks_list):
            stats['peaks_loaded'] += 1
            start = center - half
            end = start + L
            if start < 0 or end > clen:
                stats['skipped_oob'] += 1
                continue
            if blacklist is not None and blacklist.overlaps_any(chrom, start, end):
                stats['skipped_blacklist'] += 1
                continue
            if limit_per_split and counters[split] >= limit_per_split:
                break
            windows_by_split[split].append((chrom, start, end, center, i))
            counters[split] += 1

    return windows_by_split, stats


# ------------------------------
# Shard worker
# ------------------------------
def _group_windows_into_batches(
    windows: List[Tuple[str, int, int, int, int]],
    max_batch_bp: int,
) -> List[Tuple[str, int, int, List[int]]]:
    """
    Group windows into batches per chrom where a single bigWig read can cover them.
    Returns list of batches: (chrom, batch_start, batch_end, idx_list) where idx_list indexes 'windows'.
    """
    batches: List[Tuple[str, int, int, List[int]]] = []
    # windows are assumed already sorted by (chrom, start)
    cur_chrom = None
    cur_start = None
    cur_end = None
    cur_idxs: List[int] = []
    for idx, (chrom, start, end, _center, _pi) in enumerate(windows):
        if cur_chrom is None:
            cur_chrom, cur_start, cur_end, cur_idxs = chrom, start, end, [idx]
            continue
        if chrom != cur_chrom or (end - cur_start) > max_batch_bp:
            batches.append((cur_chrom, int(cur_start), int(cur_end), cur_idxs))
            cur_chrom, cur_start, cur_end, cur_idxs = chrom, start, end, [idx]
        else:
            cur_end = max(cur_end, end)
            cur_idxs.append(idx)
    if cur_chrom is not None:
        batches.append((cur_chrom, int(cur_start), int(cur_end), cur_idxs))
    return batches


def shard_worker(job: Dict[str, object]) -> str:
    """
    job keys:
      cell, fasta_path, out_path, windows, input_bw_items, input_bw_keys, input_bw_cols,
      label_tracks, bin_size, target_length, agg,
      batch_bw_per_chrom, max_batch_bp
    """
    cell = job['cell']  # type: ignore
    fasta_path = job['fasta_path']  # type: ignore
    out_path = job['out_path']  # type: ignore
    windows: List[Tuple[str, int, int, int, int]] = job['windows']  # type: ignore

    input_bw_items: List[Tuple[str, str]] = job['input_bw_items']  # type: ignore
    input_bw_keys: List[str] = job['input_bw_keys']  # type: ignore
    input_bw_cols: List[str] = job['input_bw_cols']  # type: ignore

    label_tracks: List[Tuple[str, str]] = job['label_tracks']  # type: ignore
    bin_size: int = int(job['bin_size'])  # type: ignore
    target_length: int = int(job['target_length'])  # type: ignore
    agg: str = job['agg']  # type: ignore

    batch_bw_per_chrom: bool = bool(job['batch_bw_per_chrom'])  # type: ignore
    max_batch_bp: int = int(job['max_batch_bp'])  # type: ignore

    # Open resources per process
    fa = Fasta(fasta_path, as_raw=True, one_based_attributes=False)
    inputs_ts = TrackSet(input_bw_items) if input_bw_items else None
    labels_ts = TrackSet(label_tracks) if label_tracks else None

    try:
        N = len(windows)
        if N == 0:
            # Still write a valid shard with metadata so downstream loaders can infer channel ordering.
            empty_kwargs = {
                "x": np.zeros((0, 4, 0), dtype=np.float32),
                "y": np.zeros((0, 0, 0), dtype=np.float32),
                "id": np.empty((0,), dtype=object),
                "cell": np.empty((0,), dtype=object),
                "chrom": np.empty((0,), dtype=object),
                "start": np.zeros((0,), dtype=np.int64),
                "end": np.zeros((0,), dtype=np.int64),
                "center": np.zeros((0,), dtype=np.int64),
                "input_bw_cols": np.array(input_bw_cols, dtype=object),
                "input_bw_keys": np.array(input_bw_keys, dtype=object),
            }
            # Also emit empty arrays for each input bigWig key
            for _k in input_bw_keys:
                empty_kwargs[_k] = np.zeros((0, 0), dtype=np.float32)
            np.savez_compressed(out_path, **empty_kwargs)
            return out_path

        L = int(windows[0][2] - windows[0][1])
        L_bins = L // bin_size
        if L_bins * bin_size != L:
            raise ValueError("sequence_length must be divisible by bin_size")
        crop_start = (L_bins - target_length) // 2
        crop_end = crop_start + target_length
        if crop_start < 0:
            raise ValueError(f"target_length={target_length} cannot exceed L_bins={L_bins}")

        x = np.zeros((N, 4, L), dtype=np.float32)
        y = np.zeros((N, len(labels_ts.bws) if labels_ts else 0, target_length), dtype=np.float32)
        ids = np.empty((N,), dtype=object)
        chroms = np.empty((N,), dtype=object)
        starts = np.zeros((N,), dtype=np.int64)
        ends = np.zeros((N,), dtype=np.int64)
        centers = np.zeros((N,), dtype=np.int64)

        inputs_arrays: Dict[str, np.ndarray] = {}
        for key in input_bw_keys:
            inputs_arrays[key] = np.zeros((N, L), dtype=np.float32)

        # Sort windows to improve locality for batch reading (but keep original order mapping)
        order = sorted(range(N), key=lambda i: (windows[i][0], windows[i][1]))
        inv = np.empty((N,), dtype=np.int64)
        for new_i, old_i in enumerate(order):
            inv[old_i] = new_i
        w_sorted = [windows[i] for i in order]

        # Fill metadata + DNA in sorted order
        for j, (chrom, start, end, center, pi) in enumerate(w_sorted):
            frag = fa[chrom][start:end]
            seq = frag if isinstance(frag, str) else frag.seq
            x[j] = one_hot_dna(seq)
            ids[j] = f"{cell}|{chrom}:{start}-{end}|peak={pi}"
            chroms[j] = chrom
            starts[j] = start
            ends[j] = end
            centers[j] = center

        # Inputs bigWigs: either per-window calls, or batch per chrom region
        if inputs_ts is not None and len(inputs_ts.bws) > 0:
            if not batch_bw_per_chrom:
                for j, (chrom, start, end, _center, _pi) in enumerate(w_sorted):
                    for k, key in enumerate(input_bw_keys):
                        arr = inputs_ts.bws[k].values(chrom, start, end)
                        inputs_arrays[key][j] = arr
            else:
                batches = _group_windows_into_batches(w_sorted, max_batch_bp=max_batch_bp)
                for chrom, bstart, bend, idxs in batches:
                    for k, key in enumerate(input_bw_keys):
                        buf = inputs_ts.bws[k].values(chrom, bstart, bend)  # [bend-bstart]
                        for j in idxs:
                            _chrom, start, end, _center, _pi = w_sorted[j]
                            s0 = start - bstart
                            s1 = s0 + (end - start)
                            inputs_arrays[key][j] = buf[s0:s1]

        # Labels: keep per-window stats call (pyBigWig.stats) for correctness
        if labels_ts is not None and len(labels_ts.bws) > 0:
            for j, (chrom, start, end, _center, _pi) in enumerate(w_sorted):
                y_full = labels_ts.bin_all(chrom, start, end, n_bins=L_bins, agg=agg)  # [C, L_bins]
                y[j] = y_full[:, crop_start:crop_end]

        # Reorder back to original window order
        # x,y,inputs_arrays,meta are currently in sorted order; we need original order
        x_out = x[inv]
        y_out = y[inv]
        ids_out = ids[inv]
        chroms_out = chroms[inv]
        starts_out = starts[inv]
        ends_out = ends[inv]
        centers_out = centers[inv]
        inputs_out = {k: v[inv] for k, v in inputs_arrays.items()}

        np.savez_compressed(
            out_path,
            x=x_out,
            y=y_out,
            id=ids_out,
            cell=np.array([cell]*N, dtype=object),
            chrom=chroms_out,
            start=starts_out,
            end=ends_out,
            center=centers_out,
            input_bw_cols=np.array(input_bw_cols, dtype=object),
            input_bw_keys=np.array(input_bw_keys, dtype=object),
            **inputs_out,
        )
        return out_path
    finally:
        try:
            fa.close()
        except Exception:
            pass
        if inputs_ts is not None:
            inputs_ts.close()
        if labels_ts is not None:
            labels_ts.close()


# ------------------------------
# Build per cell with shards + multiprocessing
# ------------------------------
def build_for_cell(
    cell: str,
    fasta_path: str,
    narrowpeak_path: str,
    input_bw_items: List[Tuple[str, str]],
    input_bw_cols: List[str],
    input_bw_keys: List[str],
    label_tracks_for_cell: List[Tuple[str, str]],
    out_dir: str,
    train_chroms: List[str],
    val_chroms: List[str],
    test_chroms: List[str],
    sequence_length: int,
    target_length: int,
    bin_size: int,
    agg: str,
    min_peak_distance_bp: int,
    allow_regex: str,
    limit_per_split: int,
    blacklist: Optional[Blacklist],
    num_workers: int,
    shard_size: int,
    batch_bw_per_chrom: bool,
    max_batch_bp: int,
):
    os.makedirs(out_dir, exist_ok=True)
    cell_dir = os.path.join(out_dir, cell)
    shard_dir = os.path.join(cell_dir, 'npz_shards')
    os.makedirs(shard_dir, exist_ok=True)

    windows_by_split, stats = plan_windows_for_cell(
        cell=cell,
        fasta_path=fasta_path,
        narrowpeak_path=narrowpeak_path,
        sequence_length=sequence_length,
        min_peak_distance_bp=min_peak_distance_bp,
        allow_regex=allow_regex,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=test_chroms,
        limit_per_split=limit_per_split,
        blacklist=blacklist,
    )

    print(f"[{cell}] planned windows: {{train:{len(windows_by_split['train'])}, val:{len(windows_by_split['val'])}, test:{len(windows_by_split['test'])}}} | stats: {stats}")

    ctx = get_context("spawn")  # safer on HPC
    pool = ctx.Pool(processes=max(1, int(num_workers))) if num_workers and num_workers > 1 else None

    idx_paths: Dict[str, List[str]] = {'train': [], 'val': [], 'test': []}

    def submit_jobs(split: str):
        wins = windows_by_split[split]
        # chunk into shards
        jobs = []
        for si in range(0, len(wins), shard_size):
            shard_wins = wins[si:si+shard_size]
            shard_idx = si // shard_size
            out_path = os.path.join(shard_dir, f"{split}_shard{shard_idx:06d}.npz")
            jobs.append({
                'cell': cell,
                'fasta_path': fasta_path,
                'out_path': out_path,
                'windows': shard_wins,
                'input_bw_items': input_bw_items,
                'input_bw_keys': input_bw_keys,
                'input_bw_cols': input_bw_cols,
                'label_tracks': label_tracks_for_cell,
                'bin_size': bin_size,
                'target_length': target_length,
                'agg': agg,
                'batch_bw_per_chrom': batch_bw_per_chrom,
                'max_batch_bp': max_batch_bp,
            })
        return jobs

    try:
        for split in ('train', 'val', 'test'):
            jobs = submit_jobs(split)
            if not jobs:
                continue
            if pool is None:
                for job in jobs:
                    p = shard_worker(job)
                    idx_paths[split].append(p)
            else:
                # imap_unordered for throughput on network I/O; preserve list via return paths
                for p in pool.imap_unordered(shard_worker, jobs, chunksize=1):
                    idx_paths[split].append(p)
            idx_paths[split].sort()

        # Write indices
        for sp in ('train', 'val', 'test'):
            out_list = os.path.join(cell_dir, f"{sp}_files.txt")
            with open(out_list, 'w') as f:
                for pth in idx_paths[sp]:
                    f.write(pth + "\n")

        print(f"[{cell}] wrote shards: train={len(idx_paths['train'])}, val={len(idx_paths['val'])}, test={len(idx_paths['test'])}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()


# ------------------------------
# Main CLI
# ------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fasta', required=True)
    ap.add_argument('--cells', required=True, help='cells.tsv with columns: cell, narrowpeak, bw1, bw2, ... (0..N input bigWigs)')
    ap.add_argument('--tracks', required=True, help='tracks.tsv with columns: [cell,] name, path')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--sequence_length', type=int, default=196_608)
    ap.add_argument('--target_length', type=int, default=896)
    ap.add_argument('--bin_size', type=int, default=128)
    ap.add_argument('--agg', choices=['sum', 'mean', 'max', 'min'], default='sum')
    ap.add_argument('--val_chroms', nargs='*', default=None, help="Validation chromosomes. If None, no validation split.")
    ap.add_argument('--test_chroms', nargs='*', default=None, help="Test chromosomes. If None, no test split.")
    ap.add_argument('--allow_regex', type=str, default=r'^chr([1-9]|1[0-9]|2[0-2]|X|Y)$', help='Regex of contigs to include')
    ap.add_argument('--min_peak_distance_bp', type=int, default=4096)
    ap.add_argument('--limit_per_split', type=int, default=0)
    ap.add_argument('--blacklist', nargs='*', default=None, help='Optional blacklist BED file(s). Any overlap => skip window.')

    # Acceleration knobs
    ap.add_argument('--num_workers', type=int, default=8, help='Number of worker processes (each opens its own bigWig handles).')
    ap.add_argument('--shard_size', type=int, default=2048, help='Number of windows per output NPZ shard.')
    ap.add_argument('--batch_bw_per_chrom', action='store_true', help='Batch per-base bigWig reads by covering batches of nearby windows with one bigWig read.')
    ap.add_argument('--max_batch_bp', type=int, default=5_000_000, help='Maximum genomic span (bp) to cover in one batched bigWig read.')

    args = ap.parse_args()

    cells = read_cells_tsv(args.cells)
    tracks_map = read_tracks_tsv(args.tracks)

    blacklist = load_blacklists(args.blacklist or [])

    # Determine chrom lists
    fa = Fasta(args.fasta, as_raw=True, one_based_attributes=False)
    sizes = chromosome_sizes(fa)
    allow_re = re.compile(args.allow_regex)
    all_chroms = [c for c in sizes if allow_re.match(c)]
    fa.close()

    val_chroms = args.val_chroms if args.val_chroms is not None else []
    test_chroms = args.test_chroms if args.test_chroms is not None else []
    train_chroms = [c for c in all_chroms if c not in val_chroms and c not in test_chroms]

    print(f"Train chroms: {train_chroms}")
    print(f"Val chroms: {val_chroms}")
    print(f"Test chroms: {test_chroms}")
    if blacklist is not None:
        n_chroms = len(blacklist.by_chrom)
        n_ivs = sum(len(v) for v in blacklist.by_chrom.values())
        print(f"Blacklist loaded: {n_ivs} merged intervals across {n_chroms} contigs")

    # Heuristic for network bigWigs + many cores: don't over-parallelize by default
    if args.num_workers > 16 and not args.batch_bw_per_chrom:
        print(f"[warn] num_workers={args.num_workers} on network bigWigs can thrash I/O. Consider --batch_bw_per_chrom and/or lowering workers (8-16).")

    for info in cells:
        cell = str(info['cell'])
        npeaks = str(info['narrowpeak'])
        input_bw_items = info['input_bw_items']  # type: ignore
        input_bw_cols = info['input_bw_cols']    # type: ignore
        input_bw_keys = info['input_bw_keys']    # type: ignore
        label_tracks = tracks_map.get(cell, tracks_map.get('*', []))

        build_for_cell(
            cell=cell,
            fasta_path=args.fasta,
            narrowpeak_path=npeaks,
            input_bw_items=input_bw_items,  # type: ignore
            input_bw_cols=input_bw_cols,    # type: ignore
            input_bw_keys=input_bw_keys,    # type: ignore
            label_tracks_for_cell=label_tracks,
            out_dir=args.out_dir,
            train_chroms=train_chroms,
            val_chroms=val_chroms,
            test_chroms=test_chroms,
            sequence_length=args.sequence_length,
            target_length=args.target_length,
            bin_size=args.bin_size,
            agg=args.agg,
            min_peak_distance_bp=args.min_peak_distance_bp,
            allow_regex=args.allow_regex,
            limit_per_split=args.limit_per_split,
            blacklist=blacklist,
            num_workers=args.num_workers,
            shard_size=args.shard_size,
            batch_bw_per_chrom=args.batch_bw_per_chrom,
            max_batch_bp=args.max_batch_bp,
        )

    print("All cells done.")
