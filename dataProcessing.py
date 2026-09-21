"""
Create per–cell-line train/val/test NPZ datasets for Enformer-style training using:
  • Reference FASTA
  • One or more input bigWig tracks per cell (variable count, specified in cells.tsv)
  • ATAC narrowPeak (for window selection)
  • Additional bigWig tracks as labels (tracks.tsv)
  • Optional blacklist BED(s) to exclude problematic regions

Key features
------------
1) Per-cell splits: For each cell line listed in cells.tsv, produce its own train/val/test file lists.
2) Chromosome-based splitting: Provide explicit train/val/test chromosome lists to avoid leakage across splits.
3) Peak windows only: Windows are centered on peak summits (or midpoints if summit missing).
4) Inputs include per-base bigWig signal(s): Save each input bigWig as its own per-base array alongside DNA one-hot.
5) Blacklist removal: Skip any window that overlaps a blacklist region.

Output NPZ (per window)
-----------------------
  x            : float32 [4, L]          # DNA one-hot (A,C,G,T)
  x_<bw_col>   : float32 [L]             # Per-base values from each input bigWig column in cells.tsv
  y            : float32 [C, T]          # Labels center-cropped; T = target_length (bins)
  id           : str                     # "cell|chr:start-end|peak=<peak_index>"
  cell, chrom, start, end, center        # metadata
  input_bw_cols : str[]                  # original input BW column names (in order)
  input_bw_keys : str[]                  # NPZ keys used (sanitized), parallel to input_bw_cols

File formats
------------
1) cells.tsv (tab-separated, header required):
   cell    narrowpeak    bw1    bw2    bw3 ...
   - Columns after 'cell' and 'narrowpeak' are treated as input bigWig paths.
   - You can have 0..N such columns.
   - Empty values are allowed and will be skipped for that row.

2) tracks.tsv (tab-separated):
   With cell-specific tracks (recommended):
     cell    name    path
   Or without cell column (same tracks used for all cells):
     name    path

CLI example
-----------
python dataProcessing.py \
  --fasta GRCh38.primary_assembly.genome.fa \
  --cells cells.tsv \
  --tracks tracks.tsv \
  --out_dir ./datasets \
  --sequence_length 196608 --target_length 896 --bin_size 128 \
  --val_chroms chr8 chr14 \
  --test_chroms chr10 chr22 \
  --blacklist hg38.blacklist.bed

Notes
-----
- Binning: L_bins = sequence_length / bin_size (must be integer). Labels are binned with
  pyBigWig.stats(..., nBins=L_bins), then we crop the center T = target_length bins.
- Peak center: narrowPeak column 10 (summit) is added to start; if missing, use midpoint.
- Duplicate/super-dense peaks: we de-duplicate by requiring a minimum center-to-center spacing (min_peak_distance_bp).
"""
import os
import re
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from bisect import bisect_left

try:
    import pyBigWig  # type: ignore
except Exception as e:
    raise SystemExit("Please install pyBigWig: pip install pyBigWig") from e

try:
    from pyfaidx import Fasta  # type: ignore
except Exception as e:
    raise SystemExit("Please install pyfaidx: pip install pyfaidx") from e


# ------------------------------
# DNA one-hot encoding
# ------------------------------
_BASE_TO_ROW = {ord('A'): 0, ord('a'): 0, ord('C'): 1, ord('c'): 1, ord('G'): 2, ord('g'): 2, ord('T'): 3, ord('t'): 3}


def one_hot_dna(seq: str) -> np.ndarray:
    L = len(seq)
    x = np.zeros((4, L), dtype=np.float32)
    for i, ch in enumerate(seq):
        row = _BASE_TO_ROW.get(ord(ch), -1)
        if row >= 0:
            x[row, i] = 1.0
    return x


# ------------------------------
# IO helpers
# ------------------------------
def _sanitize_npz_key(s: str) -> str:
    """
    NPZ keys should be simple strings; sanitize input BW column names into safe identifiers.
    Example: "bw-1" -> "bw_1"
    """
    s2 = re.sub(r'[^0-9A-Za-z_]+', '_', s.strip())
    s2 = re.sub(r'_+', '_', s2).strip('_')
    return s2 if s2 else "bw"


def read_cells_tsv(path: str) -> List[Dict[str, object]]:
    """
    cells.tsv header required:
        cell    narrowpeak    bw1    bw2    bw3 ...
    Returns list of dicts:
        {
          'cell': str,
          'narrowpeak': str,
          'input_bw_cols': List[str],   # original column names
          'input_bw_items': List[Tuple[str,str]],  # (npz_key, bw_path)
          'input_bw_keys': List[str],   # npz_key list in same order as cols
        }
    """
    rows: List[Dict[str, object]] = []
    with open(path, 'r') as f:
        header = f.readline().strip().split('\t')
        if not header or len(header) < 2:
            raise ValueError("cells.tsv must have at least columns: cell, narrowpeak")

        cols = {name: i for i, name in enumerate(header)}
        for r in ('cell', 'narrowpeak'):
            if r not in cols:
                raise ValueError(f"cells.tsv missing required column: {r}")

        # treat everything else as input bigWig columns
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
    """Return dict cell -> list of (name, bw_path). If file has no 'cell' column, use key '*' for all cells."""
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
            # assume two columns: name, path
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
        self.by_chrom: Dict[str, List[Tuple[int, int, int]]] = {}
        # store (start, end, center)

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
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError:
                    continue
                summit_center = None
                # narrowPeak "summit" is column 10 (0-based index 9) = offset from start
                if len(parts) >= 10 and parts[9] not in ('', '.', 'NA'):
                    try:
                        summit_center = start + int(float(parts[9]))
                    except ValueError:
                        summit_center = None
                if summit_center is None:
                    summit_center = (start + end) // 2
                self.by_chrom.setdefault(chrom, []).append((start, end, int(summit_center)))
        # sort by center
        for chrom, lst in self.by_chrom.items():
            lst.sort(key=lambda t: t[2])

    def dedupe_min_distance(self, min_dist: int):
        if min_dist <= 0:
            return
        for chrom, lst in self.by_chrom.items():
            kept: List[Tuple[int, int, int]] = []
            last_center = None
            for s, e, c in lst:
                if last_center is None or abs(c - last_center) >= min_dist:
                    kept.append((s, e, c))
                    last_center = c
            self.by_chrom[chrom] = kept


# ------------------------------
# Blacklist helpers (BED)
# ------------------------------
class Blacklist:
    """
    Interval index per chromosome (sorted + merged) with cached start arrays.
    Overlap query is O(log n + k) and avoids rebuilding lists per query.
    """
    def __init__(self):
        self.by_chrom: Dict[str, List[Tuple[int, int]]] = {}
        self.starts_by_chrom: Dict[str, np.ndarray] = {}

    def load_bed(self, path: str):
        with open(path, 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                    continue
                parts = line.rstrip().split('\t')
                if len(parts) < 3:
                    continue
                chrom = parts[0]
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError:
                    continue
                if end <= start:
                    continue
                self.by_chrom.setdefault(chrom, []).append((start, end))

        # sort + merge for speed
        for chrom, ivs in list(self.by_chrom.items()):
            if not ivs:
                continue
            ivs.sort()
            merged: List[Tuple[int, int]] = []
            for s, e in ivs:
                if not merged or s > merged[-1][1]:
                    merged.append((s, e))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            self.by_chrom[chrom] = merged
            self.starts_by_chrom[chrom] = np.fromiter((s for s, _ in merged), dtype=np.int64)

    def overlaps_any(self, chrom: str, start: int, end: int) -> bool:
        ivs = self.by_chrom.get(chrom)
        if not ivs:
            return False
        starts = self.starts_by_chrom.get(chrom)
        if starts is None or starts.size == 0:
            return False

        # Candidate index where interval start >= query start
        i = int(np.searchsorted(starts, start, side='left'))

        # Check i-1 then scan forward while starts < end
        j = max(i - 1, 0)
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
    def __init__(self, items: List[Tuple[str, str]]):  # (name, path)
        self.names = [n for n, _ in items]
        self.bws = [BW(p) for _, p in items]

    def close(self):
        for b in self.bws:
            b.close()

    def bin_all(self, chrom: str, start: int, end: int, n_bins: int, agg: str = 'sum') -> np.ndarray:
        arrs = [b.bin(chrom, start, end, n_bins, agg=agg) for b in self.bws]
        return np.stack(arrs, axis=0)  # [C, n_bins]

    def values_all(self, chrom: str, start: int, end: int) -> np.ndarray:
        arrs = [b.values(chrom, start, end) for b in self.bws]  # each [L]
        return np.stack(arrs, axis=0)  # [C, L]


# ------------------------------
# Chrom sizes
# ------------------------------
def chromosome_sizes(fa) -> Dict[str, int]:
    return {name: len(fa[name]) for name in fa.keys()}


# ------------------------------
# Main builder per cell
# ------------------------------
def build_for_cell(
    cell: str,
    fasta_path: str,
    narrowpeak_path: str,
    input_bw_items: List[Tuple[str, str]],   # (npz_key, bw_path)
    input_bw_cols: List[str],                # original column names
    input_bw_keys: List[str],                # npz_key list
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
):
    os.makedirs(out_dir, exist_ok=True)
    cell_dir = os.path.join(out_dir, cell)
    npz_dir = os.path.join(cell_dir, 'npz')
    os.makedirs(npz_dir, exist_ok=True)

    fa = Fasta(fasta_path, as_raw=True, one_based_attributes=False)
    sizes = chromosome_sizes(fa)
    allow_re = re.compile(allow_regex)

    # Peaks
    peaks = Peaks()
    peaks.load_narrowpeak(narrowpeak_path)
    peaks.dedupe_min_distance(min_peak_distance_bp)

    # Input bigWigs (0..N)
    inputs_ts = TrackSet(input_bw_items) if input_bw_items else None

    # Label tracks (0..N)
    labels_ts = TrackSet(label_tracks_for_cell) if label_tracks_for_cell else None

    L = sequence_length
    half = L // 2
    L_bins = sequence_length // bin_size
    if L_bins * bin_size != sequence_length:
        raise ValueError("sequence_length must be divisible by bin_size")
    crop_start = (L_bins - target_length) // 2
    crop_end = crop_start + target_length
    if crop_start < 0:
        raise ValueError(f"target_length={target_length} cannot exceed L_bins={L_bins}")

    def split_of(chrom: str) -> str:
        if chrom in test_chroms:
            return 'test'
        if chrom in val_chroms:
            return 'val'
        return 'train'

    # Counters
    counters = {'train': 0, 'val': 0, 'test': 0}
    idx_paths = {'train': [], 'val': [], 'test': []}
    stats = {
        'peaks_loaded': 0,
        'skipped_bad_contig': 0,
        'skipped_oob': 0,
        'skipped_blacklist': 0,
    }

    try:
        for chrom, peaks_list in peaks.by_chrom.items():
            if chrom not in sizes or not allow_re.match(chrom):
                stats['skipped_bad_contig'] += len(peaks_list)
                continue
            clen = sizes[chrom]
            split = split_of(chrom)
            kept_on_chrom = 0

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

                # DNA
                frag = fa[chrom][start:end]
                seq = frag if isinstance(frag, str) else frag.seq
                x_dna = one_hot_dna(seq)  # [4, L]

                # Inputs: per-base values, written as separate x_<col> arrays
                inputs_dict = {}
                if inputs_ts is not None and len(inputs_ts.bws) > 0:
                    x_inputs = inputs_ts.values_all(chrom, start, end)  # [K, L]
                    for k, key in enumerate(input_bw_keys):
                        inputs_dict[key] = x_inputs[k].astype(np.float32)

                # Labels: bin then crop to target_length
                if labels_ts is not None and len(labels_ts.bws) > 0:
                    y_full = labels_ts.bin_all(chrom, start, end, n_bins=L_bins, agg=agg)  # [C, L_bins]
                    y = y_full[:, crop_start:crop_end]  # [C, T]
                else:
                    y = np.zeros((0, target_length), dtype=np.float32)

                sid = f"{cell}|{chrom}:{start}-{end}|peak={i}"
                out_path = os.path.join(npz_dir, f"{chrom}_{start}_{end}_p{i}.npz")

                np.savez_compressed(
                    out_path,
                    x=x_dna.astype(np.float32),
                    y=y.astype(np.float32),
                    id=sid,
                    cell=cell,
                    chrom=chrom,
                    start=np.int64(start),
                    end=np.int64(end),
                    center=np.int64(center),
                    input_bw_cols=np.array(input_bw_cols, dtype=object),
                    input_bw_keys=np.array(input_bw_keys, dtype=object),
                    **inputs_dict,
                )

                idx_paths[split].append(out_path)
                counters[split] += 1
                kept_on_chrom += 1

            print(f"[{cell}] {chrom}: kept {kept_on_chrom} windows -> split {split}")
    finally:
        if inputs_ts is not None:
            inputs_ts.close()
        if labels_ts is not None:
            labels_ts.close()
        fa.close()

    # Write indices
    for sp in ('train', 'val', 'test'):
        out_list = os.path.join(cell_dir, f"{sp}_files.txt")
        with open(out_list, 'w') as f:
            for pth in idx_paths[sp]:
                f.write(pth + "\n")
    print(f"[{cell}] samples: {counters} | stats: {stats}")


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

    args = ap.parse_args()

    # Read configs
    cells = read_cells_tsv(args.cells)
    tracks_map = read_tracks_tsv(args.tracks)

    # Blacklist
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

    # Build per cell
    for info in cells:
        cell = str(info['cell'])
        npeaks = str(info['narrowpeak'])
        input_bw_items = info['input_bw_items']  # type: ignore
        input_bw_cols = info['input_bw_cols']    # type: ignore
        input_bw_keys = info['input_bw_keys']    # type: ignore

        # Cell-specific label tracks if present, else global
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
        )

    print("All cells done.")
