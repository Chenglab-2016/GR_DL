
import os
import json
import math
import argparse
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

def infer_motif_num_from_meme(meme_path: str) -> int:
    """Infer number of motifs by counting 'MOTIF ' header lines in a MEME file."""
    n = 0
    with open(meme_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("MOTIF "):
                n += 1
    if n <= 0:
        raise ValueError(f"No motifs found in MEME file: {meme_path}")
    return n

# Import user's model definitions (updated: significant motifs + addition inputs)
from model_motif import Enformer, EnformerConfig

"""
usage: #train
python train_eval.py \
  --mode train \
  --out_dir path/to/out/dir \
  --train_list train/files/list \
  --val_list /val/files/list \
  --heads "head=1" \
  --model_type motif-based-model \
  --motif_use_prior \
  --motif_pwm_path path/to/motif/meme \
  --sequence_length 2048 \
  --target_length 1024 \
  --batch 8 \
  --epochs 30 \
  --auto_fit_geometry \
  --lr 2e-4 \
  --amp
#test
python train_eval.py \
  --mode test \
  --motif_use_prior \
  --motif_pwm_path /path/to/motif/meme \
  --test_list test/files/list \
  --out_dir path/to/out/dir \
  --ckpt Path to load/save checkpoint \
  --heads "head=1 \
  --batch 8 \
  --amp
"""

# -------------------------------
# Utilities
# -------------------------------

def to_device(batch, device):
    """Recursively move tensors in batch to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)
    return batch

def nan_to_mask(t: torch.Tensor) -> torch.Tensor:
    """Return mask of valid (non-NaN) entries for a tensor."""
    return ~torch.isnan(t)

def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared error ignoring NaNs in target.
    Shapes: [B, T, C]
    """
    mask = nan_to_mask(target)
    diff = (pred - torch.nan_to_num(target, nan=0.0)) ** 2
    diff = diff * mask
    denom = mask.sum().clamp_min(1)
    return diff.sum() / denom

def masked_stats_mask(pred: torch.Tensor, target: torch.Tensor):
    # Valid where BOTH pred and target are finite; also ignore NaNs in target (legacy)
    m = torch.isfinite(pred) & torch.isfinite(target) & ~torch.isnan(target)
    return m

def masked_mse_safe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    m = torch.isfinite(pred) & torch.isfinite(target) & ~torch.isnan(target)
    if m.sum() == 0:
        # graph-connected 0.0
        return (pred * 0.0).sum()
    diff = (torch.nan_to_num(pred) - torch.nan_to_num(target))**2
    return diff[m].mean()

def masked_pearson(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Pearson correlation between flattened pred and target, ignoring NaNs in target.
    pred, target: np arrays [N, T, C] or flattened; NaNs in target ignored.
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    mask = ~np.isnan(target)
    if mask.sum() < 2:
        return float('nan')
    x = pred[mask].astype(np.float64)
    y = target[mask].astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())
    if denom == 0:
        return float('nan')
    return float((x * y).sum() / denom)

def read_list_file(list_path: str) -> List[str]:
    paths = []
    with open(os.path.expanduser(list_path)) as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            paths.append(os.path.expanduser(p))
    if not paths:
        raise ValueError(f"No paths found in {list_path}")
    return paths

def masked_huber_safe(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    m = torch.isfinite(pred) & torch.isfinite(target) & ~torch.isnan(target)
    if m.sum() == 0:
        return (pred * 0.0).sum()
    huber = torch.nn.SmoothL1Loss(reduction='none', beta=beta)
    loss = huber(torch.nan_to_num(pred), torch.nan_to_num(target))
    return loss[m].mean()

def masked_pearson_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    m = torch.isfinite(pred) & torch.isfinite(target) & ~torch.isnan(target)
    if m.sum() < 2:
        # return 0 (no penalty) but keep graph
        return (pred * 0.0).sum()
    x = torch.nan_to_num(pred)[m].double()
    y = torch.nan_to_num(target)[m].double()
    x = x - x.mean()
    y = y - y.mean()
    num = (x * y).sum()
    den = torch.sqrt((x.pow(2).sum()).clamp_min(1e-12)) * torch.sqrt((y.pow(2).sum()).clamp_min(1e-12))
    r = num / den
    return (1.0 - r).float()

def log1p_clamped(t: torch.Tensor) -> torch.Tensor:
    # For count-like signals (non-negative expected); clamp negatives to 0 before log1p
    return torch.log1p(torch.clamp(t, min=0))

class RunningVarEMA:
    def __init__(self, init: float = 1.0, momentum: float = 0.99):
        self.var = init
        self.m = momentum
        self.ready = False

    def update_with_batch(self, t: torch.Tensor):
        # variance computed on valid bins only
        m = torch.isfinite(t) & ~torch.isnan(t)
        if m.sum() < 2:
            return self.var
        x = torch.nan_to_num(t[m]).float()
        v = torch.var(x)
        if not torch.isfinite(v):
            return self.var
        self.var = self.m * self.var + (1 - self.m) * float(v.detach().item())
        self.ready = True
        return self.var

    def get(self):
        return max(self.var, 1e-6)

# -------------------------------
# Dataset
# -------------------------------

class MultiHeadNPZDataset(Dataset):
    def __init__(self, npz_path: str, heads: Dict[str, int], apply_log1p: bool = False):
        super().__init__()
        self.path = os.path.expanduser(npz_path)
        self.data = np.load(self.path, allow_pickle=True)
        self.heads = heads
        self.apply_log1p = apply_log1p
        head_names = list(heads.keys())
        head_sizes = [heads[h] for h in head_names]
        sumC = int(np.sum(head_sizes))

        # ---------- Load inputs ----------
        x = self.data['x']
        # Normalize x to have a batch axis N
        if x.ndim == 1:            # [L] (ints)
            x = x[None, ...]       # [1, L]
        elif x.ndim == 2:          # [4,L] or [L,4]
            x = x[None, ...]       # [1, 4, L] or [1, L, 4]
        elif x.ndim == 3:
            pass                   # [N, 4, L] or [N, L, 4]
        else:
            raise ValueError(f"Unexpected x shape {x.shape} in {self.path}")
        self.x = x

        # ---------- Load additional inputs (x_bw1, x_bw2, ...), optional ----------
        # New convention: store each additional per-base track as its own key (e.g. x_bw1).
        # We stack them into x_addition: [N, A, L].
        self.x_addition = None
        addition_keys = []

        # Preferred: explicit ordering
        if 'input_bw_keys' in self.data:
            try:
                addition_keys = [str(k) for k in self.data['input_bw_keys']]
            except Exception:
                addition_keys = []

        if not addition_keys:
            # Fallback: infer by key pattern
            for k in self.data.files:
                if k == 'x':
                    continue
                if k.startswith('x_bw'):
                    addition_keys.append(k)
            addition_keys = sorted(addition_keys)

        if addition_keys:
            chans = []
            for k in addition_keys:
                if k not in self.data:
                    continue
                arr = self.data[k]
                # normalize to [N, L]
                if arr.ndim == 1:
                    arr = arr[None, ...]
                elif arr.ndim == 2:
                    # could be [L, N] in weird cases; assume [N,L] if first dim == N
                    pass
                else:
                    raise ValueError(f"Unexpected {k} shape {arr.shape} in {self.path}; expected 1D/2D")
                chans.append(arr)
            if chans:
                # stack -> [A, N, L] then transpose -> [N, A, L]
                xa = np.stack(chans, axis=0)
                xa = np.transpose(xa, (1, 0, 2))
                self.x_addition = xa

        # ---------- Load targets ----------
        self.targets = {}
        self.combined_y = False
        self.y_all = None
        self.channels_last = None

        have_all = all((f"y_{h}" in self.data) for h in head_names)
        if have_all:
            # explicit per-head arrays
            for h in head_names:
                arr = self.data[f'y_{h}']
                if arr.ndim == 2:      # single sample: [T,C] or [C,T]
                    arr = arr[None, ...]
                elif arr.ndim != 3:
                    raise ValueError(f"Target y_{h} must be 2D or 3D, got {arr.shape} in {self.path}")
                self.targets[h] = arr
        else:
            # single 'y' that we will split across heads
            if 'y' not in self.data:
                missing = [f"y_{h}" for h in head_names if f"y_{h}" not in self.data]
                raise KeyError(f"Expected per-head targets ({missing}) or a single 'y' in {self.path}")

            y_all = self.data['y']
            # add batch if single-sample 2D
            if y_all.ndim == 2:
                y_all = y_all[None, ...]     # -> [1,T,C] or [1,C,T]
            if y_all.ndim != 3:
                raise ValueError(f"'y' must be 2D/3D; got {y_all.shape} in {self.path}")

            # detect channel axis using sumC
            if y_all.shape[-1] == sumC:
                channels_last = True   # [N,T,C]
            elif y_all.shape[-2] == sumC:
                channels_last = False  # [N,C,T]
            else:
                raise ValueError(f"Sum of head sizes ({sumC}) doesn't match channel dim of 'y' {y_all.shape} in {self.path}")

            self.combined_y = True
            self.y_all = y_all
            self.channels_last = channels_last



        # ---------- optional metadata ----------
        self.meta = {}
        for name in ['ids', 'chrom', 'start', 'end', 'center', 'cell']:
            if name in self.data:
                m = self.data[name]
                # normalize scalars or 1D to 1-element list if single sample
                if np.isscalar(m) or (isinstance(m, np.ndarray) and m.ndim == 0):
                    m = np.array([m])
                self.meta[name] = m
            else:
                self.meta[name] = None

        # ---------- dataset length ----------
        self.N = self.x.shape[0]
        # sanity: targets should have same N
        if self.combined_y:
            if self.y_all.shape[0] != self.N:
                raise ValueError(f"N mismatch: x N={self.N} vs y N={self.y_all.shape[0]} in {self.path}")
        else:
            for h in head_names:
                if self.targets[h].shape[0] != self.N:
                    raise ValueError(f"N mismatch for {h}: x N={self.N} vs y_{h} N={self.targets[h].shape[0]} in {self.path}")


    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        # x
        x = torch.from_numpy(self.x[idx]).float()
        if x.ndim == 2 and x.shape[0] != 4 and x.shape[1] == 4:
            x = x.transpose(0, 1)  # [L,4] -> [4,L]
        if x.ndim == 1:
            x = x.long()

        # additional input channels
        x_addition = None
        if self.x_addition is not None:
            xa = torch.from_numpy(self.x_addition[idx]).float()  # [A,L]
            if xa.ndim != 2:
                raise ValueError(f"x_addition must be 2D [A,L], got {xa.shape} in {self.path}")
            x_addition = xa

        # y per head -> [T, C_h]
        y = {}
        if not self.combined_y:
            for h, C_h in self.heads.items():
                arr = self.targets[h][idx]    # [T,C] or [C,T]
                t = torch.from_numpy(arr).float()
                if t.ndim != 2:
                    raise ValueError(f"Target for head {h} must be 2D, got {t.shape}")
                if self.apply_log1p:
                    t = log1p_clamped(t)
                y[h] = t

        else:
            arr = self.y_all[idx]             # [T,C] or [C,T]
            t = torch.from_numpy(arr).float()

            # Optional: get saved channel names
            y_names = None
            if 'y_names' in self.data:
                try:
                    y_names = [str(x) for x in self.data['y_names']]
                except Exception:
                    pass

            # Normalize to [C,T] for easy slicing
            if self.channels_last:
                y_ct = t.transpose(0, 1)  # [C,T]
            else:
                y_ct = t                  # already [C,T]

            # Map by name if available, else fallback to positional order
            if y_names is not None:
                name_to_idx = {name: i for i, name in enumerate(y_names)}
                for h, C_h in self.heads.items():
                    if C_h != 1:
                        raise ValueError(f"Head {h} expects {C_h} channels; mapper assumes 1.")
                    if h not in name_to_idx:
                        raise KeyError(f"Head {h} not found in y_names {y_names}")
                    i = name_to_idx[h]
                    y[h] = y_ct[i:i+1, :].transpose(0, 1)  # -> [T,1]
            else:
                cumsum = 0
                for h, C_h in self.heads.items():
                    y[h] = y_ct[cumsum:cumsum + C_h, :].transpose(0, 1)  # -> [T,C_h]
                    cumsum += C_h
            for h in list(y.keys()):
                t = y[h]
                if self.apply_log1p:
                    t = log1p_clamped(t)
                y[h] = t                            
        # meta
        m = {k: (self.meta[k][idx] if self.meta[k] is not None else None) for k in self.meta}

        return {'x': x, 'x_addition': x_addition, 'y': y, 'meta': m}

def collate_batch(samples: List[dict]) -> dict:
    """Collate samples into batch; x in [B,4,L] or [B,L] long; x_addition in [B,A,L]; y dict heads -> [B,T,C]."""
    # x
    xs = [s['x'] for s in samples]
    if xs[0].dtype in (torch.long, torch.int64, torch.int32):
        x = torch.stack(xs, dim=0).long()
    else:
        x = torch.stack(xs, dim=0).float()

    # x_addition
    if samples[0]['x_addition'] is not None:
        x_addition = torch.stack([s['x_addition'] for s in samples], dim=0).float()
    else:
        x_addition = None

    # y
    heads = samples[0]['y'].keys()
    y = {h: torch.stack([s['y'][h] for s in samples], dim=0).float() for h in heads}

    # meta list
    meta = {k: [s['meta'][k] for s in samples] for k in samples[0]['meta']}

    return {'x': x, 'x_addition': x_addition, 'y': y, 'meta': meta}


# -------------------------------
# Training / Validation
# -------------------------------

def train_one_epoch(model: Enformer,
                    loader: DataLoader,
                    optim: torch.optim.Optimizer,
                    device: torch.device,
                    head_weights: Dict[str, float],
                    running_var: Dict[str, "RunningVarEMA"],
                    scaler: Optional["torch.amp.GradScaler"] = None,
                    grad_clip: Optional[float] = None,
                    head_loss_mix: Optional[Dict[str,float]]=None) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_per_head = {h: 0.0 for h in head_weights}
    total_batches = 0

    for batch in loader:
        batch = to_device(batch, device)
        x = batch['x']
        x_addition = batch['x_addition']
        targets = batch['y']  # dict head -> [B,T,C]

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(scaler is not None and device.type == 'cuda')):
            outputs = model(x, x_addition_full=x_addition)  # dict head -> [B,T,C]
            loss_heads = {}

            for h in outputs:
                pred_h = outputs[h]
                tgt_h  = targets[h]

                # --- variance balancing (EMA over targets, raw scale) ---
                v = running_var[h].update_with_batch(tgt_h)
                v = max(v, 1e-6)

                # --- head-specific objectives ---
                hname = h.lower()
                loss_mse = masked_mse_safe(pred_h, tgt_h)
                loss_pr  = masked_pearson_torch(pred_h, tgt_h)  # (1 - r)

                if head_loss_mix and hname in head_loss_mix:
                    r = head_loss_mix[hname]
                    loss_h = r * (loss_mse / v) + (1 - r) * loss_pr
                else:
                    loss_h = masked_mse_safe(pred_h, tgt_h) / v


                loss_heads[h] = loss_h * head_weights[h]

            loss = sum(loss_heads.values())

            # --- finite guard (skip bad batches) ---
            if not torch.isfinite(loss):
                # optional debug:
                # print({k: float(v.detach().item()) for k,v in loss_heads.items()})
                optim.zero_grad(set_to_none=True)
                continue

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

        total_loss += float(loss.detach().item())
        for h in outputs:
            total_per_head[h] += float(loss_heads[h].detach().item())
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    avg_heads = {h: total_per_head[h] / max(1, total_batches) for h in head_weights}
    return avg_loss, avg_heads



@torch.no_grad()
def evaluate(model: Enformer,
             loader: DataLoader,
             device: torch.device,
             head_weights: Dict[str, float],
             running_var: Dict[str, "RunningVarEMA"],
             head_loss_mix: Optional[Dict[str,float]]=None) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_per_head = {h: 0.0 for h in head_weights}
    total_batches = 0

    for batch in loader:
        batch = to_device(batch, device)
        x = batch['x']
        x_addition = batch['x_addition']
        targets = batch['y']
        outputs = model(x, x_addition_full=x_addition)

        loss_heads = {}
        for h in outputs:
            pred_h = outputs[h]
            tgt_h  = targets[h]
            v = max(running_var[h].get(), 1e-6)
            hname = h.lower()
            loss_mse = masked_mse_safe(pred_h, tgt_h)
            loss_pr  = masked_pearson_torch(pred_h, tgt_h)
            if head_loss_mix and hname in head_loss_mix:
                r = head_loss_mix[hname]
                loss_h = r * (loss_mse / v) + (1 - r) * loss_pr
            else:
                loss_h = masked_mse_safe(pred_h, tgt_h) / v

            loss_heads[h] = loss_h * head_weights[h]

        loss = sum(loss_heads.values())

        total_loss += float(loss.detach().item())
        for h in outputs:
            total_per_head[h] += float(loss_heads[h].detach().item())
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    avg_heads = {h: total_per_head[h] / max(1, total_batches) for h in head_weights}
    return avg_loss, avg_heads



# -------------------------------
# Testing (dump + metrics)
# -------------------------------

@torch.no_grad()
def run_test(model: Enformer,
             loader: DataLoader,
             device: torch.device,
             out_dir: str,
             heads: Dict[str, int]):
    os.makedirs(out_dir, exist_ok=True)

    # Collect arrays
    preds = {h: [] for h in heads}
    trues = {h: [] for h in heads}
    meta_collect = {'ids': [], 'chrom': [], 'start': [], 'end': [], 'center': [], 'cell': []}

    for batch in loader:
        batch = to_device(batch, device)
        x = batch['x']
        x_addition = batch['x_addition']
        y = batch['y']
        outputs = model(x, x_addition_full=x_addition)  # dict head -> [B,T,C]

        for h in heads:
            preds[h].append(outputs[h].cpu().numpy())
            trues[h].append(y[h].cpu().numpy())

        # meta (keep as python objects)
        for k in meta_collect:
            meta_collect[k].extend(batch['meta'][k])

    # Stack
    for h in heads:
        preds[h] = np.concatenate(preds[h], axis=0)  # [N,T,C_h]
        trues[h] = np.concatenate(trues[h], axis=0)  # [N,T,C_h]

    # Compute metrics per head
    metrics = {}
    for h in heads:
        mse = float(np.nanmean((preds[h] - trues[h]) ** 2))
        pr = masked_pearson(preds[h], trues[h])
        metrics[h] = {'mse': mse, 'pearson': pr}

    # Save dump npz
    dump_path = os.path.join(out_dir, 'test_dump.npz')
    np.savez_compressed(
        dump_path,
        **{f'y_pred_{h}': preds[h] for h in heads},
        **{f'y_target_{h}': trues[h] for h in heads},
        ids=np.array(meta_collect['ids'], dtype=object),
        chrom=np.array(meta_collect['chrom'], dtype=object),
        start=np.array(meta_collect['start']),
        end=np.array(meta_collect['end']),
        center=np.array(meta_collect['center']),
        cell=np.array(meta_collect['cell'], dtype=object),
    )

    # Save metrics json
    with open(os.path.join(out_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Also return metrics for printing
    return metrics, dump_path


# -------------------------------
# Main (CLI)
# -------------------------------

def parse_head_sizes(s: str) -> Dict[str, int]:
    """
    Parse "head1=1,head2=3" -> {"head1": 1, "head2": 3}
    """
    out = {}
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        name, val = part.split('=')
        out[name.strip()] = int(val)
    return out

def parse_head_weights(s: Optional[str], heads: Dict[str, int]) -> Dict[str, float]:
    """
    Parse "head1=1.0,head2=0.5" with defaults 1.0 for missing.
    """
    weights = {h: 1.0 for h in heads}
    if not s:
        return weights
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        name, val = part.split('=')
        weights[name.strip()] = float(val)
    return weights

def build_cfg(args) -> EnformerConfig:
    # If motif_num is not provided (<=0) and a MEME file is provided, allow auto-inference.
    auto_motifs = (args.motif_pwm_path and args.motif_num <= 0 and str(args.motif_pwm_path).endswith(('.meme', '.txt')))
    if auto_motifs:
        # We pass num_motif=None into the model and let it auto-detect.
        # Also compute it here for logging / reproducibility.
        try:
            args.motif_num = infer_motif_num_from_meme(args.motif_pwm_path)
        except Exception:
            pass
    cfg = EnformerConfig(
        sequence_length=args.sequence_length,
        target_length=args.target_length,
        stem_channels=args.stem_channels,
        tower_blocks=args.tower_blocks,
        tower_channels_base=args.tower_channels_base,
        stem_pool=args.stem_pool,
        tower_pool=args.tower_pool,
        transformer_layers=args.transformer_layers,
        transformer_dim=args.transformer_dim,
        transformer_mlp_dim=args.transformer_mlp_dim,
        transformer_heads=args.transformer_heads,
        attn_dropout=args.attn_dropout,
        dropout=args.dropout,
        use_rel_pos_bias=not args.no_rel_pos_bias,
        heads=parse_head_sizes(args.heads),
        use_addition_input=not args.no_addition,
        combine_addition=args.combine_addition,
        model_type=args.model_type,

        # key changes
        num_motif=(None if (args.motif_num is None or args.motif_num <= 0) else args.motif_num),
        motif_include_rc=not args.motif_no_rc,
        motif_bidirectional_except_ctcf=args.motif_bidir_except_ctcf,
        motif_learnable=args.motif_learnable,
        motif_has_bias=not args.motif_no_bias,
        motif_kernel=args.motif_kernel,
        motif_use_prior=args.motif_use_prior,
        motif_pwm_path=args.motif_pwm_path,
        motif_ctcf_regex=args.motif_ctcf_regex,     # <— plumb regex through

        # Significant-motif filtering (z-score vs uniform background)
        motif_keep_significant_only=not args.motif_disable_significant_filter,
        motif_zscore_threshold=args.motif_zscore_threshold,
    )
    return cfg


def build_dataset_from_args(list_file: Optional[str],
                            single_npz: Optional[str],
                            heads: Dict[str, int],
                            apply_log1p: bool = False) -> Dataset:
    """
    Prefer list_file if provided; otherwise fall back to single_npz path.
    """
    if list_file:
        paths = read_list_file(list_file)
        datasets = [MultiHeadNPZDataset(p, heads=heads, apply_log1p=apply_log1p) for p in paths]
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)
    if single_npz:
        return MultiHeadNPZDataset(single_npz, heads=heads, apply_log1p=apply_log1p)
    raise ValueError("Provide either a list file or a single npz path.")

def parse_head_loss_mix(s: Optional[str]) -> Dict[str, float]:
    """
    Parse custom head-specific MSE/Pearson ratios.
    e.g. "gr:0.3,ct:0.7" -> {"gr":0.3, "ct":0.7}
    """
    if not s:
        return {}
    out = {}
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        name, val = part.split(':')
        out[name.strip().lower()] = float(val)
    return out


def main():
    p = argparse.ArgumentParser(description="Train/Val/Test for Enformer (raw vs motif modes, multi-head).")
    p.add_argument('--mode', type=str, choices=['train', 'test'], required=True)

    # Data: allow list files or single npz for backward compatibility
    p.add_argument('--train_list', type=str, help='Text file with training npz paths (one per line)')
    p.add_argument('--val_list', type=str, help='Text file with validation npz paths (one per line)')
    p.add_argument('--test_list', type=str, help='Text file with test npz paths (one per line)')
    p.add_argument('--train_npz', type=str, help='Single training npz (fallback)')
    p.add_argument('--val_npz', type=str, help='Single validation npz (fallback)')
    p.add_argument('--test_npz', type=str, help='Single test npz (fallback)')

    # Output directory
    p.add_argument('--out_dir', type=str, required=True, help='Directory to save checkpoints and test results.')

    # Model config
    p.add_argument('--heads', type=str, required=True, help='e.g. "chip=1,aux3=3"')
    p.add_argument('--model_type', type=str, default='raw-sequence_model', choices=['raw-sequence_model', 'motif-based-model'])
    # Additional inputs (e.g., x_bw1, x_bw2, ...)
    p.add_argument('--combine_addition', type=str, default='concat', choices=['concat', 'sum'],
                   help='How to merge additional per-base input channels with the DNA representation.')
    p.add_argument('--no_addition', action='store_true',
                   help='If set, ignore additional inputs and use only DNA-derived inputs.')

    # Backward compatibility (deprecated)
    p.add_argument('--combine_atac', type=str, default=None, choices=['concat', 'sum'],
                   help='[DEPRECATED] Use --combine_addition')
    p.add_argument('--no_atac', action='store_true',
                   help='[DEPRECATED] Use --no_addition')
    p.add_argument('--head_loss_mix', type=str, default=None,
               help='Comma-separated ratios like "gr:0.3,ct:0.7" meaning loss = r*(MSE/v) + (1-r)*Pearson')

    # Sequence/target geometry
    p.add_argument('--sequence_length', type=int, default=65536)
    p.add_argument('--target_length', type=int, default=1024)

    # Architecture
    p.add_argument('--stem_channels', type=int, default=256)
    p.add_argument('--tower_blocks', type=int, default=6)
    p.add_argument('--tower_channels_base', type=int, default=256)
    p.add_argument('--stem_pool', type=int, default=1)
    p.add_argument('--tower_pool', type=int, default=2)

    p.add_argument('--transformer_layers', type=int, default=8)
    p.add_argument('--transformer_dim', type=int, default=512)
    p.add_argument('--transformer_mlp_dim', type=int, default=2048)
    p.add_argument('--transformer_heads', type=int, default=8)
    p.add_argument('--attn_dropout', type=float, default=0.1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--no_rel_pos_bias', action='store_true')

    # Motif options
    p.add_argument('--motif_num', type=int, default=0,
                help='Number of motifs. Use 0 to auto-detect from MEME; positive integer to force.')
    p.add_argument('--motif_no_rc', action='store_true')
    p.add_argument('--motif_bidir_except_ctcf', action='store_true')
    p.add_argument('--motif_learnable', action='store_true')
    p.add_argument('--motif_no_bias', action='store_true')
    p.add_argument('--motif_kernel', type=int, default=29)

    # Significant motif filtering (z-score vs uniform background)
    p.add_argument('--motif_zscore_threshold', type=float, default=3.0,
                   help='Keep a motif channel only if its max scan score has z >= threshold (vs uniform background).')
    p.add_argument('--motif_disable_significant_filter', action='store_true',
                   help='If set, do NOT zero-out non-significant motif channels.')
    p.add_argument('--motif_use_prior', action='store_true')
    p.add_argument('--motif_pwm_path', type=str, default=None)
    p.add_argument('--motif_ctcf_regex', type=str,
                default=r'(?i)(?<![A-Z0-9])CTCF(?![A-Z0-9])',
                help='Regex to detect the CTCF motif name in MEME (default matches CTCF but not CTCFL).')

    # Training hyperparams
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--head_weights', type=str, default=None, help='e.g. "chip=1.0,aux3=0.5"')
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--amp', action='store_true')

    # Checkpoint
    p.add_argument('--ckpt', type=str, default=None, help='Path to load/save checkpoint. If None, uses out_dir/best.pt')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--auto_fit_geometry', action='store_true',
               help='Auto-reduce pooling so native bins >= target_length')
    p.add_argument('--log1p_targets', action='store_true',
               help='Apply log1p clamp to all target heads in both train and val datasets.')

    args = p.parse_args()

    # ---- Backward compatibility with older flags / datasets ----
    # Older scripts used --no_atac / --combine_atac and fed x_atac_full.
    # New convention is --no_addition / --combine_addition and x_addition_full.
    if getattr(args, 'no_atac', False):
        args.no_addition = True
    if getattr(args, 'combine_atac', None) is not None:
        args.combine_addition = args.combine_atac

    if args.auto_fit_geometry:
    # compute how many effective pool-by-2 steps we can afford
        import math
        seq = args.sequence_length
        T   = args.target_length
        s_stem  = max(1, int(args.stem_pool))
        s_tower = max(1, int(args.tower_pool))

        # total allowable downsample factor so that L' >= T
        max_factor = seq // T if T > 0 else 1
        # always consume stem_pool first
        remain = max(1, max_factor // s_stem)

        # solve s_tower^k <= remain  -> k <= floor(log_base(remain))
        if s_tower > 1:
            k = int(math.floor(math.log(remain, s_tower)))
        else:
            k = args.tower_blocks  # no effect if s_tower==1

        # never exceed requested blocks, but reduce if needed
        args.tower_blocks = min(args.tower_blocks, max(0, k))
        print(f"[auto_fit] adjusted tower_blocks -> {args.tower_blocks} "
            f"(native bins will be >= target_length)")

    os.makedirs(os.path.expanduser(args.out_dir), exist_ok=True)
    out_dir = os.path.expanduser(args.out_dir)

    heads = parse_head_sizes(args.heads)

    running_var = {h: RunningVarEMA(init=1.0, momentum=0.99) for h in heads}

    head_weights = parse_head_weights(args.head_weights, heads)

    head_loss_mix = parse_head_loss_mix(args.head_loss_mix)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = os.path.expanduser(args.ckpt) if args.ckpt else os.path.join(out_dir, 'best.pt')

    if args.mode == 'train':
        # Datasets & loaders
        ds_train = ds_train = build_dataset_from_args(args.train_list, args.train_npz, heads=heads, apply_log1p=args.log1p_targets)
        ds_val   = build_dataset_from_args(args.val_list,   args.val_npz,   heads=heads, apply_log1p=args.log1p_targets)

        dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=2,
                              pin_memory=True, collate_fn=collate_batch, drop_last=False)
        dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=2,
                            pin_memory=True, collate_fn=collate_batch, drop_last=False)

        # Model & optim
        cfg = build_cfg(args)
        model = Enformer(cfg).to(device)

        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scaler = torch.amp.GradScaler('cuda') if args.amp and device.type == 'cuda' else None


        best_val = float('inf')
        start_epoch = 0

        if args.resume and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if 'cfg' in ckpt:
                prev = ckpt['cfg']
                if prev.get('num_motif') != cfg.num_motif or prev.get('motif_pwm_path') != cfg.motif_pwm_path:
                    print("[warn] Current motif config differs from checkpoint. "
                        "If shapes mismatch, training will fail. Consider aligning flags or starting fresh.")

            model.load_state_dict(ckpt['model_state'])
            if 'optim_state' in ckpt: optim.load_state_dict(ckpt['optim_state'])
            if 'scaler_state' in ckpt and scaler is not None: scaler.load_state_dict(ckpt['scaler_state'])
            if 'best_val' in ckpt: best_val = ckpt['best_val']
            if 'epoch' in ckpt: start_epoch = ckpt['epoch'] + 1
            print(f"[resume] loaded checkpoint from {ckpt_path} (epoch {start_epoch})")

        for epoch in range(start_epoch, args.epochs):
            train_loss, train_heads = train_one_epoch(
                model, dl_train, optim, device, head_weights, running_var, scaler, args.grad_clip, head_loss_mix
            )
            val_loss, val_heads = evaluate(
                model, dl_val, device, head_weights, running_var, head_loss_mix
            )

            print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")
            print(f"  train per-head: " + "  ".join([f"{h}={train_heads[h]:.5f}" for h in heads]))
            print(f"  val   per-head: " + "  ".join([f"{h}={val_heads[h]:.5f}" for h in heads]))

            # Save best
            if val_loss < best_val:
                best_val = val_loss
                save_obj = {
                    'model_state': model.state_dict(),
                    'cfg': asdict(cfg),
                    'optim_state': optim.state_dict(),
                    'best_val': best_val,
                    'epoch': epoch,
                }
                if scaler is not None:
                    save_obj['scaler_state'] = scaler.state_dict()
                torch.save(save_obj, ckpt_path)
                print(f"[ckpt] saved best to {ckpt_path} (val_loss={best_val:.5f})")

        print("Training complete. Best val loss:", best_val)

    elif args.mode == 'test':
        ds_test = ds_test = build_dataset_from_args(args.test_list, args.test_npz, heads=heads, apply_log1p=args.log1p_targets)
        dl_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=2,
                            pin_memory=True, collate_fn=collate_batch, drop_last=False)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'cfg' in ckpt:
            cfg = EnformerConfig(**ckpt['cfg'])   # ← exact same config as training
        else:
            # Fallback for very old checkpoints
            cfg = build_cfg(args)

        model = Enformer(cfg).to(device)
        model.load_state_dict(ckpt['model_state'])
        print(f"[test] loaded checkpoint and config from {ckpt_path}")

        metrics, dump_path = run_test(model, dl_test, device, out_dir, heads)


        print("[test] metrics per head:")
        for h, m in metrics.items():
            print(f"  {h}: mse={m['mse']:.6f}  pearson={m['pearson']:.6f}")
        print(f"[test] saved test dump to {dump_path}")
    else:
        raise ValueError("mode must be 'train' or 'test'")

if __name__ == "__main__":
    main()
