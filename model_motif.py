# model.py
from dataclasses import dataclass
from typing import Dict, Optional

import os
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
"""
usage: #Motif-based mode
cfg = EnformerConfig(
    model_type="motif-based-model",
    sequence_length=65_536, target_length=1_024,
    stem_pool=1, tower_blocks=6,
    use_addition_input=True, combine_addition="concat",
    heads={"chip": 1},

    motif_include_rc=True,
    motif_bidirectional_except_ctcf=False,
    motif_learnable=False,                    # freeze motif conv
    motif_has_bias=True,
    motif_kernel=29,
    motif_use_prior=True,
    motif_pwm_path="/path/to/consensus_pwms.meme",  # local file
)
model = Enformer(cfg)

#Raw sequence (one-hot) mode
cfg = EnformerConfig(
    model_type="raw-sequence_model",
    sequence_length=65_536, target_length=1_024,
    stem_pool=1, tower_blocks=6,               # 64× downsample so 65536 -> 1024
    use_addition_input=True, combine_addition="concat",
    heads={"chip": 1, "aux3": 3},
)
model = Enformer(cfg)
"""


def infer_motif_num_from_meme(meme_path: str) -> int:
    """Infer number of motifs by counting 'MOTIF ' header lines in a MEME file."""
    n = 0
    with open(meme_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # MEME motifs typically start with: "MOTIF <name>"
            if line.startswith("MOTIF "):
                n += 1
    if n <= 0:
        raise ValueError(f"No motifs found in MEME file: {meme_path}")
    return n


# -------------------------------
# Config
# -------------------------------
@dataclass
class EnformerConfig:
    sequence_length: int = 65_536
    target_length: int = 1_024

    # Channels & depth
    stem_channels: int = 256
    tower_blocks: int = 6
    tower_channels_base: int = 256  # grows with depth

    # Pooling
    stem_pool: int = 2
    tower_pool: int = 2  # applied per block

    # Transformer
    transformer_layers: int = 8
    transformer_dim: int = 512
    transformer_mlp_dim: int = 2048
    transformer_heads: int = 8
    attn_dropout: float = 0.1
    dropout: float = 0.1
    use_rel_pos_bias: bool = True

    # Heads
    heads: Dict[str, int] = None  # e.g., {"human": 5313}

    # ---- Additional input controls (e.g., x_bw1, x_bw2, ...) ----
    # If True, expect additional per-base input channels (e.g. concatenated bigWig signals).
    # If False, model uses only the DNA-derived representation (raw one-hot or motif map).
    use_addition_input: bool = True
    combine_addition: str = "concat"  # "concat" (default) or "sum"

    # ---- New: input mode ----
    # "raw-sequence_model" (one-hot DNA) or "motif-based-model" (scan motifs first)
    model_type: str = "raw-sequence_model"

    # ---- New: motif scanning options ----
    num_motif: Optional[int] = None
    motif_include_rc: bool = True
    motif_bidirectional_except_ctcf: bool = False  # keep CTCF fwd/rev separate
    motif_learnable: bool = False
    motif_has_bias: bool = True
    motif_kernel: int = 29

    # ---- New: keep only significant motifs via z-score vs uniform background ----
    # If True, motif channels whose max score is NOT significant are zeroed out entirely.
    motif_keep_significant_only: bool = True
    # Z-score threshold for significance (computed vs uniform background, per motif channel).
    motif_zscore_threshold: float = 3.0

    # ---- New: motif prior (MEME or tensor) ----
    motif_use_prior: bool = False
    motif_pwm_path: Optional[str] = None  # local path to .meme / .npy / .pt
    # Regex to detect CTCF motif name in MEME (default matches CTCF but not CTCFL)
    motif_ctcf_regex: str = r'(?i)(?<![A-Z0-9])CTCF(?![A-Z0-9])'

    def __post_init__(self):
        # Auto-infer motif count from MEME file if available and not explicitly provided.
        if (self.num_motif is None or self.num_motif <= 0) and self.motif_pwm_path:
            if str(self.motif_pwm_path).endswith(('.meme', '.txt')) and os.path.exists(self.motif_pwm_path):
                self.num_motif = infer_motif_num_from_meme(self.motif_pwm_path)


# -------------------------------
# Utils
# -------------------------------
class OneHotDNA(nn.Module):
    """Convert integer-encoded bases {0:A,1:C,2:G,3:T} to one-hot.
    If input already looks like one-hot, pass through.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B,L] long/int, [B,L,4] float, or [B,4,L]
        if x.dim() == 2 and x.dtype in (torch.long, torch.int64, torch.int32):
            out = F.one_hot(x.long(), num_classes=4).float()  # [B,L,4]
            return out
        if x.dim() == 3:
            if x.shape[-1] == 4:  # [B,L,4]
                return x
            if x.shape[1] == 4:   # [B,4,L] -> [B,L,4]
                return x.transpose(1, 2)
        raise ValueError("Expected input as [B,L] (ints), [B,L,4], or [B,4,L]")


class ConvBlock1D(nn.Module):
    """Conv → GELU → (Dropout) → Conv → GELU with residual (channels preserved)."""
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,L]
        residual = x
        y = self.norm(x)
        y = self.act(self.conv1(y))
        y = self.drop(y)
        y = self.act(self.conv2(y))
        return y + residual


class Unfold1d(nn.Module):
    """1D Unfold via 2D Unfold on a singleton height dimension."""
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(1, stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,L]
        B, C, L = x.shape
        x2d = x.unsqueeze(2)  # [B,C,1,L]
        patches = self.unfold(x2d)  # [B, C*kernel, L_out]
        K = self.kernel_size
        L_out = patches.shape[-1]
        return patches.view(B, C, K, L_out)  # [B,C,K,L_out]


class SoftmaxPool1d(nn.Module):
    """Content-aware pooling with learned logits per position inside each window."""
    def __init__(self, channels: int, kernel_size: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.score = nn.Conv1d(channels, 1, kernel_size=1)
        self.unfold = Unfold1d(kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,L]
        if x.size(-1) % self.kernel_size != 0:
            L_trim = x.size(-1) - (x.size(-1) % self.kernel_size)
            x = x[..., :L_trim]
        patches = self.unfold(x)                        # [B,C,K,L']
        logits = self.unfold(self.score(x)).squeeze(1)  # [B,K,L']
        attn = torch.softmax(logits, dim=1).unsqueeze(1)  # [B,1,K,L']
        return (attn * patches).sum(dim=2)              # [B,C,L']


class TargetCenterCrop(nn.Module):
    def __init__(self, target_length: int):
        super().__init__()
        self.target_length = target_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,C] or [B,C,L] — crop along the length axis
        if x.dim() != 3:
            raise ValueError("Expected a 3D tensor for cropping")
        length_dim = 1 if x.shape[1] >= x.shape[2] else 2
        L = x.shape[length_dim]
        T = self.target_length
        if T > L:
            raise ValueError(f"target_length {T} > current length {L}")
        start = (L - T) // 2
        end = start + T
        return x[:, start:end, :] if length_dim == 1 else x[:, :, start:end]


# -------------------------------
# Transformer with relative position bias
# -------------------------------
class RelativePositionBias1D(nn.Module):
    def __init__(self, num_heads: int, max_dist: int):
        super().__init__()
        self.num_heads = num_heads
        size = 2 * max_dist + 1
        self.bias = nn.Parameter(torch.zeros(num_heads, size))

    def forward(self, qlen: int, klen: int) -> torch.Tensor:
        # Returns [num_heads, qlen, klen]
        device = self.bias.device
        q_idx = torch.arange(qlen, device=device)[:, None]
        k_idx = torch.arange(klen, device=device)[None, :]
        rel = (k_idx - q_idx).clamp(-self.bias.size(1)//2, self.bias.size(1)//2) + self.bias.size(1)//2
        return self.bias[:, rel]


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.1,
                 attn_dropout: float = 0.1, rel_pos: Optional[RelativePositionBias1D] = None):
        super().__init__()
        self.rel_pos = rel_pos
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                          dropout=attn_dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,D]
        residual = x
        x = self.ln1(x)
        attn_bias = None
        if self.rel_pos is not None:
            B, L, _ = x.shape
            bias = self.rel_pos(L, L)             # [H, L, L]
            attn_bias = bias.repeat(B, 1, 1)      # [B*H, L, L]
        y, _ = self.attn(x, x, x, attn_mask=attn_bias)
        x = residual + y
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, layers: int, dim: int, heads: int, mlp_dim: int,
                 dropout: float, attn_dropout: float, use_rel_pos: bool):
        super().__init__()
        rel = RelativePositionBias1D(heads, max_dist=4096) if use_rel_pos else None
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout, attn_dropout, rel)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# -------------------------------
# Motif Scanner (with optional MEME/tensor prior)
# -------------------------------
class MotifScanner(nn.Module):
    """
    Motif scanner with optional PWM prior initialization from MEME or a tensor file.
    - Input: one-hot DNA as [B,L,4] or [B,4,L]
    - Output: [B,L,M_eff] where M_eff = M (+1 if bidir_except_ctcf True)
    """
    def __init__(self,
                 num_motif: Optional[int] = None,
                 include_rc: bool = True,
                 bidir_except_ctcf: bool = False,
                 kernel: int = 29,
                 learnable: bool = False,
                 has_bias: bool = True,
                 keep_significant_only: bool = False,
                 zscore_threshold: float = 3.0,
                 use_prior: bool = False,
                 pwm_path: Optional[str] = None,
                 ctcf_regex: str = r"(?i)(?<![A-Z0-9])CTCF(?![A-Z0-9])"):
        super().__init__()
        self.include_rc = include_rc
        self.bidir_except_ctcf = bidir_except_ctcf
        self.kernel = kernel
        self.use_prior = use_prior
        self.pwm_path = pwm_path
        self.ctcf_regex = re.compile(ctcf_regex)
        self.ctcf_idx: Optional[int] = None  # determined after parsing names
        self.motif_names: Optional[list] = None  # post-fusion names set on first forward

        # --- Significance gating ---
        # If enabled, we compute a per-motif z-score vs a uniform background (A/C/G/T=0.25)
        # using the *linear* motif score (pre-ReLU). If a motif is NOT significant for a given
        # sequence, we zero that entire motif channel across all positions.
        self.keep_significant_only = bool(keep_significant_only)
        self.zscore_threshold = float(zscore_threshold)

        # ---- Determine kernels & names (and num_motif) BEFORE building conv ----
        kernels: Optional[torch.Tensor] = None   # [M,4,K]
        names: Optional[list] = None

        if self.use_prior:
            if pwm_path is None:
                raise ValueError("motif_pwm_path must be provided when motif_use_prior=True")

            # If MEME: parse ALL motifs to get M automatically (unless num_motif is explicitly set)
            if pwm_path.endswith(('.meme', '.txt')):
                kernels_all, names_all = self._parse_meme_all_kernels_with_names(pwm_path, K=self.kernel)
                if num_motif is not None and num_motif > 0:
                    kernels_all = kernels_all[:num_motif]
                    names_all = names_all[:num_motif]
                kernels, names = kernels_all, names_all
                num_motif = kernels.shape[0]  # auto-detected
            elif pwm_path.endswith('.npy') or pwm_path.endswith(('.pt', '.pth')):
                if num_motif is None or num_motif <= 0:
                    raise ValueError("Please provide num_motif when using .npy/.pt priors")
                kernels = self._load_pwms_as_kernels(num_motif, self.kernel, pwm_path)  # validates shape
                names = [f"motif_{i}" for i in range(num_motif)]
            else:
                raise ValueError(f"Unrecognized prior format for '{pwm_path}'. Expected .meme/.txt/.npy/.pt")
        else:
            # No prior: need num_motif; generic names; kernels will be random init
            # If a MEME file is provided, we can still auto-infer the motif count (and names)
            # even though we are not initializing weights from the prior.
            if (num_motif is None or num_motif <= 0) and pwm_path and pwm_path.endswith(('.meme', '.txt')):
                try:
                    _, names_all = self._parse_meme_all_kernels_with_names(pwm_path, K=self.kernel)
                    names = names_all
                    num_motif = len(names_all)
                except Exception:
                    num_motif = infer_motif_num_from_meme(pwm_path)
                    names = [f"motif_{i}" for i in range(num_motif)]

            if num_motif is None or num_motif <= 0:
                raise ValueError("num_motif must be provided when motif_use_prior=False (or provide a .meme/.txt file to auto-infer)")

            if names is None:
                names = [f"motif_{i}" for i in range(num_motif)]

        # At this point, num_motif and names are known
        self.num_motif = int(num_motif)  # type: ignore[arg-type]

        # Build conv with correct output channels
        conv_out = self.num_motif * (2 if include_rc else 1)
        self.conv = nn.Conv1d(4, conv_out, self.kernel, padding=self.kernel // 2, bias=has_bias)
        self.act = nn.ReLU()

        # Initialize with prior if provided
        if self.use_prior and kernels is not None:
            if include_rc:
                rc = self._reverse_complement_kernels(kernels)      # [M,4,K]
                kernels_full = torch.cat([kernels, rc], dim=0)      # [2M,4,K]
                raw_names = names + [f"{n}_RC" for n in names]
            else:
                kernels_full = kernels
                raw_names = names

            if self.conv.weight.shape != kernels_full.shape:
                raise ValueError(
                    f"Conv weight shape {tuple(self.conv.weight.shape)} != kernels {tuple(kernels_full.shape)}"
                )
            with torch.no_grad():
                self.conv.weight.copy_(kernels_full)

            self._raw_names = raw_names  # names BEFORE fusion logic
        else:
            # No prior: random init, keep generic names
            base = [f"motif_{i}" for i in range(self.num_motif)]
            self._raw_names = base + ([f"motif_{i}_RC" for i in range(self.num_motif)] if include_rc else [])

        # Detect CTCF index (in the *forward* names list, which will drop _RC suffix)
        # We search the original forward names to identify the index before fusion.
        # For MEME, names are as given; for npy/pt/generic they are "motif_i".
        self._forward_names = [n.replace("_RC", "") for n in self._raw_names[:self.num_motif]]
        for i, n in enumerate(self._forward_names):
            if self.ctcf_regex.search(n):
                self.ctcf_idx = i
                break  # first match

        if not learnable:
            for p in self.conv.parameters():
                p.requires_grad = False

        # Precompute background mean/std for the EFFECTIVE output channels after fusion logic.
        # These are stored as buffers for fast z-score computation during forward.
        with torch.no_grad():
            bg_mean, bg_std = self._compute_uniform_bg_stats_for_effective_channels()
        self.register_buffer("bg_mean", bg_mean)  # [M_eff]
        self.register_buffer("bg_std", bg_std)    # [M_eff]

    def forward(self, x_onehot: torch.Tensor) -> torch.Tensor:
        if x_onehot.dim() != 3:
            raise ValueError("MotifScanner expects 3D input")
        # to [B,4,L]
        x = x_onehot if x_onehot.shape[1] == 4 else x_onehot.transpose(1, 2)
        # Compute linear + ReLU outputs so we can do significance gating on the linear score.
        y_lin = self.conv(x)        # [B, M*(1 or 2), L]
        y_relu = self.act(y_lin)    # [B, M*(1 or 2), L]

        lin_eff = self._fuse_effective_channels(y_lin)   # [B, M_eff, L]
        out_eff = self._fuse_effective_channels(y_relu)  # [B, M_eff, L]

        # Optionally keep only significant motifs (per sequence, per motif channel).
        if self.keep_significant_only:
            # z-score vs uniform background (A/C/G/T=0.25), using max linear score across positions.
            max_lin = lin_eff.max(dim=2).values  # [B, M_eff]
            z = (max_lin - self.bg_mean) / (self.bg_std + 1e-6)
            keep = (z >= self.zscore_threshold).to(out_eff.dtype)  # [B, M_eff]
            out_eff = out_eff * keep.unsqueeze(-1)

        return out_eff.transpose(1, 2)  # [B, L, M_eff]

    # ---- helpers ----
    @staticmethod
    def _reverse_complement_kernels(w: torch.Tensor) -> torch.Tensor:
        """
        Reverse complement a kernel bank shaped [M,4,K]:
        - reverse along K
        - swap channels A<->T and C<->G
        """
        A, C, G, T = 0, 1, 2, 3
        swap = torch.tensor([T, G, C, A], dtype=torch.long, device=w.device)
        w_rc = w.index_select(1, swap)
        w_rc = torch.flip(w_rc, dims=[2])
        return w_rc

    def _fuse_effective_channels(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the same RC-fusion logic as the original forward, but to an arbitrary score tensor.

        Args:
            y: [B, M*(1 or 2), L] (linear or activated)
        Returns:
            out: [B, M_eff, L]
        """
        if y.dim() != 3:
            raise ValueError("Expected y as [B,C,L]")

        if self.include_rc:
            fwd = y[:, :self.num_motif, :]
            rev = y[:, self.num_motif:, :]

            if self.bidir_except_ctcf and (self.ctcf_idx is not None):
                fused = fwd + rev
                # Keep CTCF separate (fwd in place, append rev)
                fused[:, self.ctcf_idx, :] = fwd[:, self.ctcf_idx, :]
                ctcf_rev = rev[:, self.ctcf_idx:self.ctcf_idx + 1, :]
                out = torch.cat([fused, ctcf_rev], dim=1)  # [B, M+1, L]

                # Set names once
                if self.motif_names is None:
                    names = self._forward_names.copy()
                    names[self.ctcf_idx] = f"{names[self.ctcf_idx]}_fwd"
                    names = names + [f"{self._forward_names[self.ctcf_idx]}_rev"]
                    self.motif_names = names
                return out

            # Standard fusion
            out = fwd + rev
            if self.motif_names is None:
                self.motif_names = self._forward_names
            return out

        # No RC
        if self.motif_names is None:
            self.motif_names = self._forward_names
        return y

    def _compute_uniform_bg_stats_for_effective_channels(self):
        """Compute mean/std per EFFECTIVE motif channel under a uniform background.

        Background model: independent bases with P(A)=P(C)=P(G)=P(T)=0.25.
        We compute the distribution of the *linear* conv score at a position:
            s = sum_k w[base_k, k] + b
        Mean/variance at a position are computed analytically and then used for z-scoring
        the max score across positions.

        Returns:
            (mean, std): each [M_eff]
        """
        w = self.conv.weight.detach()  # [C_out, 4, K]
        b = self.conv.bias.detach() if (self.conv.bias is not None) else None  # [C_out]

        # Split fwd/rev if RC is enabled
        if self.include_rc:
            w_fwd = w[:self.num_motif]
            w_rev = w[self.num_motif:]
            b_fwd = b[:self.num_motif] if b is not None else None
            b_rev = b[self.num_motif:] if b is not None else None

            # Default effective kernels = fwd + rev
            w_eff = w_fwd + w_rev
            b_eff = (b_fwd + b_rev) if (b is not None) else None

            if self.bidir_except_ctcf and (self.ctcf_idx is not None):
                # For CTCF channel, keep fwd only; append rev-only channel
                idx = int(self.ctcf_idx)
                w_eff = w_eff.clone()
                w_eff[idx] = w_fwd[idx]
                if b_eff is not None:
                    b_eff = b_eff.clone()
                    b_eff[idx] = b_fwd[idx]

                w_eff = torch.cat([w_eff, w_rev[idx:idx + 1]], dim=0)
                if b_eff is not None:
                    b_eff = torch.cat([b_eff, b_rev[idx:idx + 1]], dim=0)
        else:
            w_eff = w
            b_eff = b

        # Analytic mean/var per channel at a single position.
        # For each k: X_k takes one of 4 weights uniformly.
        mean_k = w_eff.mean(dim=1)                      # [M_eff, K]
        mean = mean_k.sum(dim=1)                        # [M_eff]
        ex2_k = (w_eff ** 2).mean(dim=1)                # [M_eff, K]
        var_k = (ex2_k - mean_k ** 2).clamp_min(0.0)    # [M_eff, K]
        var = var_k.sum(dim=1)                          # [M_eff]

        if b_eff is not None:
            mean = mean + b_eff

        std = torch.sqrt(var + 1e-6)
        return mean, std

    def _load_pwms_as_kernels(self, M: int, K: int, path: str) -> torch.Tensor:
        """
        Returns a tensor [M,4,K] suitable for Conv1d.weight initialization.
        Supports:
          - MEME text file (A/C/G/T matrices)
          - .npy / .pt with shape [M,4,K]
        """
        if path is None:
            raise ValueError("motif_pwm_path must be provided when motif_use_prior=True")

        if path.endswith(".npy"):
            arr = torch.tensor(np.load(path), dtype=torch.float32)
            return self._validate_kernels(arr, M, K)
        if path.endswith(".pt") or path.endswith(".pth"):
            arr = torch.load(path, map_location="cpu")
            if isinstance(arr, dict) and "kernels" in arr:
                arr = arr["kernels"]
            arr = torch.tensor(arr, dtype=torch.float32) if not isinstance(arr, torch.Tensor) else arr.float()
            return self._validate_kernels(arr, M, K)

        # Otherwise treat as MEME
        return self._parse_meme_to_kernels(path, M, K)

    @staticmethod
    def _validate_kernels(t: torch.Tensor, M: int, K: int) -> torch.Tensor:
        if t.dim() != 3 or t.shape[0] != M or t.shape[1] != 4 or t.shape[2] != K:
            raise ValueError(f"Expected kernels shape [M,4,K]=[{M},4,{K}], got {tuple(t.shape)}")
        return t

    # New: parse ALL motifs in MEME (no M required), return [M,4,K], names
    def _parse_meme_all_kernels_with_names(self, path: str, K: int):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r") as f:
            text = f.read()

        parts = re.split(r"(?m)^\s*MOTIF\s+([^\s]+).*?$", text)
        names_all, blocks_all = [], []
        it = iter(parts[1:])
        for name, block in zip(it, it):
            names_all.append(str(name))
            blocks_all.append(block)

        kernels = []
        names = []
        for name, blk in zip(names_all, blocks_all):
            m = re.search(
                r"letter-probability matrix[^\n]*\n((?:\s*[\d.eE+-]+\s+){3}[\d.eE+-]+\s*\n)+",
                blk,
            )
            if not m:
                continue

            rows = re.findall(
                r"^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$",
                m.group(0), flags=re.MULTILINE
            )
            if not rows:
                continue

            mat = torch.tensor(
                [[float(a), float(c), float(g), float(t)] for (a, c, g, t) in rows],
                dtype=torch.float32
            )  # [K_block, 4]
            mat = mat.transpose(0, 1)  # [4, K_block]

            # pad/trim to target K
            if mat.shape[1] < K:
                mat = F.pad(mat, (0, K - mat.shape[1]))
            elif mat.shape[1] > K:
                mat = mat[:, :K]

            # convert probs → (approx) log-odds vs bg=0.25 and zero-center
            if torch.all((mat >= 0) & (mat <= 1)):
                bg = torch.full_like(mat, 0.25)
                mat = torch.log((mat + 1e-6) / (bg + 1e-6))
                mat = mat - mat.mean(dim=0, keepdim=True)

            kernels.append(mat)
            names.append(name)

        if len(kernels) == 0:
            raise ValueError("No usable motifs found in MEME file")

        return torch.stack(kernels, dim=0), names

    # Backward-compat shims (used elsewhere in your code)
    def _parse_meme_to_kernels_with_names(self, path: str, M: int, K: int):
        kernels, names = self._parse_meme_all_kernels_with_names(path, K)
        if len(kernels) < M:
            raise ValueError(f"MEME file contained only {len(kernels)} motifs; need {M}")
        return kernels[:M], names[:M]

    def _parse_meme_to_kernels(self, path: str, M: int, K: int):
        kernels, _ = self._parse_meme_to_kernels_with_names(path, M, K)
        return kernels

# -------------------------------
# Enformer with optional ATAC input and motif mode
# -------------------------------
class Enformer(nn.Module):
    def __init__(self, cfg: EnformerConfig):
        super().__init__()
        self.cfg = cfg
        heads_cfg = cfg.heads or {"default": 10}

        # ---- Input mapping (raw or motif-based) ----
        self.onehot = OneHotDNA()
        self.model_type = cfg.model_type

        # Stem input conv: fixed for raw (4->Cs) or Lazy for motif (M_eff->Cs)
        if self.model_type == "motif-based-model":
            self.motif_scanner = MotifScanner(
                num_motif=cfg.num_motif,
                include_rc=cfg.motif_include_rc,
                bidir_except_ctcf=cfg.motif_bidirectional_except_ctcf,
                kernel=cfg.motif_kernel,
                learnable=cfg.motif_learnable,
                has_bias=cfg.motif_has_bias,
                keep_significant_only=cfg.motif_keep_significant_only,
                zscore_threshold=cfg.motif_zscore_threshold,
                use_prior=cfg.motif_use_prior,
                pwm_path=cfg.motif_pwm_path,
                ctcf_regex=cfg.motif_ctcf_regex,
            )
            self.stem_conv = nn.LazyConv1d(cfg.stem_channels, kernel_size=15, padding=7)
        else:
            self.motif_scanner = None
            self.stem_conv = nn.Conv1d(4, cfg.stem_channels, kernel_size=15, padding=7)

        self.stem_block = ConvBlock1D(cfg.stem_channels, kernel_size=5, dropout=cfg.dropout)
        self.stem_pool = SoftmaxPool1d(cfg.stem_channels, kernel_size=cfg.stem_pool)

        # Convolutional tower
        tower = []
        channels = cfg.tower_channels_base
        self.stem_proj = nn.Conv1d(cfg.stem_channels, channels, kernel_size=1)
        for i in range(cfg.tower_blocks):
            tower.append(ConvBlock1D(channels, kernel_size=5, dropout=cfg.dropout))
            tower.append(SoftmaxPool1d(channels, kernel_size=cfg.tower_pool))
            if (i + 1) % 2 == 0:
                new_channels = channels * 2
                tower.append(nn.Conv1d(channels, new_channels, kernel_size=1))
                channels = new_channels
        self.tower = nn.Sequential(*tower)
        self._tower_out_channels = channels

        # Additional input projection (lazy in_channels) ----
        # This is intended for per-base signals like x_bw1, x_bw2, ... packed as [B,A,L].
        self.use_addition = cfg.use_addition_input
        self.combine_addition = cfg.combine_addition
        if self.use_addition:
            # Project additional channels [B,A,L'] to match tower channels [B,Ct,L']
            self.addition_proj = nn.LazyConv1d(out_channels=self._tower_out_channels, kernel_size=1)

        # Transformer trunk (input dims depend on concat with ATAC)
        in_ch = self._tower_out_channels * (2 if (self.use_addition and self.combine_addition == "concat") else 1)
        self.pre_xfmr = nn.Conv1d(in_ch, cfg.transformer_dim, kernel_size=1)
        self.transformer = TransformerStack(
            layers=cfg.transformer_layers,
            dim=cfg.transformer_dim,
            heads=cfg.transformer_heads,
            mlp_dim=cfg.transformer_mlp_dim,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
            use_rel_pos=cfg.use_rel_pos_bias,
        )
        self.target_crop = TargetCenterCrop(cfg.target_length)

        # Pointwise + Heads
        self.pointwise = nn.Sequential(
            nn.LayerNorm(cfg.transformer_dim),
            nn.Linear(cfg.transformer_dim, 2 * cfg.transformer_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(2 * cfg.transformer_dim, n),
                nn.Softplus()
            )
            for name, n in heads_cfg.items()
        })

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _format_addition(x_addition: torch.Tensor, L_out: int) -> torch.Tensor:
        """Ensure additional inputs have shape [B, A, L_out].

        Accepts [B,A,L] or [B,L,A] and interpolates to match L_out if needed.
        """
        if x_addition.dim() != 3:
            raise ValueError("x_addition_full must be 3D: [B,A,L] or [B,L,A]")
        B, d1, d2 = x_addition.shape
        # Try to bring to [B,A,L]
        if d1 <= 8 and d2 > 8:  # heuristic: small A (1-8 channels), large length
            addition = x_addition  # [B,A,L]
        elif d2 <= 8 and d1 > 8:
            addition = x_addition.transpose(1, 2)  # [B,A,L]
        else:
            # ambiguous, assume last dim is length if larger
            addition = x_addition if d2 >= d1 else x_addition.transpose(1, 2)
        # Interpolate length if mismatch
        if addition.size(-1) != L_out:
            addition = F.interpolate(addition, size=L_out, mode='linear', align_corners=False)
        return addition

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(
        self,
        x: torch.Tensor,
        x_addition_full: Optional[torch.Tensor] = None,
        head: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args
        x:            DNA as [B,L,4], [B,4,L], or [B,L] (ints 0..3)
        x_addition_full:  Additional per-base inputs as [B,A,L] or [B,L,A]; interpolated to match downsampled length.
                          This should contain your x_bw1, x_bw2, ... channels stacked along A.
        head:         optional head name to return only that head
        Returns: dict from head name -> [B, T, n_features]
        """
        # ---- Input branch (raw vs motif) ----
        x = self.onehot(x)  # [B,L,4] or [B,4,L]
        if self.model_type == "motif-based-model":
            # scan motifs -> [B,L,M_eff], then to [B,M_eff,L]
            x = self.motif_scanner(x)     # [B,L,M_eff]
            x = x.transpose(1, 2)         # [B,M_eff,L]
        else:
            # raw DNA -> ensure [B,4,L]
            if x.shape[-1] == 4:
                x = x.transpose(1, 2)     # [B,4,L]

        # ---- Common stem/tower ----
        x = self.stem_conv(x)              # Conv1d or LazyConv1d
        x = self.stem_block(x)
        x = self.stem_pool(x)              # [B,Cs,L/2]
        x = self.stem_proj(x)              # [B,Ct,L/2]
        x = self.tower(x)                  # [B,Ct,L']
        L_prime = x.size(-1)

        # ---- Additional input integration (optional) ----
        if self.use_addition:
            if x_addition_full is None:
                raise ValueError("Model configured with use_addition_input=True but x_addition_full=None")
            addition = self._format_addition(x_addition_full, L_prime)  # [B,A,L']
            addition = self.addition_proj(addition)                     # [B,Ct,L']
            if self.combine_addition == "sum":
                x = x + addition
            else:  # concat
                x = torch.cat([x, addition], dim=1)                     # [B,2*Ct,L']

        # ---- Transformer trunk ----
        x = self.pre_xfmr(x).transpose(1, 2)   # [B,L',D]
        x = self.transformer(x)                # [B,L',D]

        # ---- Crop to target bins ----
        x = self.target_crop(x)                # [B,T,D]

        # ---- Pointwise & heads ----
        h = self.pointwise(x)                  # [B,T,2D]
        if head is not None:
            if head not in self.heads:
                raise KeyError(f"Head '{head}' not found. Available: {list(self.heads.keys())}")
            return {head: self.heads[head](h)}
        else:
            return {name: proj(h) for name, proj in self.heads.items()}


    @torch.no_grad()
    def extract_motifmap(self, x: torch.Tensor):
        if self.model_type != "motif-based-model":
            raise RuntimeError("extract_motifmap requires cfg.model_type='motif-based-model'")
        x_oh = self.onehot(x)                 # [B,L,4] (handles ints / [B,4,L])
        feats = self.motif_scanner(x_oh)      # [B,L,M_eff] (sets self.motif_scanner.motif_names)
        names = self.motif_scanner.motif_names
        return feats, names
