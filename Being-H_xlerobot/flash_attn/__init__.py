"""
Compatibility shim for `flash_attn` used by upstream Being-H.

Upstream Being-H imports `from flash_attn import flash_attn_varlen_func` at module import time.

Goals:
- If the real `flash-attn` package is installed, use it (fast CUDA kernels).
- Otherwise provide a pure-PyTorch fallback (SDPA) so training/inference can still run.

Important:
- This file lives under `Being-H_xlerobot/flash_attn/` and therefore appears first on sys.path when you run
  scripts from `Being-H_xlerobot/`. To avoid permanently shadowing the real package, we "extend" the package
  search path and try to import the real implementation from `flash_attn.flash_attn_interface`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

# Make this a "namespace-like" package so `flash_attn.<submodule>` can resolve to
# the real installed flash-attn package when present (e.g. bert_padding, flash_attn_interface).
#
# Without this, running scripts from `Being-H_xlerobot/` would shadow the installed package and make
# upstream try-imports print "FlashAttention2 is not installed." even when it *is* installed.
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)


_REAL_VARLEN_FUNC = None
_REAL_LOOKUP_DONE = False


def _try_import_real_flash_attn():
    """Return real flash_attn_varlen_func if installed, else None."""
    global _REAL_VARLEN_FUNC, _REAL_LOOKUP_DONE  # noqa: PLW0603
    if _REAL_LOOKUP_DONE:
        return _REAL_VARLEN_FUNC

    try:
        # Allow importing submodules from a separately installed `flash_attn` package.
        import pkgutil

        # Extend __path__ to include site-packages/flash_attn if it exists.
        global __path__  # noqa: PLW0603
        __path__ = pkgutil.extend_path(__path__, __name__)

        from flash_attn.flash_attn_interface import flash_attn_varlen_func as real_impl  # type: ignore

        _REAL_VARLEN_FUNC = real_impl
        _REAL_LOOKUP_DONE = True
        # Replace the wrapper with the real implementation to avoid overhead and to expose the true function.
        globals()["flash_attn_varlen_func"] = real_impl  # type: ignore[misc]
        return _REAL_VARLEN_FUNC
    except Exception:
        _REAL_VARLEN_FUNC = None
        _REAL_LOOKUP_DONE = True
        return _REAL_VARLEN_FUNC


def flash_attn_varlen_func(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    causal: bool = False,
    **_: object,
) -> torch.Tensor:
    """
    Drop-in replacement for `flash_attn.flash_attn_varlen_func`.

    Expected shapes (same as flash-attn):
    - q: (total_q, nheads_q, headdim)
    - k/v: (total_k, nheads_k, headdim)  (supports GQA by repeating k/v heads if needed)
    - cu_seqlens_q/k: (batch + 1,) prefix sums, starting with 0

    Returns:
    - out: (total_q, nheads_q, headdim)
    """
    real_impl = _try_import_real_flash_attn()
    if real_impl is not None:
        return real_impl(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"Expected q/k/v rank-3, got q={q.shape}, k={k.shape}, v={v.shape}")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("cu_seqlens_q/cu_seqlens_k must be 1D prefix-sum tensors.")

    # Convert prefix sums to python lists for slicing.
    # (Indexing tensors by python ints keeps everything on-device.)
    cu_q = [int(x) for x in cu_seqlens_q.detach().to("cpu").tolist()]
    cu_k = [int(x) for x in cu_seqlens_k.detach().to("cpu").tolist()]

    if len(cu_q) != len(cu_k):
        raise ValueError(f"Batch size mismatch: cu_seqlens_q={len(cu_q)}, cu_seqlens_k={len(cu_k)}")
    batch = len(cu_q) - 1
    if batch <= 0:
        return q.new_zeros((0, q.shape[1], q.shape[2]))

    hq = int(q.shape[1])
    hk = int(k.shape[1])
    if hk != hq:
        if hk <= 0 or hq % hk != 0:
            raise ValueError(f"GQA head mismatch: q heads={hq}, k/v heads={hk} (not divisible)")
        repeat = hq // hk
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    outs = []
    device = q.device

    # Let PyTorch choose the fastest SDPA backend available.
    # We pass an explicit boolean mask for the causal + prefix case (Lk can be > Lq).
    for i in range(batch):
        qs, qe = cu_q[i], cu_q[i + 1]
        ks, ke = cu_k[i], cu_k[i + 1]
        if qe <= qs:
            continue

        qi = q[qs:qe]  # (Lq, H, D)
        ki = k[ks:ke]  # (Lk, H, D)
        vi = v[ks:ke]

        # SDPA expects (B, H, L, D)
        qi = qi.transpose(0, 1).unsqueeze(0)
        ki = ki.transpose(0, 1).unsqueeze(0)
        vi = vi.transpose(0, 1).unsqueeze(0)

        if causal:
            Lq = int(qi.shape[2])
            Lk = int(ki.shape[2])
            prefix = max(0, Lk - Lq)

            # Mask future positions: mask=True means "do not attend".
            q_idx = torch.arange(Lq, device=device).unsqueeze(1)
            k_idx = torch.arange(Lk, device=device).unsqueeze(0)
            attn_mask = (k_idx > (prefix + q_idx)).unsqueeze(0).unsqueeze(0)  # (1,1,Lq,Lk)
            out = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=None, dropout_p=0.0, is_causal=False)

        outs.append(out.squeeze(0).transpose(0, 1))  # (Lq, H, D)

    if not outs:
        return q.new_zeros((0, q.shape[1], q.shape[2]))
    return torch.cat(outs, dim=0)
