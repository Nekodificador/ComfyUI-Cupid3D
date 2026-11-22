"""Grouped attention over a sparse tensor with torch's own sdpa.

Upstream only implements the windowed and serialized paths with xformers or
flash-attn, both of which need a build matching the installed torch. These two
functions cover the same shapes with scaled_dot_product_attention so the
pipeline runs on a plain torch install.
"""
from typing import List
import torch
from torch.nn.functional import scaled_dot_product_attention

__all__ = ['sdpa_equal_groups', 'sdpa_varlen_groups']


def sdpa_equal_groups(qkv_feats: torch.Tensor) -> torch.Tensor:
    """[B, N, 3, H, C] -> [B, N, H, C], every group the same length."""
    q, k, v = qkv_feats.unbind(dim=2)                                   # [B, N, H, C]
    out = scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2)                                          # [B, N, H, C]


def sdpa_varlen_groups(qkv_feats: torch.Tensor, seq_lens: List[int]) -> torch.Tensor:
    """[M, 3, H, C] split into consecutive groups of `seq_lens` -> [M, H, C].

    Pads every group to the longest one and masks the padded keys. Padded query
    rows keep attending to the real keys, so no row ends up fully masked (sdpa
    answers those with NaN); their garbage output is dropped on the way out.
    """
    _, _, H, C = qkv_feats.shape
    B, N = len(seq_lens), max(seq_lens)
    lens = torch.tensor(seq_lens, device=qkv_feats.device)
    valid = torch.arange(N, device=qkv_feats.device).unsqueeze(0) < lens.unsqueeze(1)    # [B, N]
    flat = valid.reshape(-1)

    padded = qkv_feats.new_zeros(B * N, 3, H, C)
    padded[flat] = qkv_feats                                            # groups are consecutive
    q, k, v = padded.reshape(B, N, 3, H, C).unbind(dim=2)               # [B, N, H, C]
    out = scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=valid[:, None, None, :],
    ).transpose(1, 2)                                                   # [B, N, H, C]
    return out.reshape(B * N, H, C)[flat]


def _demo():
    """Check both paths against a plain per-group softmax attention."""
    torch.manual_seed(0)
    H, C = 4, 16

    def naive(qkv, lens):
        outs, s = [], 0
        for n in lens:
            q, k, v = qkv[s:s + n].unbind(dim=1)                        # [n, H, C]
            w = torch.softmax(torch.einsum('qhc,khc->hqk', q, k) / C ** 0.5, dim=-1)
            outs.append(torch.einsum('hqk,khc->qhc', w, v))
            s += n
        return torch.cat(outs)

    for lens in ([8, 8, 8], [7, 1, 12, 3], [5]):
        qkv = torch.randn(sum(lens), 3, H, C, dtype=torch.float64)
        got = sdpa_varlen_groups(qkv, lens)
        assert torch.allclose(got, naive(qkv, lens), atol=1e-9), f'varlen mismatch for {lens}'

    lens = [6, 6, 6, 6]
    qkv = torch.randn(sum(lens), 3, H, C, dtype=torch.float64)
    got = sdpa_equal_groups(qkv.reshape(len(lens), 6, 3, H, C)).reshape(-1, H, C)
    assert torch.allclose(got, naive(qkv, lens), atol=1e-9), 'equal-length mismatch'
    print('sdpa_groups: both paths match naive softmax attention')


if __name__ == '__main__':
    _demo()
