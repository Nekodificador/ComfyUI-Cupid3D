# Changes to the vendored CUPID

`vendor/Cupid` is [CUPID](https://github.com/cupid3d/Cupid) at commit
`ed7a102`, under the MIT licence kept in `vendor/Cupid/LICENSE`, with the
trainers, datasets and sample assets removed and the changes below applied.
They exist so the model runs on a plain torch install, with no extension to
compile.

| Where | Change |
|---|---|
| `cupid/modules/sparse/attention/sdpa_groups.py` | New file: the grouped attention the windowed and serialized paths need, on torch's own `scaled_dot_product_attention`. Run it as a script for its self-check. |
| `cupid/modules/sparse/attention/{windowed,serialized,full}_attn.py` | An `sdpa` branch alongside the existing backends. |
| `cupid/modules/{sparse,attention}/__init__.py` | `sdpa` is the default backend and a value the environment may select. |
| `vox2seq_pkg/vox2seq/__init__.py` | Falls back to the pure-torch implementation that already ships beside the CUDA extension when that extension was never built. |
| `cupid/representations/mesh/flexicubes/flexicubes.py` | A local shape assertion in place of the one imported from kaolin. FlexiCubes keeps NVIDIA's licence in its own folder. |

Net effect: flash-attn, xformers, kaolin, nvdiffrast, pytorch3d and
diffoctreerast are no longer needed. What is still needed is listed in the
README.
