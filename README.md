# ComfyUI-Cupid3D

One image in, a 3D mesh and the camera that shot it out — a ComfyUI wrapper
around [CUPID](https://github.com/cupid3d/Cupid) (CVPR'26).

Experimental, and a work in progress.

![Cupid3D](example_workflows/Cupid3D%20-%20Image%20to%20Mesh%20and%20Camera.jpg)

## The node

**Cupid3D Reconstruct** — `IMAGE` + optional `MASK`, seed, steps, cfg,
`crop_to_content`, `place_on_ground` → `MESH`, `LOAD3D_CAMERA`.

**The camera describes your input image, exactly as you fed it in**, so the same
image is the background of `Preview3D`, `Load3D` or NKD Preview 3D with no
padding, letterboxing or reframing — and the render lines up with the plate.
Render at the plate's own size and the two cannot disagree.

**`place_on_ground`** (default on) rests the mesh on `y = 0`, the floor in any
viewer, rather than straddling the origin. The camera moves with it, so the
alignment with the photo is untouched.

**`MASK`** is the subject, white on black — what segmentation nodes hand you.
`LoadImage`'s own MASK is `1 - alpha`, so that one needs an `InvertMask` in
between. With no mask at all the background is removed with rembg.

Geometry comes back with **vertex colours and normals**, not a baked texture.
For UVs and a texture, run the mesh through core's `UnwrapMesh` and
`BakeTextureFromViews`.

## Install

Clone into `custom_nodes/` and install, into the Python ComfyUI runs on:

```
pip install spconv-cu126 easydict
pip install "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
```

Also needed:

- **An NVIDIA GPU and the CUDA 12.x toolkit installed.** spconv's prebuilt
  kernels load against it; the node puts `%CUDA_PATH%\bin` on the DLL search
  path for you on Windows.
- **~7 GB of weights** from `hbb1/Cupid`, downloaded on first run, plus DINOv2
  through `torch.hub`.
- About **11.5 GB of VRAM** at peak, measured on a 719×719 input.

No flash-attn, xformers, kaolin, nvdiffrast, pytorch3d or diffoctreerast: the
vendored copy of CUPID under `vendor/` is patched to run on a plain torch
install, and `vendor/PATCHES.md` lists what was changed and why.

Verified on Windows 11, torch 2.10+cu130, an RTX 5090. Upstream is tested on
Linux only.

## Example workflows

All three open from **Templates** in the sidebar — search "Cupid".

Each one loads an image from ComfyUI's own `input/` folder, so copy
`assets/cupid_test_dragon.png` (from upstream's examples) there to run the first
one as it ships, and point Load Image at your own photo for the other two.

**Image to Mesh and Camera** — core nodes only. Reconstructs, then renders the
mesh back through its own predicted camera so you can see the two agree.

**Scene integration with Klein 3 removal** — reconstructs the subject and drops
it back onto the photo with the subject painted out, so it can be moved around
the scene.

**Scene integration advanced (removal + blend)** — the same, plus a second
Klein 3 pass that blends the render into the photograph, and a mask picker to
choose the subject with one click.

The two scene workflows need, beyond this pack, **ComfyUI-NKD-Basic-Tools**,
**ComfyUI-NKD-Klein-Tools**, **ComfyUI-NKD-VFX-Tools**, **RES4LYF** and
**basic_data_handling**; the advanced one also **ComfyUI-NKD-Preview-Tools** and
**ComfyUI-NKD-Notes**. Each one's Note lists its own models. They are examples
of what the camera output is for — the first workflow needs none of them.

## Not implemented

- **No baked texture**, as above.
- **No multi-object scenes.** Upstream's per-object path needs MoGe and
  pytorch3d.
- **No Gaussian or radiance-field output**; both need `diffoctreerast`.
- **One image per run.** Extra batch items are ignored, since the node emits a
  single camera.

## Credits

[CUPID](https://github.com/cupid3d/Cupid) by its authors — the model, the
weights and the bulk of the code under `vendor/`, used under the MIT licence,
whose notice is kept in `vendor/Cupid/LICENSE`. CUPID is itself built on
[TRELLIS](https://github.com/microsoft/TRELLIS). FlexiCubes, under
`vendor/Cupid/cupid/representations/mesh/flexicubes`, keeps NVIDIA's licence in
that folder.

## License

GPL-3.0. See [LICENSE](LICENSE).
