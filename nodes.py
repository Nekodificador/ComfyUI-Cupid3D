"""Cupid3D: single image -> 3D mesh plus the camera that shot it."""
import logging

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import ComfyExtension, Types, io

from . import cupid_runtime
from .cupid_camera import camera_for_image


def _to_pil_rgba(image: torch.Tensor, mask: torch.Tensor | None) -> Image.Image:
    """IMAGE [H, W, 3] (+ MASK [H, W]) -> RGBA at the original size."""
    rgb = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
    if mask is None:
        return Image.fromarray(rgb, "RGB")
    alpha = (mask.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
    if alpha.shape != rgb.shape[:2]:
        alpha = np.array(Image.fromarray(alpha, "L").resize(
            (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST))
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def _zup_to_yup(v: torch.Tensor) -> torch.Tensor:
    """[..., 3] z-up -> y-up, matching upstream to_glb and core's Trellis2 decode."""
    return torch.stack([v[..., 0], v[..., 2], -v[..., 1]], dim=-1)


class Cupid3DReconstruct(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Cupid3DReconstruct",
            display_name="Cupid3D Reconstruct",
            category="3d/cupid",
            description=(
                "Reconstructs a 3D mesh from one image and estimates the camera that shot it. "
                "The camera is a LOAD3D_CAMERA describing the input image itself, so that image "
                "goes straight in as the background of Preview3D or NKD Preview 3D. Geometry "
                "carries vertex colours, not a baked texture."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Single image. Without a mask the background is "
                                                "removed with rembg, as upstream does."),
                io.Mask.Input("mask", optional=True,
                              tooltip="The subject, white on black. Segmentation nodes give this "
                                      "directly; LoadImage's MASK is the inverse (it is 1 - alpha), "
                                      "so put an InvertMask in between."),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True),
                io.Int.Input("steps", default=25, min=1, max=200,
                             tooltip="Flow-matching steps for both stages."),
                io.Float.Input("cfg_strength", default=5.0, min=0.0, max=20.0, step=0.1),
                io.Boolean.Input("crop_to_content", default=True,
                                 tooltip="Crop to the subject before solving, as upstream does. "
                                         "The camera is corrected back to the full frame."),
                io.Boolean.Input("place_on_ground", default=True,
                                 tooltip="Drop the mesh onto y=0 so it rests on a viewer's floor "
                                         "instead of straddling the origin. The camera moves with "
                                         "it, so it still lines up with the image. Off leaves the "
                                         "mesh in Cupid's own centred frame."),
            ],
            outputs=[
                io.Mesh.Output(display_name="mesh"),
                io.Load3DCamera.Output(display_name="camera_info"),
            ],
        )

    @classmethod
    def execute(cls, image, mask=None, seed=42, steps=25, cfg_strength=5.0,
                crop_to_content=True, place_on_ground=True) -> io.NodeOutput:
        if image.shape[0] > 1:
            logging.info("Cupid3D: using the first of %d images; the node emits one camera.",
                         image.shape[0])
        cupid_runtime.prepare_imports()
        from cupid.utils.sample_utils import pad_to_square

        pil = _to_pil_rgba(image[0], None if mask is None else mask[0])
        width, height = pil.size
        # The model wants a square; the camera is put back on the original frame
        # afterwards, so what comes out of here matches what went in.
        square = pad_to_square(pil)
        side = square.size[0]
        pad = ((side - width) // 2, (side - height) // 2)

        cupid_runtime.free_memory_for_run()
        pipeline = cupid_runtime.get_pipeline()
        pipeline.to(cupid_runtime.compute_device())
        try:
            sampler_params = {"steps": int(steps), "cfg_strength": float(cfg_strength)}
            out = pipeline.run(
                square,
                seed=int(seed),
                formats=["mesh"],
                preprocess_image=crop_to_content,
                sparse_structure_sampler_params=dict(sampler_params),
                slat_sampler_params=dict(sampler_params),
            )
        finally:
            pipeline.to(cupid_runtime.offload_device())
            cupid_runtime.soft_empty_cache()

        mesh_out = out["mesh"][0]
        pose = out["pose"][0]

        vertices = _zup_to_yup(mesh_out.vertices.float())
        lift = (0.0, 0.0, 0.0)
        if place_on_ground:
            # The lowest vertex is a real surface here, not a stray: on the test
            # mesh 50 vertices sit within 0.1% of the height of the minimum.
            dy = -float(vertices[..., 1].min())
            vertices = vertices + vertices.new_tensor([0.0, dy, 0.0])
            lift = (0.0, dy, 0.0)

        # vertex_attrs is [0:3] colour and [3:6] a 0-1 encoded normal: decoded as
        # 2x-1 it agrees with the geometric vertex normals to cos 0.94, raw only 0.29.
        attrs = mesh_out.vertex_attrs.float()
        vertex_colors = attrs[..., :3].clamp(0.0, 1.0).unsqueeze(0).cpu()
        normals = torch.nn.functional.normalize(attrs[..., 3:6] * 2.0 - 1.0, dim=-1)

        mesh = Types.MESH(
            vertices=vertices.unsqueeze(0).cpu(),
            faces=mesh_out.faces.int().unsqueeze(0).cpu(),
            vertex_colors=vertex_colors,
            normals=_zup_to_yup(normals).unsqueeze(0).cpu(),
        )
        camera_info = camera_for_image(
            pose["extrinsic"].cpu().numpy(), pose["intrinsic"].cpu().numpy(),
            size=(width, height), pad=pad, world_offset=lift)
        return io.NodeOutput(mesh, camera_info)


class Cupid3DExtension(ComfyExtension):
    @classmethod
    async def get_node_list(cls) -> list[type[io.ComfyNode]]:
        return [Cupid3DReconstruct]


async def comfy_entrypoint() -> Cupid3DExtension:
    return Cupid3DExtension()
