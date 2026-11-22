"""Import glue for the vendored Cupid source, plus the pipeline's lifetime.

The tree under vendor/Cupid is upstream ed7a102 with the patches listed in the
README, so it runs on a plain torch install with no flash-attn, xformers,
kaolin or compiled extensions.
"""
import os
import sys
import logging
from pathlib import Path

import comfy.model_management

VENDOR = Path(__file__).parent / "vendor" / "Cupid"
REPO_ID = "hbb1/Cupid"

# The pipeline loads ~7 GB of fp16 weights and peaked at 11.5 GB of VRAM on a
# 719x719 input, so ComfyUI's own models get evicted before a run.
_RUN_VRAM_BUDGET = 13 * 1024**3

_pipeline = None
_imports_ready = False


def prepare_imports():
    """Put the vendored packages on sys.path and make spconv loadable."""
    global _imports_ready
    if _imports_ready:
        return

    # spconv's prebuilt kernels link against the CUDA 12 runtime. cumm only
    # looks in torch/lib on Windows, and torch ships cu13 libs there, so the
    # toolkit's own bin directory goes on the DLL search path first.
    if sys.platform == "win32":
        cuda_bin = Path(os.environ.get("CUDA_PATH", "")) / "bin"
        if cuda_bin.is_dir():
            os.add_dll_directory(str(cuda_bin))
        else:
            logging.warning("Cupid3D: CUDA_PATH does not point at a CUDA toolkit, so spconv will "
                            "fail to load. Install the CUDA 12.x toolkit.")

    os.environ.setdefault("SPCONV_ALGO", "native")
    for path in (VENDOR, VENDOR / "vox2seq_pkg"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    _imports_ready = True


def get_pipeline():
    """Load the pipeline once; it then lives on the offload device between runs."""
    global _pipeline
    if _pipeline is None:
        prepare_imports()
        from cupid.pipelines import Cupid3DPipeline

        logging.info("Cupid3D: loading %s (downloads ~7 GB on first use)", REPO_ID)
        _pipeline = Cupid3DPipeline.from_pretrained(REPO_ID)
        _pipeline.to(offload_device())
    return _pipeline


def free_pipeline():
    global _pipeline
    _pipeline = None
    soft_empty_cache()


def free_memory_for_run():
    """Evict ComfyUI's own models so the pipeline has room."""
    comfy.model_management.free_memory(_RUN_VRAM_BUDGET, compute_device())


def compute_device():
    return comfy.model_management.get_torch_device()


def offload_device():
    return comfy.model_management.unet_offload_device()


def soft_empty_cache():
    comfy.model_management.soft_empty_cache()
