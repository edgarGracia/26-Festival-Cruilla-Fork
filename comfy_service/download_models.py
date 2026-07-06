import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from loguru import logger

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "Comfy" / "models"


@dataclass(frozen=True)
class ModelSpec:
    """A single weight file to place in ComfyUI's models folder.

    Args:
        repo_id (str): Hugging Face repository id.
        remote_file (str): File path inside the repo (may include subfolders).
        dest_subdir (str): ComfyUI models subfolder to place it in
            (e.g. "unet", "clip", "vae", "loras").
        local_name (str | None): Filename to save as. Defaults to the basename
            of ``remote_file``.
        required (bool): Whether the workflow needs this file to run.
    """

    repo_id: str
    remote_file: str
    dest_subdir: str
    local_name: str | None = None
    required: bool = True

    @property
    def filename(self) -> str:
        return self.local_name or Path(self.remote_file).name


MANIFEST: list[ModelSpec] = [
    ModelSpec(
        repo_id="QuantStack/Qwen-Image-Edit-2509-GGUF",
        remote_file="Qwen-Image-Edit-2509-Q8_0.gguf",
        dest_subdir="unet",
    ),
    ModelSpec(
        repo_id="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        remote_file="Qwen2.5-VL-7B-Instruct-Q4_K_S.gguf",
        dest_subdir="clip",
    ),
    ModelSpec(
        repo_id="Comfy-Org/Qwen-Image_ComfyUI",
        remote_file="split_files/vae/qwen_image_vae.safetensors",
        dest_subdir="vae",
    ),
    ModelSpec(
        repo_id="lightx2v/Qwen-Image-Lightning",
        remote_file="Qwen-Image-Lightning-4steps-V1.0.safetensors",
        dest_subdir="loras",
    ),
]


def download_model(spec: ModelSpec, models_dir: Path, token: str | None) -> Path:
    """Download one weight file into the ComfyUI models tree.

    Skips the download when the destination already exists. Uses the Hugging
    Face cache, so a re-run only copies rather than re-fetching.

    Args:
        spec (ModelSpec): The file to fetch.
        models_dir (Path): Root ComfyUI models folder.
        token (str | None): Hugging Face token for gated repos. None for public.

    Raises:
        huggingface_hub.utils.HfHubHTTPError: If the repo or file is missing or
            access is denied.

    Returns:
        Path: The final path of the placed file.
    """
    dest_dir = models_dir / spec.dest_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / spec.filename

    if dest_path.exists():
        logger.info(f"Already present, skipping: {dest_path.relative_to(models_dir)}")
        return dest_path

    logger.info(f"Downloading {spec.repo_id}/{spec.remote_file}")
    cached = hf_hub_download(
        repo_id=spec.repo_id, filename=spec.remote_file, token=token
    )
    shutil.copyfile(cached, dest_path)
    logger.info(f"Placed -> {dest_path.relative_to(models_dir)}")
    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the ComfyUI weights for the 3-ingredients workflow."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"ComfyUI models folder (default: {DEFAULT_MODELS_DIR}).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token for gated repos (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the manifest and exit without downloading.",
    )
    args = parser.parse_args()

    if args.list:
        for spec in MANIFEST:
            logger.info(f"{spec.dest_subdir}/{spec.filename}  <-  {spec.repo_id}")
        return

    logger.info(f"Target models dir: {args.models_dir}")
    for spec in MANIFEST:
        download_model(spec, args.models_dir, args.token)
    logger.info("All weights ready.")


if __name__ == "__main__":
    main()
