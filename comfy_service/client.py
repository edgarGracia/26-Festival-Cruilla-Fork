import copy
import json
import random
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import requests
import websocket
from loguru import logger
from PIL import Image


@dataclass
class ComfyServerConfig:
    """Connection details for a running ComfyUI server.

    Args:
        host (str): Hostname or IP where ComfyUI listens.
        port (int): HTTP/websocket port (ComfyUI default is 8188).
        timeout (float): Seconds to wait for a single generation before
            giving up.
    """

    host: str = "127.0.0.1"
    port: int = 8188
    timeout: float = 600.0

    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws"


@dataclass
class NodeMap:
    """Maps semantic inputs to node ids in the API-format workflow.

    The defaults match ``3ingredients API.json``. Re-export the workflow from
    ComfyUI and update these ids if the graph changes.

    Args:
        base_image (str): LoadImage node for Image 1 (the base scene).
        person_image (str): LoadImage node for Image 2 (the person).
        object_image (str): LoadImage node for Image 3 (the object).
        prompt (str): Node holding the edit instruction.
        prompt_field (str): Widget name on the prompt node to overwrite.
        seed (str): KSampler node whose seed is randomised per request.
    """

    base_image: str = "30"
    person_image: str = "31"
    object_image: str = "32"
    prompt: str = "25"
    prompt_field: str = "prompt"
    seed: str = "35"


@dataclass
class GenerationRequest:
    """A single image-edit job for the 3-ingredients workflow.

    Args:
        base_image (Path): Base scene photo (Image 1).
        person_image (Path): Person to insert (Image 2).
        object_image (Path): Object to add (Image 3).
        prompt (str | None): Edit instruction. When None, the prompt already
            baked into the workflow is kept.
        seed (int | None): KSampler seed. When None, a random seed is used.
    """

    base_image: Path
    person_image: Path
    object_image: Path
    prompt: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.base_image = Path(self.base_image)
        self.person_image = Path(self.person_image)
        self.object_image = Path(self.object_image)


class ComfyClient:
    """Drives a ComfyUI server from an API-format workflow over HTTP/websocket.

    The client uploads the input images, injects the request values into a copy
    of the workflow, queues it, waits for completion on the websocket, then
    downloads the produced image. It never imports ComfyUI, so the app stays
    decoupled from Comfy's dependencies.

    Args:
        workflow_path (str | Path): Path to the API-format workflow JSON
            (the ``... API.json`` export, not the UI workflow).
        config (ComfyServerConfig, optional): Server connection details.
        node_map (NodeMap, optional): Node-id mapping for input injection.

    Raises:
        FileNotFoundError: If the workflow file does not exist.

    Example:
        >>> client = ComfyClient("3ingredients API.json")
        >>> image = client.generate(GenerationRequest(
        ...     base_image="inputs/ca/bg_urban.png",
        ...     person_image="user_image.png",
        ...     object_image="inputs/blue_sunglasses.png",
        ...     prompt="Place the person into the scene wearing the sunglasses.",
        ... ))
        >>> image.save("outputs/result.png")
    """

    def __init__(
        self,
        workflow_path: str | Path,
        config: ComfyServerConfig | None = None,
        node_map: NodeMap | None = None,
    ) -> None:
        self.workflow_path = Path(workflow_path)
        if not self.workflow_path.is_file():
            raise FileNotFoundError(f"Workflow not found: {self.workflow_path}")

        self.config = config or ComfyServerConfig()
        self.node_map = node_map or NodeMap()
        self.client_id = str(uuid.uuid4())
        self._workflow_template: dict = json.loads(
            self.workflow_path.read_text(encoding="utf-8")
        )

    def generate(self, request: GenerationRequest) -> Image.Image:
        """Run one edit job end to end and return the generated image.

        Args:
            request (GenerationRequest): The images, prompt and seed to use.

        Raises:
            requests.HTTPError: If the server rejects an upload or the prompt.
            RuntimeError: If generation finishes with no output image.
            TimeoutError: If the job exceeds ``config.timeout`` seconds.

        Returns:
            Image.Image: The decoded output image in RGB.
        """
        base_name = self._upload_image(request.base_image)
        person_name = self._upload_image(request.person_image)
        object_name = self._upload_image(request.object_image)

        prompt_graph = self._build_prompt(
            request, base_name, person_name, object_name
        )
        prompt_id = self._queue_prompt(prompt_graph)
        logger.info(f"Queued prompt {prompt_id}")

        self._await_completion(prompt_id)
        return self._fetch_output_image(prompt_id)

    def _upload_image(self, image_path: Path) -> str:
        """Upload an image to the server's input folder.

        Args:
            image_path (Path): Local image file to upload.

        Raises:
            FileNotFoundError: If the file does not exist.
            requests.HTTPError: If the server rejects the upload.

        Returns:
            str: The filename ComfyUI stored it under, for use in LoadImage.
        """
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        with image_path.open("rb") as handle:
            files = {"image": (image_path.name, handle, "application/octet-stream")}
            response = requests.post(
                f"{self.config.http_url}/upload/image",
                files=files,
                data={"overwrite": "true"},
                timeout=60,
            )
        response.raise_for_status()
        stored_name = response.json()["name"]
        logger.debug(f"Uploaded {image_path.name} -> {stored_name}")
        return stored_name

    def _build_prompt(
        self,
        request: GenerationRequest,
        base_name: str,
        person_name: str,
        object_name: str,
    ) -> dict:
        """Copy the workflow template and inject the request values.

        Args:
            request (GenerationRequest): Source of prompt and seed.
            base_name (str): Server-side filename for the base image.
            person_name (str): Server-side filename for the person image.
            object_name (str): Server-side filename for the object image.

        Raises:
            KeyError: If a mapped node id is missing from the workflow.

        Returns:
            dict: A ready-to-queue API-format prompt graph.
        """
        graph = copy.deepcopy(self._workflow_template)
        node_map = self.node_map

        graph[node_map.base_image]["inputs"]["image"] = base_name
        graph[node_map.person_image]["inputs"]["image"] = person_name
        graph[node_map.object_image]["inputs"]["image"] = object_name

        if request.prompt is not None:
            graph[node_map.prompt]["inputs"][node_map.prompt_field] = request.prompt

        seed = request.seed if request.seed is not None else random.randint(0, 2**63 - 1)
        graph[node_map.seed]["inputs"]["seed"] = seed

        return graph

    def _queue_prompt(self, prompt_graph: dict) -> str:
        """Submit a prompt graph to the ComfyUI queue.

        Args:
            prompt_graph (dict): The API-format prompt to run.

        Raises:
            requests.HTTPError: If the server rejects the prompt (e.g. a
                missing model or node). The response body lists the reason.

        Returns:
            str: The prompt id used to track and fetch results.
        """
        payload = {"prompt": prompt_graph, "client_id": self.client_id}
        response = requests.post(
            f"{self.config.http_url}/prompt", json=payload, timeout=60
        )
        if response.status_code >= 400:
            logger.error(f"Prompt rejected: {response.text}")
        response.raise_for_status()
        return response.json()["prompt_id"]

    def _await_completion(self, prompt_id: str) -> None:
        """Block until the given prompt finishes executing.

        Listens to the server's websocket for the completion signal, which is
        an ``executing`` message with a null node for this prompt id.

        Args:
            prompt_id (str): The prompt to wait for.

        Raises:
            TimeoutError: If no completion arrives within ``config.timeout``.
        """
        query = urlencode({"clientId": self.client_id})
        connection = websocket.create_connection(
            f"{self.config.ws_url}?{query}", timeout=self.config.timeout
        )
        try:
            connection.settimeout(self.config.timeout)
            while True:
                message = connection.recv()
                if not isinstance(message, str):
                    # Binary frames are latent previews; ignore them.
                    continue

                event = json.loads(message)
                if event.get("type") == "executing":
                    data = event["data"]
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        return
                elif event.get("type") == "execution_error":
                    data = event["data"]
                    if data.get("prompt_id") == prompt_id:
                        raise RuntimeError(f"Execution error: {data}")
        except websocket.WebSocketTimeoutException as exc:
            raise TimeoutError(
                f"Prompt {prompt_id} exceeded {self.config.timeout}s"
            ) from exc
        finally:
            connection.close()

    def _fetch_output_image(self, prompt_id: str) -> Image.Image:
        """Download the first output image produced by a finished prompt.

        Args:
            prompt_id (str): The completed prompt id.

        Raises:
            RuntimeError: If the history holds no image output.
            requests.HTTPError: If the image download fails.

        Returns:
            Image.Image: The output image in RGB.
        """
        history = requests.get(
            f"{self.config.http_url}/history/{prompt_id}", timeout=60
        ).json()
        outputs = history.get(prompt_id, {}).get("outputs", {})

        for node_output in outputs.values():
            for image_info in node_output.get("images", []):
                if image_info.get("type") == "temp":
                    continue
                return self._download_image(image_info)

        raise RuntimeError(f"No output image for prompt {prompt_id}")

    def _download_image(self, image_info: dict) -> Image.Image:
        """Fetch a single image described by a history output entry.

        Args:
            image_info (dict): An entry with ``filename``, ``subfolder`` and
                ``type`` keys, as returned by ``/history``.

        Raises:
            requests.HTTPError: If the download fails.

        Returns:
            Image.Image: The image in RGB.
        """
        params = {
            "filename": image_info["filename"],
            "subfolder": image_info.get("subfolder", ""),
            "type": image_info.get("type", "output"),
        }
        response = requests.get(
            f"{self.config.http_url}/view", params=params, timeout=60
        )
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
