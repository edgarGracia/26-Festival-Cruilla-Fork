"""Client for running the "3ingredients" ComfyUI workflow on a remote server.

Usage:
    from comfy_client import run_3ingredients_workflow

    image_bytes = run_3ingredients_workflow(
        base_image="bg_los_salvajes.png",       # node 30 - base/background photo
        person_image="fotocampus.jpg",          # node 31 - person to insert
        object_image="blue_sunglasses.png",      # node 32 - object to insert
        prompt="Optional custom instructions",   # node 1 - omit to keep the default
    )
    with open("result.png", "wb") as f:
        f.write(image_bytes)
"""

import json
import uuid
from pathlib import Path
from urllib import request as urllib_request
from urllib.parse import urlencode

import websocket  # pip install websocket-client

SERVER_ADDRESS = "158.109.8.152:8188"
HTTP_TIMEOUT   = 10    # seconds for upload / queue / history / download calls
WS_TIMEOUT     = 300   # seconds to wait for the workflow to finish on the server
WORKFLOW_PATH = "/home/cvcadmin/cruilla/Festival-Cruilla/3ingredient_api.json"

# node ids in the workflow that take the three input images
BASE_IMAGE_NODE = "30"
PERSON_IMAGE_NODE = "31"
OBJECT_IMAGE_NODE = "32"
PROMPT_NODE = "1"
SAVE_IMAGE_NODE = "22"


def _upload_image(image_path, server_address, client_id):
    image_path = Path(image_path)
    with open(image_path, "rb") as f:
        data = f.read()

    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    body += data
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib_request.Request(
        f"http://{server_address}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib_request.urlopen(req, timeout=HTTP_TIMEOUT) as response:
        result = json.loads(response.read())
    return result["name"], result.get("subfolder", "")


def _queue_prompt(workflow, client_id, server_address):
    payload = {"prompt": workflow, "client_id": client_id}
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(f"http://{server_address}/prompt", data=data)
    with urllib_request.urlopen(req, timeout=HTTP_TIMEOUT) as response:
        return json.loads(response.read())


def _get_history(prompt_id, server_address):
    with urllib_request.urlopen(f"http://{server_address}/history/{prompt_id}", timeout=HTTP_TIMEOUT) as response:
        return json.loads(response.read())


def _get_image(filename, subfolder, folder_type, server_address):
    params = urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
    with urllib_request.urlopen(f"http://{server_address}/view?{params}", timeout=HTTP_TIMEOUT) as response:
        return response.read()


def run_3ingredients_workflow(
    base_image,
    person_image,
    object_image,
    prompt=None,
    server_address=SERVER_ADDRESS,
    workflow_path=WORKFLOW_PATH,
):
    """Run the 3ingredients ComfyUI workflow and return the output image bytes.

    base_image / person_image / object_image are local file paths for the
    three LoadImage inputs (nodes 30/31/32). prompt overrides the workflow's
    text prompt (node 1); if None, the prompt baked into the workflow file
    is used unchanged.
    """
    with open(workflow_path, "r") as f:
        workflow = json.load(f)

    client_id = str(uuid.uuid4())

    base_name, base_sub = _upload_image(base_image, server_address, client_id)
    person_name, person_sub = _upload_image(person_image, server_address, client_id)
    object_name, object_sub = _upload_image(object_image, server_address, client_id)

    workflow[BASE_IMAGE_NODE]["inputs"]["image"] = (
        f"{base_sub}/{base_name}" if base_sub else base_name
    )
    workflow[PERSON_IMAGE_NODE]["inputs"]["image"] = (
        f"{person_sub}/{person_name}" if person_sub else person_name
    )
    workflow[OBJECT_IMAGE_NODE]["inputs"]["image"] = (
        f"{object_sub}/{object_name}" if object_sub else object_name
    )

    if prompt is not None:
        workflow[PROMPT_NODE]["inputs"]["value"] = prompt

    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}", timeout=WS_TIMEOUT)

    try:
        queued = _queue_prompt(workflow, client_id, server_address)
        prompt_id = queued["prompt_id"]

        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message.get("type") == "executing":
                    data = message["data"]
                    if data.get("prompt_id") == prompt_id and data.get("node") is None:
                        break  # execution finished
    finally:
        ws.close()

    history = _get_history(prompt_id, server_address)[prompt_id]
    node_output = history["outputs"][SAVE_IMAGE_NODE]
    image_info = node_output["images"][0]

    return _get_image(
        image_info["filename"],
        image_info["subfolder"],
        image_info["type"],
        server_address,
    )


if __name__ == "__main__":
    result = run_3ingredients_workflow(
        base_image="bg_los_salvajes.png",
        person_image="fotocampus.jpg",
        object_image="blue_sunglasses.png",
    )
    with open("result.png", "wb") as f:
        f.write(result)
    print("Saved result.png")
