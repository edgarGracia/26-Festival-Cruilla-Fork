from enum import Enum, auto
from pathlib import Path

import gradio as gr
import qrcode
from loguru import logger
from PIL import Image

import config
from main_parallel import run_pipeline
from upload_utils import upload_results

T_USER_VECT = tuple[str, str, str, str, str]  # mood, instrument, era, casa, locale


class State(Enum):
    home = auto()
    take_photo = auto()
    generating = auto()
    results = auto()


# Global app state
state: State = State.home
user_image: Image.Image | None = None
result_image: Image.Image | None = None
user_vector: T_USER_VECT | None = None
qr_out_image: Image.Image | None = None
root_path = Path(__file__).parent

# Assets paths
camera_button_path = str(root_path / "res/camera.png").replace("\\", "/")
home_background_path = str(root_path / "res/home_background.png").replace("\\", "/")
generation_background_path = str(root_path / "res/generation_background.png").replace("\\", "/")
result_background_path = str(root_path / "res/result_background.png").replace("\\", "/")

# Page CSS
main_css = f"""
body, gradio-app, .gradio-container {{
    background: #ffec04 !important;
}}

#home_background, #generation_background, #result_background,
#generating_background {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-size: cover;
    background-position: top left;
    background-repeat: no-repeat;
    z-index: 5;
    pointer-events: none;
}}
#home_background {{
    background-image: url("/gradio_api/file={home_background_path}");
}}
#generation_background {{
    background-image: url("/gradio_api/file={generation_background_path}");
}}
#result_background, #generating_background {{
    background-image: url("/gradio_api/file={result_background_path}");
}}

#qr_code_text {{
    position: fixed;
    left: -9999px;
}}

#webcam_photo {{
    border-width: 0px !important;
    pointer-events: none;
    background: transparent;
    visibility: hidden;
}}
#webcam_photo [title="grant webcam access"] {{
    display: none !important;
}}
#webcam_photo .button-wrap {{
    display: none !important;
}}
#webcam_photo video, #webcam_photo img {{
    visibility: visible;
    position: fixed;
    left: 50vw;
    top: 50vh;
    transform: translate(-50%, -50%);
    width: 64vw;
    max-width: none;
    z-index: 1;
}}
#countdown_wrap {{
    pointer-events: none;
}}
#countdown {{
    position: fixed;
    top: 50vh;
    left: 50vw;
    transform: translate(-50%, -50%);
    display: none;
    font-size: 25vh;
    color: white;
    text-shadow: 0 0 20px rgba(0,0,0,0.6);
    z-index: 150;
}}

#qr_image {{
    visibility: hidden;
}}
#qr_image img {{
    visibility: visible;
    position: fixed;
    left: 11.5vw;
    top: 11vh;
    width: 18.75vw;
    max-width: none;
    z-index: 10;
    pointer-events: none;
}}
#result_video {{
    visibility: hidden;
}}
#result_video video {{
    visibility: visible;
    position: fixed;
    left: 67.2vw;
    top: 49.8vh;
    transform: translate(-50%, -50%);
    width: 57vw;
    max-width: none;
    z-index: 1;
}}
#button_restart {{
    position: fixed;
    bottom: 4vh;
    left: 11.5vw;
    width: 18.75vw;
    z-index: 100;
    font-size: 3vh;
}}
#spinner_wrap {{
    position: fixed;
    left: 67.2vw;
    top: 49.8vh;
    transform: translate(-50%, -50%);
    z-index: 100;
}}
#spinner {{
    width: 80px;
    height: 80px;
    border: 8px solid rgba(0,0,0,0.15);
    border-top-color: #333;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}}
@keyframes spin {{
    to {{ transform: rotate(360deg); }}
}}
#camera_button {{
    background-image: url("/gradio_api/file={camera_button_path}");
    background-size: contain;
    background-position: center;
    background-repeat: no-repeat;
    background-color: transparent;
    position: fixed;
    bottom: 3vh;
    left: 50vw;
    transform: translateX(-50%);
    z-index: 100;
    height: 11vh;
    width: 10vw;
}}
"""  # noqa: F541

# Page JS
main_js = f"""
() => {{
    const checkWebcamInterval = 1000;
    const focusQRInterval = 1000;

    function checkAndClickWebcam() {{
        const selector = 'div[title="grant webcam access"]';
        const divs = document.querySelectorAll(selector);

        divs.forEach(div => {{
            const button = div.querySelector('button');
            if (button) {{
                button.click();
                console.log("Grant webcam access");
            }}
        }});
    }}
    setInterval(checkAndClickWebcam, checkWebcamInterval);
    
    document.addEventListener('keydown', (event) => {{
        if (event.key === 'Enter') {{
            console.log("Enter key pressed");
            const qr_input_button = document.getElementById("qr_input_button");
            if (qr_input_button) {{
                console.log("qr_input_button click");
                qr_input_button.click();
            }} else {{
                console.log("qr_input_button not found");
            }}
        }}
    }});

    function focusQRInput() {{
        const column_home = document.getElementById("column_home");
        if (!column_home || column_home.classList.contains('hide')) {{
            return;
        }}
        const qr_code_text = document.querySelector('#qr_code_text textarea');
        if (qr_code_text) {{
            if (document.activeElement !== qr_code_text) {{
                console.log("QR code text input focus");
                qr_code_text.focus();
            }}
        }} else {{
            console.log("QR code text input not found");
        }}
    }}
    setInterval(focusQRInput, focusQRInterval);
}}
"""  # noqa: F541


take_photo_js = f"""
() => {{
    let count = {config.cam_count_down};
    const countdownElement = document.getElementById('countdown');
    
    // Show the countdown
    countdownElement.style.display = 'block';
    countdownElement.textContent = count;

    // Update countdown every second
    const timer = setInterval(() => {{
        count--;
        if (count > 0) {{
            countdownElement.textContent = count;
        }} else {{
            // Hide when countdown reaches 0
            countdownElement.style.display = 'none';
            clearInterval(timer);
            
            // Take a photo
            const imageInputDiv = document.getElementById("webcam_photo");
            if (imageInputDiv) {{
                const captureButton = imageInputDiv.querySelector('[aria-label="capture photo"]');

                if (captureButton) {{
                    captureButton.click();
                }} else {{
                    console.warn('Capture button not found inside #webcam');
                }}
            }} else {{
                console.warn('Div with ID "webcam" not found');
            }}
        }}
    }}, 1000);
}}
"""


html_count_down = """<div id="countdown" class="countdown-container"></div>"""


html_generating = """<div id="generating_background"></div>
<div id="spinner_wrap"><div id="spinner"></div></div>"""


html_home = """<div id="home_background"></div>"""
html_generation = """<div id="generation_background"></div>"""
html_result = """<div id="result_background"></div>"""


def parse_qr(gr_text_qr_code: str) -> T_USER_VECT:
    mood_dict = {
        "1": "happy",
        "2": "sad",
        "3": "chill",
        "4": "hype",
    }
    instrument_dict = {
        "1": "guitar",
        "2": "piano",
        "3": "trumpet",
        "4": "drums",
    }
    era_dict = {
        "1": "medieval",
        "2": "90s",
        "3": "futuristic",
        "4": "actual",
    }
    casa_dict = {
        "1": "urban",
        "2": "pop",
        "3": "techno",
        "4": "indie",
        "5": "rock",
    }
    locale_dict = {
        "1": "ca",
        "2": "es",
        "3": "en",
    }

    mood, instrument, era, casa, locale = gr_text_qr_code.split(",")

    mood = mood_dict[mood]
    instrument = instrument_dict[instrument]
    era = era_dict[era]
    casa = casa_dict[casa]
    locale = locale_dict[locale]
    return mood, instrument, era, casa, locale


def set_state(new_state: State):
    global state
    logger.info(f"Setting state: {new_state}")
    state = new_state


def set_user_image(image: Image.Image | None):
    global user_image
    logger.debug(f"Set user image: {image}")
    user_image = image


def set_user_vector(vector: tuple[str, str, str, str] | None):
    global user_vector
    logger.debug(f"Set user vector: {vector}")
    user_vector = vector


def set_qr_out_image(image: Image.Image | None):
    global qr_out_image
    logger.debug(f"Set result QR image: {image}")
    qr_out_image = image


def get_qr_out_image():
    return qr_out_image


def on_button_qr_input(gr_text_qr_code):
    if gr_text_qr_code and state is State.home:
        logger.info(f"QR code text input: {gr_text_qr_code}")
        try:
            user_vector = parse_qr(gr_text_qr_code)
            set_user_vector(user_vector)
            set_state(State.take_photo)
        except Exception:
            msg = f"Error parsing the QR data ({gr_text_qr_code})"
            logger.exception(msg)
            gr.Warning(msg)
    return ""


def on_image_photo(gr_image_photo) -> str:
    video_path = None
    set_user_image(gr_image_photo)
    if gr_image_photo is not None:
        set_state(State.generating)
        video_path = generate()
    return video_path


def generate() -> str:
    try:
        # Run generation
        mood, instrument, era, casa, locale = user_vector
        img_path = root_path / "user_image.png"
        out_path = root_path / "result_image.png"
        user_image.save(img_path)
        logger.info("Running pipeline")
        result = run_pipeline(
            image_path=str(img_path),
            output_path=str(out_path),
            mood=mood,
            instrument=instrument,
            era=era,
            casa=casa,
            language=locale,
            skip_music=False,
        )
        logger.success(f"Pipeline ran successfully: {result}")
        video_path = result["final_video"]
        logger.debug(f"Video path: {video_path}")
        
        # Upload results and create QR
        if config.upload_results:
            try:
                public_link = upload_results(video_path, config.upload_gen_prefix)
                logger.info(f"Results uploaded: {public_link}")
                qr_image = qrcode.make(public_link).get_image()
                set_qr_out_image(qr_image)
                upload_results(img_path, config.upload_inp_prefix)
            except Exception as e:
                logger.exception("Exception raised while uploading results")
                gr.Warning(f"Exception raised while uploading results: {e}")
        
        # Update UI state
        set_state(State.results)
        return video_path
    except Exception as e:
        logger.exception("Exception raised in generation")
        gr.Warning(f"Error in generation: {e}")
        reset()
    return None


def reset():
    logger.info("Resetting state")
    set_user_image(None)
    set_user_vector(None)
    set_state(State.home)
    set_qr_out_image(None)
    return None, None


def on_timer_update_state():
    gr_col_home = gr.Column(visible=False)
    gr_col_take_photo = gr.Column(visible=False)
    gr_col_generating = gr.Column(visible=False)
    gr_col_result = gr.Column(visible=False)

    if state is State.home:
        gr_col_home = gr.Column(visible=True)
    elif state is State.take_photo:
        gr_col_take_photo = gr.Column(visible=True)
    elif state is State.generating:
        gr_col_generating = gr.Column(visible=True)
    elif state is State.results:
        gr_col_result = gr.Column(visible=True)

    return gr_col_home, gr_col_take_photo, gr_col_generating, gr_col_result


with gr.Blocks(js=main_js, css=main_css) as demo:
    gr_timer_update_state = gr.Timer(config.ui_update_state_interval)

    with gr.Column(visible=state is State.home, elem_id="column_home") as gr_col_home:
        gr.HTML(html_home)
        gr_text_qr_code = gr.Textbox(elem_id="qr_code_text")
        gr_button_qr_input = gr.Button(visible=False, elem_id="qr_input_button")

    with gr.Column(
        visible=state is State.take_photo, elem_id="take_photo_column"
    ) as gr_col_take_photo:
        gr.HTML(html_generation)
        gr.HTML(html_count_down, elem_id="countdown_wrap")
        gr_image_photo = gr.Image(
            show_download_button=False,
            show_share_button=False,
            show_fullscreen_button=False,
            show_label=False,
            sources=["webcam"],
            elem_id="webcam_photo",
            interactive=True,
            type="pil",
            webcam_options=gr.WebcamOptions(mirror=False),
        )
        gr_button_take_photo = gr.Button("", elem_id="camera_button")

    with gr.Column(visible=state is State.generating) as gr_col_generating:
        gr_html_generating = gr.HTML(html_generating)

    with gr.Column(visible=state is State.results) as gr_col_result:
        gr.HTML(html_result)
        with gr.Row():
            with gr.Column(scale=1):
                gr_image_result_qr = gr.Image(
                    get_qr_out_image,
                    show_download_button=False,
                    show_share_button=False,
                    show_fullscreen_button=False,
                    show_label=False,
                    elem_id="qr_image",
                    every=1,
                )
                gr_button_result_restart = gr.Button(
                    "Tornar a començar", elem_id="button_restart"
                )
            gr_video_results = gr.Video(
                None,
                show_label=False,
                interactive=False,
                autoplay=True,
                show_share_button=False,
                loop=True,
                scale=10,
                elem_id="result_video",
            )

    gr_button_qr_input.click(
        on_button_qr_input,
        gr_text_qr_code,
        gr_text_qr_code,
        show_progress=False,
    )
    gr_button_take_photo.click(
        lambda: None,
        js=take_photo_js,
    )
    gr_image_photo.input(
        on_image_photo,
        gr_image_photo,
        gr_video_results,
        show_progress=False,
    )
    gr_button_result_restart.click(
        reset,
        outputs=[gr_video_results, gr_image_photo],
        show_progress=False,
    )
    gr_timer_update_state.tick(
        on_timer_update_state,
        None,
        [gr_col_home, gr_col_take_photo, gr_col_generating, gr_col_result],
    )


if __name__ == "__main__":
    logger.info("Launching Gradio app")
    demo.launch(
        server_name="0.0.0.0",
        allowed_paths=[str(root_path)],
    )
