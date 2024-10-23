import os
import uuid
import shutil
import logging
import tempfile
import infer  # Assuming you have an infer module for processing
import gradio as gr
import json
import asyncio  # Make sure to import asyncio

# Function to read paths dynamically from config_files.json
def get_paths_from_config():
    config_file = "./config_files.json"
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found at {config_file}")

# Get predefined paths dynamically from config_files.json
predefined_paths = get_paths_from_config()

# Environment and cache paths
CACHE_PATH = os.path.join(tempfile.gettempdir(), "cache/general")
ASP = os.path.join(tempfile.gettempdir(), "audios/")
AFSP = os.path.join(tempfile.gettempdir(), "final/")
os.makedirs(CACHE_PATH, exist_ok=True)
os.makedirs(ASP, exist_ok=True)
os.makedirs(AFSP, exist_ok=True)

logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Data model for synthesis request
class SynthesisRequest:
    def __init__(self, text, model_path, config_path, speakers_path, speaker_name=None):
        self.text = text
        self.model_path = model_path
        self.config_path = config_path
        self.speakers_path = speakers_path
        self.speaker_name = speaker_name  # Optional

async def synthesize_audio(text, language, speaker_name):
    """Synthesize audio based on the provided TTS model and configuration."""
    paths = predefined_paths[language]

    speaker_id = paths["speakers"][speaker_name]  # Map speaker to ID
    data = SynthesisRequest(
        text=text,
        model_path=paths["model_path"],
        config_path=paths["config_path"],
        speakers_path=paths["speakers_path"],
        speaker_name=speaker_id
    )

    # Handle speakers file
    speakers_file_path = data.speakers_path
    target_speakers_file = os.path.join(CACHE_PATH, "speakers.pth")

    if not os.path.exists(speakers_file_path):
        logging.error(f"Speakers file not found: {speakers_file_path}")
        return "Speakers file not found."

    try:
        shutil.copy(speakers_file_path, target_speakers_file)
        logging.info(f"Overwrote speakers.pth with {speakers_file_path}")
    except Exception as e:
        logging.error(f"Failed to copy speakers file: {e}")
        return f"Error copying speakers file: {e}"

    output_filename = f"{uuid.uuid4()}.wav"
    paths_dict = {
        "model_path": data.model_path,
        "config_path": data.config_path
    }

    try:
        # Using asyncio.to_thread to run the synchronous function
        if asyncio.iscoroutinefunction(infer.infer_multi):
            await infer.infer_multi(data.text, speaker_id, None, output_filename, "eng", paths_dict)
        else:
            await asyncio.to_thread(infer.infer_multi, data.text, speaker_id, None, output_filename, "eng", paths_dict)
    except Exception as e:
        logging.error(f"An error occurred during inference: {e}")
        return f"Error: {e}"

    output_dir = os.path.join(ASP, output_filename)
    fod = os.path.join(AFSP, output_filename)
    infer.clean(output_dir, fod)

    return fod  # Return the final synthesized audio file path

# Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        text_input = gr.Textbox(label="Text to Synthesize")
        language_dropdown = gr.Dropdown(
            label="Language",
            choices=["--Select Language--", "Hindi", "English"],
            value="--Select Language--",
            interactive=True
        )
        speaker_dropdown = gr.Dropdown(label="Speaker Name", choices=[])

    # Update speakers when language changes
    language_dropdown.change(
        lambda lang: gr.update(choices=list(predefined_paths[lang]["speakers"].keys())),
        inputs=language_dropdown,
        outputs=speaker_dropdown
    )

    synthesize_button = gr.Button("Synthesize")
    output_audio = gr.Audio(label="Synthesized Audio", interactive=False)

    # Button click event to synthesize audio
    synthesize_button.click(synthesize_audio, inputs=[text_input, language_dropdown, speaker_dropdown], outputs=output_audio)

demo.launch(share=True)
