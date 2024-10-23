import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uuid
import shutil
import logging
import tempfile
import infer  # Assuming you have an infer module for processing
import gradio as gr
import json

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

# FastAPI app initialization
app = FastAPI()

# Data model for synthesis request
class SynthesisRequest(BaseModel):
    text: str
    model_path: str
    config_path: str
    speakers_path: str
    speaker_name: str = None  # Optional

def remove_contents(path):
    """Utility to clear contents of a directory."""
    for file_name in os.listdir(path):
        pth = os.path.join(path, file_name)
        if os.path.isfile(pth):
            os.remove(pth)

@app.post("/synthesize/")
async def synthesize(request: SynthesisRequest):
    """Synthesizes audio based on the provided TTS model and configuration."""
    model_path = request.model_path
    config_path = request.config_path
    speaker_name = request.speaker_name

    speakers_file_path = request.speakers_path
    target_speakers_file = os.path.join(CACHE_PATH, "speakers.pth")

    if not os.path.exists(speakers_file_path):
        logging.error(f"Speakers file not found: {speakers_file_path}")
        raise HTTPException(status_code=400, detail=f"Speakers file not found: {speakers_file_path}")

    try:
        shutil.copy(speakers_file_path, target_speakers_file)
        logging.info(f"Overwrote speakers.pth with {speakers_file_path}")
    except Exception as e:
        logging.error(f"Failed to copy speakers file: {e}")
        raise HTTPException(status_code=400, detail=f"Error copying speakers file: {e}")

    output_filename = f"{uuid.uuid4()}.wav"
    paths = {
        "model_path": model_path,
        "config_path": config_path
    }

    try:
        if asyncio.iscoroutinefunction(infer.infer_multi):
            await infer.infer_multi(request.text, speaker_name, None, output_filename, "eng", paths)
        else:
            await asyncio.to_thread(infer.infer_multi, request.text, speaker_name, None, output_filename, "eng", paths)
    except Exception as e:
        logging.error(f"An error occurred during inference: {e}")
        raise HTTPException(status_code=400, detail=f"Error: {e}")

    output_dir = os.path.join(ASP, output_filename)
    fod = os.path.join(AFSP, output_filename)
    infer.clean(output_dir, fod)

    return {"file_path": fod}  # Return the final synthesized audio file path

# Gradio Blocks interface
async def synthesize_audio(text, language, speaker_name):
    """Gradio interface function to synthesize audio via FastAPI."""
    paths = predefined_paths[language]
    
    speaker_id = paths["speakers"][speaker_name]  # Map speaker to ID
    data = SynthesisRequest(
        text=text,
        model_path=paths["model_path"],
        config_path=paths["config_path"],
        speakers_path=paths["speakers_path"],
        speaker_name=speaker_id
    )
    
    response = await synthesize(data)  # Send request to FastAPI endpoint
    
    return response['file_path']  # Return the file path for Gradio

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
    
    synthesize_button.click(synthesize_audio, inputs=[text_input, language_dropdown, speaker_dropdown], outputs=output_audio)

demo.launch(share=True)

