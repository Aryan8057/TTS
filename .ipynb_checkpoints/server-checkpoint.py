from fastapi import FastAPI, HTTPException, UploadFile, Form, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
from typing import Optional
from pydub import AudioSegment
import os
import uuid
import sys
import boto3
import shutil
import logging
import tempfile

# Use the system's temp directory for caching
CACHE_PATH = os.path.join(tempfile.gettempdir(), "cache/general")
ASP = os.path.join(tempfile.gettempdir(), "audios/")
AFSP = os.path.join(tempfile.gettempdir(), "final/")
# Ensure cache directory exists
os.makedirs(CACHE_PATH, exist_ok=True)
os.makedirs(ASP, exist_ok=True)
os.makedirs(AFSP, exist_ok=True)

import infer
import configs

logging.basicConfig(
    filename='app.log',  # Log file name
    level=logging.DEBUG,  # Set log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

# Optionally, add a handler to output to the console as well
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

app = FastAPI()

def download_file_from_s3(bucket_name, s3_file_path, local_file_path):
    s3 = boto3.client('s3')

    try:
        # Download the file
        s3.download_file(bucket_name, s3_file_path, local_file_path)
        print(f"File downloaded successfully to {local_file_path}")
    except Exception as e:
        logging.error(f"Error downloading file from S3: {e}")
        raise HTTPException(status_code=500, detail="Error downloading file from S3")

def remove_contents(path):
    for file_name in os.listdir(path):
        pth = os.path.join(path,file_name)
        if os.path.isfile(pth):
            os.remove(pth)

class SynthesisRequest(BaseModel):
    text: str
    model_path: str
    config_path : str
    speakers_path : str
    speaker_name : str
    speaker_wav: str

@app.post("/synthesize/")
async def synthesize(request: SynthesisRequest):
    model_path = ""
    config_path = ""
    speaker_name = None
    speaker_wav = None

    if request.speaker_name != "None":
        speaker_name = request.speaker_name
    if request.speaker_wav != "None":
        speaker_wav = request.speaker_wav

    text = request.text
    model = request.model_path
    config = request.config_path
    speakers = request.speakers_path
    bucket = model.split("/")[2]
    model_key = "/".join((model.split("/")[3:]))
    config_key = "/".join((config.split("/")[3:]))
    speakers_key = "/".join((speakers.split("/")[3:]))
    model_name = model.split("/")[-1]
    config_name =  config.split("/")[-1]
    speaker_save_name = speakers.split("/")[-1]

    if speaker_wav and speaker_name:
        raise HTTPException(status_code=400, detail="Please enter either wav path or speaker")
    if speaker_wav:
        wav_key = "/".join((speaker_wav.split("/")[3:]))
        download_file_from_s3(bucket, wav_key, os.path.join(CACHE_PATH, wav_key.split("/")[-1]))
    contents = os.listdir(CACHE_PATH)
    print(len(contents))

    if len(contents) < 3:
        print("downloading")
        download_file_from_s3(bucket, model_key, os.path.join(CACHE_PATH, model_name))
        download_file_from_s3(bucket, config_key, os.path.join(CACHE_PATH, config_name))
        download_file_from_s3(bucket, speakers_key, os.path.join(CACHE_PATH, speaker_save_name))
    else:
        key = model_name.split("-")[0]
        curr_key = os.listdir(CACHE_PATH)[0].split("-")[0]
        if key != curr_key:
            remove_contents(CACHE_PATH)
            print("downloading...")
            download_file_from_s3(bucket, model_key, os.path.join(CACHE_PATH, model_name))
            download_file_from_s3(bucket, config_key, os.path.join(CACHE_PATH, config_name))
            download_file_from_s3(bucket, speakers_key, os.path.join(CACHE_PATH, speaker_save_name))
            
    model_path = os.path.join(CACHE_PATH, model_name)
    config_path = os.path.join(CACHE_PATH, config_name)
    print(f"model path is {model_path}")

    output_filename = f"{uuid.uuid4()}.wav"
    paths = {
        "model_path": model_path,
        "config_path": config_path
    }

    print(model_path)
    print(config_path)
    print(f"speaker name is {speaker_name}")
    
    try:
        infer.infer_multi(text, speaker_name, speaker_wav, output_filename, "eng", paths)
    except Exception as e:
        logging.error(f"An error occurred while inferencing: {e}")
        raise HTTPException(status_code=400, detail=f"Error: {e}")

    output_dir = os.path.join(tempfile.gettempdir(),f'audios/{output_filename}')
    fod = os.path.join(tempfile.gettempdir(),f'final/{output_filename}')
    infer.clean(output_dir, fod)

    return FileResponse(fod, media_type='audio/wav', filename=output_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8182)
