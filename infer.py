import os
from TTS.api import TTS
import subprocess
import logging
import re
import uuid
import boto3
import tempfile

def infer_multi(text, speaker, speaker_wav, output_filename, lang, paths):
    # Initialize TTS model
    print("called")
    p = paths["config_path"]
    print(p)
    print(f"called infer_multi {p}")
    tts = TTS(
        model_path=paths["model_path"],
        config_path=paths["config_path"],
        progress_bar=True,
        gpu=False
    )
    print("donw")
    
    # Set output directory to /tmp
    output_dir = os.path.join(tempfile.gettempdir(),f"audios/{output_filename}")
    
    if speaker is not None:
        print("text one is called")
        tts.tts_to_file(text=text, speaker=speaker, speaker_wav=None, file_path=output_dir)
    else:
        print("wav is called")
        tts.tts_to_file(text=text, speaker=None, speaker_wav=speaker_wav, file_path=output_dir)

def apply_noise_gate(input_audio_path, output_audio_path, threshold='-30dB'):
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_audio_path,
        '-af', f"agate=threshold={threshold}",
        output_audio_path
    ]
    try:
        subprocess.run(ffmpeg_command, check=True)
        logging.info(f"Noise gate applied successfully. Output saved to {output_audio_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error applying noise gate: {e}")

def get_audio_levels(audio_path):
    ffmpeg_command = [
        'ffmpeg',
        '-i', audio_path,
        '-filter:a', 'astats=metadata=1:measure_overall=0',
        '-f', 'null', '-'
    ]
    try:
        result = subprocess.run(ffmpeg_command, stderr=subprocess.PIPE, text=True, check=True)
        rms_peak_dB = None
        peak_level_dB = None

        for line in result.stderr.splitlines():
            if 'RMS peak dB' in line:
                rms_peak_dB = float(re.search(r'RMS peak dB\s*:\s*([-.\d]+)', line).group(1))
                logging.info(f"rms_peak_dB {rms_peak_dB}")
            if 'Peak level dB' in line:
                peak_level_dB = float(re.search(r'Peak level dB\s*:\s*([-.\d]+)', line).group(1))
                logging.info(f"peak_level_dB {peak_level_dB}")

        return rms_peak_dB, peak_level_dB
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting audio levels: {e}")
        return None, None

def apply_compressor(input_audio_path, compressed_audio_path, threshold, ratio):
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_audio_path,
        '-af', f"acompressor=threshold={threshold}dB:ratio={ratio}:attack=50:release=200",
        compressed_audio_path
    ]
    try:
        subprocess.run(ffmpeg_command, check=True)
        logging.info(f"Compressor applied successfully. Output saved to {compressed_audio_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error applying compressor: {e}")

def process_audio(input_audio_path, output_audio_path):
    current_audio_path = input_audio_path

    while True:
        print("compressing")
        rms_peak_dB, peak_level_dB = get_audio_levels(current_audio_path)

        if rms_peak_dB is None or peak_level_dB is None:
            logging.error("Error obtaining audio levels. Aborting compression.")
            return

        if abs(peak_level_dB) >= 7.0:
            logging.info(f"rms_peak_dB {rms_peak_dB}")
            logging.info(f"peak_level_dB {peak_level_dB}")
            logging.info(f"Peak level is within acceptable range at {peak_level_dB} dB, no further compression needed.")
            break

        threshold = rms_peak_dB
        ratio = min(20, (peak_level_dB - rms_peak_dB))

        compressed_output = os.path.join(
            tempfile.gettempdir(),
            f"compressed_{os.path.basename(current_audio_path)}"
        )
        apply_compressor(current_audio_path, compressed_output, threshold, ratio)

        current_audio_path = compressed_output

    noise_gate_output = os.path.join(
        tempfile.gettempdir(),
        f"noise_gated_{os.path.basename(current_audio_path)}"
    )

    # Apply the noise gate
    apply_noise_gate(current_audio_path, noise_gate_output)

    try:
        # Move the final output file to the intended output path
        os.rename(noise_gate_output, output_audio_path)
        logging.info(f"Final processed audio saved to {output_audio_path}")
    except OSError as e:
        logging.error(f"Error moving final output file: {e}")

def clean(input_audio, output_audio):
    process_audio(input_audio, output_audio)
