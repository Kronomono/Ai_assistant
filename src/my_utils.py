import ffmpeg
import numpy as np
import os
import torch
import librosa
import moviepy.editor as mp
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoTokenizer, AutoModelForSequenceClassification, pipeline
from safetensors import safe_open


# making audio for assistant voice
def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()
# all for understanding your voice
def load_whisper_model(model_name, model_path):
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    with safe_open(model_path, framework="pt", device="cuda:0") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
    model.load_state_dict(state_dict, strict=False)
    return model

def extract_audio(file_path):
    if file_path.endswith(('.mp4', '.mkv', '.avi')):
        video = mp.VideoFileClip(file_path)
        audio_path = file_path.rsplit('.', 1)[0] + '.wav'
        video.audio.write_audiofile(audio_path)
    else:
        audio_path = file_path
    return audio_path

def transcribe_audio(model, processor, audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# load model for emotion detection
def load_emotion_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # Create a pipeline for emotion detection
    emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return emotion_classifier
# Get the Mood from the text
def get_overall_mood(text):
    emotion_classifier = load_emotion_model("j-hartmann/emotion-english-distilroberta-base")
    # Split the text into manageable segments
    segments = text.split("\n")  # Assuming each paragraph is separated by a newline

    # Get predictions for each segment
    predictions = emotion_classifier(segments)

    # Aggregate results to find overall mood
    moods = [pred['label'] for pred in predictions]
    overall_mood = max(set(moods), key=moods.count)  # Get the most frequent mood

    return overall_mood