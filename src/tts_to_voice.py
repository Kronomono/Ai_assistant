import os
import sys
import asyncio
import edge_tts
from pydub import AudioSegment
import librosa
import torch
from scipy.io import wavfile
from my_utils import load_audio, load_emotion_model
from vc_infer_pipeline import VC
from rvc import load_hubert, get_vc, Config

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

# get Model for emotion detection
def get_model_and_pitch(mood):
    if mood == "joy":
        return "voice_models/Starfire.pth", 8.0
    elif mood == "sadness":
        return "voice_models/RaidenShogunEN.pth", 2.0
    elif mood == "anger":
        return "voice_models/KafkaJP.pth", 3.0
    elif mood == "fear":
        return "voice_models/tokisaki-300.pth", 6.0
    elif mood == "love":
        return "voice_models/RieTakahashi.pth", 12.0
    elif mood == "surprise":
        return "voice_models/akanev2.pth", 10.0
    elif mood == "neutral":
        return "voice_models/akanev2.pth", 6.0
    else:
        return "voice_models/akanev2.pth", 6.0

# TTS generation
async def text_to_temp_wav(text, voice, speed, temp_wav_path):
    mood = get_overall_mood(text)
    print(f"The overall mood of the text is: {mood}")
    temp_mp3_path = temp_wav_path.replace('.wav', '.mp3')
    communicate = edge_tts.Communicate(text, voice, rate=speed)
    await communicate.save(temp_mp3_path)
    # Convert MP3 to WAV
    audio = AudioSegment.from_mp3(temp_mp3_path)
    audio.export(temp_wav_path, format="wav")
    os.remove(temp_mp3_path)
    return mood

def verify_audio_file(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        return True
    except Exception as e:
        print(f"Failed to load {audio_path} with librosa: {e}")
        return False

def generate_tts_audio(txt_file, temp_wav_path, voice, speed):
    with open(txt_file, 'r') as file:
        text = file.read()
    mood = asyncio.run(text_to_temp_wav(text, voice, speed, temp_wav_path))
    if not os.path.exists(temp_wav_path):
        raise FileNotFoundError(f"Failed to generate temporary TTS audio file at {temp_wav_path}")
    if not verify_audio_file(temp_wav_path):
        raise RuntimeError(f"Temporary output WAV file {temp_wav_path} is not readable.")
    return mood

# Voice conversion
def voice_conversion(temp_wav_path, output_wav_path, model_path, hubert_model_path, pitch_change):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    is_half = False

    # Load Hubert model
    hubert_model = load_hubert(device, is_half, hubert_model_path)

    # Load voice conversion model
    config = Config(device=device, is_half=is_half)
    cpt, version, net_g, tgt_sr, vc = get_vc(device, is_half, config, model_path)

    # Run the voice conversion
    audio = load_audio(temp_wav_path, 16000)
    times = [0, 0, 0]
    audio_opt = vc.pipeline(hubert_model, net_g, 0, audio, temp_wav_path, times, pitch_change, 'rmvpe', '', 0.5, 1, 3, tgt_sr, 0, 0.25, version, 0.33, 128)
    wavfile.write(output_wav_path, tgt_sr, audio_opt)
    os.remove(temp_wav_path)

if __name__ == "__main__":
    # Parameters for TTS
    txt_file = 'Testing.txt'
    temp_wav_path = 'temp.wav'
    voice = "en-US-AriaNeural"
    speed = "+0%"

    # Parameters for voice conversion
    output_wav_path = 'output.wav'
    hubert_model_path = 'rvc_models/hubert_base.pt'

    # Generate TTS audio and get mood
    mood = generate_tts_audio(txt_file, temp_wav_path, voice, speed)
    
    # Get model path and pitch change based on mood
    model_path, pitch_change = get_model_and_pitch(mood)
    print(f"Using model path: {model_path} and pitch change: {pitch_change} based on the mood: {mood}")
    
    # Convert voice
    voice_conversion(temp_wav_path, output_wav_path, model_path, hubert_model_path, pitch_change)

    print(f"Voice conversion completed. Output file: {output_wav_path}")
