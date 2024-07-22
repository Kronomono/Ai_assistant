import os
from transformers import WhisperProcessor
from my_utils import load_whisper_model, extract_audio, transcribe_audio

class WhisperTranscriber:
    def __init__(self, input_file, model, processor):
        self.input_file = input_file
        self.model = model
        self.processor = processor

    def transcribe(self):
        # Extract and transcribe the audio
        audio_path = extract_audio(self.input_file)
        transcription = transcribe_audio(self.model, self.processor, audio_path)

        # Clean up if necessary
        if self.input_file.endswith(('.mp4', '.mkv', '.avi')):
            os.remove(audio_path)  # Delete the extracted .wav file if the input was a video

        return transcription
    @staticmethod
    def write_to_txt(transcription):
        with open("test_output.txt", 'w', encoding='utf-8') as f:
                f.write(transcription)
