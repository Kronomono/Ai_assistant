import os
import asyncio
from dotenv import load_dotenv
import discord
from discord import Intents, Client
from pydub import AudioSegment
from transcribe import WhisperTranscriber
from my_utils import load_whisper_model
from transformers import WhisperProcessor
from LLM import llm_wrapper

# Load environment variables from .env file
load_dotenv()

# Get the Discord token from the environment variable
TOKEN = os.getenv("DISCORD_TOKEN")

# Define the master's username
MASTER_USERNAME = "kronomono"

# Directory to save audio messages
AUDIO_DIR = "discord_audio_message"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Intents to allow the bot to read messages and be aware of mentions
intents = Intents.default()
intents.messages = True
intents.message_content = True
intents.members = True  # Enable member intents

# Load the Whisper model and processor
model_name = "openai/whisper-large-v2"
model_path = "whisper_models/large_model_v2.safetensors"
whisper_model = load_whisper_model(model_name, model_path)
processor = WhisperProcessor.from_pretrained(model_name)

class MyClient(Client):
    def __init__(self, whisper_model, processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.whisper_model = whisper_model
        self.processor = processor
        self.transcriber = WhisperTranscriber("", self.whisper_model, self.processor)
        self.is_ready = asyncio.Event()
    
    async def on_ready(self):
        print(f'Logged on as {self.user}')
        master_user = discord.utils.get(self.users, name=MASTER_USERNAME)
        if master_user:
            try:
                await master_user.send("Akane is online :white_check_mark:")
                print(f"Sent 'Akane is online' message to {MASTER_USERNAME}")
            except discord.errors.Forbidden:
                print(f"Unable to send DM to {MASTER_USERNAME}. Make sure the user allows DMs from the bot.")
        else:
            print(f"Could not find user {MASTER_USERNAME}.")
        self.is_ready.set()
        
    async def on_message(self, message):
        await self.is_ready.wait()
        
        if message.author == self.user:
            return

        if isinstance(message.channel, discord.DMChannel):
            if message.author.name != MASTER_USERNAME:
                await message.channel.send("You can't use me")
                return

            await message.channel.send("Akane has received your message. Please wait for a moment.")

            if message.attachments:
                await self.handle_attachments(message)
            else:
                await self.process_text_command(message.content, message.channel)

    async def handle_attachments(self, message):
        for attachment in message.attachments:
            if attachment.filename.endswith(('.mp3', '.wav', '.ogg')):
                file_path = os.path.join(AUDIO_DIR, attachment.filename)
                await attachment.save(file_path)
                print(f"Saved audio file to {file_path}")

                wav_path = os.path.join(AUDIO_DIR, "input.wav")

                if file_path.endswith('.ogg'):
                    audio = AudioSegment.from_ogg(file_path)
                    audio.export(wav_path, format='wav')
                    print(f"Converted {file_path} to {wav_path}")
                    os.remove(file_path)
                else:
                    os.rename(file_path, wav_path)
                    print(f"Renamed {file_path} to {wav_path}")

                await self.transcribe_and_process_audio(wav_path, message.channel)

    async def transcribe_and_process_audio(self, audio_path, channel):
        self.transcriber.input_file = audio_path
        loop = asyncio.get_running_loop()
        try:
            await channel.send("Akane is processing your audio message")
            transcription = await loop.run_in_executor(None, self.transcriber.transcribe)
            print(f"Transcription: {transcription}")
            
            await self.process_text_command(transcription, channel)
            os.remove(audio_path)
        except Exception as e:
            await channel.send(f"Error: {e}")

    async def process_text_command(self, text, channel):
        loop = asyncio.get_running_loop()
        try:
            prompt = f"User input: {text}\nRespond to this input:"
            generated_text = await loop.run_in_executor(None, lambda: llm_wrapper.generate_response(prompt))
            
            await channel.send(f"Akane's response: {generated_text}")
        except Exception as e:
            await channel.send(f"Error processing with LLM: {e}")

client = MyClient(whisper_model, processor, intents=intents)
client.run(TOKEN)