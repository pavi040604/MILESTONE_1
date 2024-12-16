from dotenv import load_dotenv
import os

load_dotenv()

import vosk
import pyaudio
import json
import os  # For environment variables

# Load Vosk model path from environment variable
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH")

model = vosk.Model(VOSK_MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, 16000)

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)

print("Listening... Speak into the microphone.")

while True:
    data = stream.read(4000)
    if recognizer.AcceptWaveform(data):
        result = recognizer.Result()
        text = json.loads(result)["text"]
        print(f"Transcription: {text}")
