import vosk
import pyaudio
import json
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()


import vosk
import pyaudio
import json
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os 

# I am Loading sensitive data from environment variables
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GOOGLE_SHEETS_CREDENTIALS = os.getenv("GOOGLE_SHEETS_CREDENTIALS")

print("Loading Vosk model...")
model = vosk.Model(VOSK_MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, 16000)

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
print("Vosk Model loaded. Listening...")

# Hugging Face 
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}


def analyze_sentiment(text):
    """Analyze sentiment using Hugging Face API."""
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result[0] if isinstance(result, list) and len(result) > 0 else {"label": "ERROR", "score": 0.0}
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return {"label": "ERROR", "score": 0.0}


# Google Sheets 
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_SHEETS_CREDENTIALS, scope)
client = gspread.authorize(creds)
sheet = client.open("sheet").sheet1


def append_to_sheet(sentiment, transcription):
    """Append the sentiment and transcription to Google Sheets."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([timestamp, sentiment['label'], sentiment['score'], transcription])


print("Ready for real-time speech analysis. Speak into the microphone.")

try:
    while True:
        data = stream.read(4000)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            transcription = result.get("text", "")
            if transcription.strip():
                print(f"Transcription: {transcription}")

                
                sentiment = analyze_sentiment(transcription)
                print(f"Sentiment: {sentiment['label']}, Score: {sentiment['score']}")

                # Save data
                append_to_sheet(sentiment, transcription)
                print("Data saved to Google Sheets.")
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
