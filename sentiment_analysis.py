from dotenv import load_dotenv
import os

load_dotenv()

import requests

#API key
import requests
import os  


API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
API_KEY = os.getenv("HUGGINGFACE_API_KEY")  

headers = {"Authorization": f"Bearer {API_KEY}"}


def analyze_sentiment(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result[0]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return {"label": "ERROR", "score": 0.0}


if __name__ == "__main__":
    transcription = "The product is amazing and I'm really happy with it."
    sentiment = analyze_sentiment(transcription)
    print(f"Sentiment: {sentiment['label']}, Score: {sentiment['score']}")
