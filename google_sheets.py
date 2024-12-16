from dotenv import load_dotenv
import os

load_dotenv()

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os  


CREDENTIALS_FILE = os.getenv("GOOGLE_SHEETS_CREDENTIALS")

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
client = gspread.authorize(creds)

sheet = client.open("sheet").sheet1


def append_to_sheet(sentiment, transcription):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([timestamp, sentiment, transcription])


if __name__ == "__main__":
    append_to_sheet("POSITIVE", "The product is amazing and I'm really happy with it.")
    print("Data written to Google Sheets!")
