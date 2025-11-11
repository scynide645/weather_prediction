import requests, json
from dotenv import load_dotenv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_FILE = BASE_DIR / 'data' / 'raw' / 'bmkg_raw.json'

load_dotenv()
bmkg_url = os.getenv("BMKG_ENDPOINT")

req = requests.get(bmkg_url)
with open(RAW_FILE, 'w') as f:
    json.dump(req.json(), f)