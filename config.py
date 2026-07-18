import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Project folders
PDF_FOLDER = "data"
FAISS_INDEX_PATH = "faiss_index"