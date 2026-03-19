from dotenv import load_dotenv
import os
import logging

load_dotenv()  # reads .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # reads a specific variable

QDRANT_URL = os.getenv("QDRANT_URL")
