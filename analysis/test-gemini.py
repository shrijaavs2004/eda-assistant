from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

print("✅ Connected to Gemini!")

print("\nAvailable Models:")
for m in client.models.list():
    print(m.name)
