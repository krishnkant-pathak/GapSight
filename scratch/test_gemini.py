import asyncio
import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

async def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"Loaded Key: {api_key[:10]}...{api_key[-10:] if api_key else ''}")
    
    if not api_key:
        print("Error: No key loaded")
        return
        
    client = genai.Client(api_key=api_key)
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hello"
        )
        print("Success! Gemini response:")
        print(response.text)
    except APIError as e:
        print(f"APIError occurred: {e}")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
