import openai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    # Use the ChatCompletion endpoint with the recommended model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Replace with "gpt-4" if available
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, OpenAI!"}
        ],
        max_tokens=50,
        temperature=0.7
    )
    print("API Response:", response.choices[0].message.content)
except openai.error.OpenAIError as e:
    print("OpenAI API Error:", e)
except Exception as e:
    print("Error:", e)





