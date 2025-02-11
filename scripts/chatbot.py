import requests
import json
import os
from pymongo import MongoClient

# Load Mistral API Key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("⚠️ MISTRAL_API_KEY is not set! Please configure it in your environment variables.")

# Define API endpoint
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Connect to MongoDB
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["career_advisor"]
chat_history_collection = db["chat_history"]

def chat_with_ai(user_query, chat_history=[]):
    """Interact with Mistral API to generate AI responses for user queries and store chat history in MongoDB."""

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-medium",
        "messages": chat_history + [{"role": "user", "content": user_query}],
        "temperature": 0.7
    }

    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raises an error for 4xx or 5xx responses

        ai_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No response from AI.")
        chat_history.append({"role": "assistant", "content": ai_response})

        # Save chat history to MongoDB
        chat_history_collection.insert_one({"user_query": user_query, "ai_response": ai_response})

        return ai_response

    except requests.exceptions.RequestException as e:
        return f"⚠️ Error connecting to AI service: {e}"

if __name__ == "__main__":
    print("💬 AI Career Advisor Chatbot (type 'exit' to quit)")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        response = chat_with_ai(user_input, chat_history)
        print(f"AI: {response}")
