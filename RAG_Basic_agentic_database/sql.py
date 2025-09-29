import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")


def chat(messages):
    URL="https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        # "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        # "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
    }
    payload = {
        "model": "x-ai/grok-4-fast:free",
        "messages": messages,
    }
    # MULTI MODEL TEST
    data=json.dumps({
        "model": "x-ai/grok-4-fast:free",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "What is in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            }
            ]
        }
        ],
    
  })
    try:
        response = requests.post(URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None


if __name__ == "__main__":

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello who are you which model and what is your parent company!"},
        {"role": "user", "content": "hey can you give me a good promt so that i give you and you create a sql query to fetch data result from tables joining tables and like agentic database" }
    ]
    response = chat(messages)
    print(response["choices"][0]["message"]["content"])