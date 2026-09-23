import os
from google import genai
from google.genai import types

api_key = "AQ.Ab8RN6LR3vuYIoIJMy17V10oWHSxE597x4ZOr3K8ekyTSywvNg"
client = genai.Client(api_key=api_key)

print("Listing models...")
try:
    for model in client.models.list():
        if "gemini" in model.name.lower():
            print(f"Model: {model.name}, Supported actions: {model.supported_actions}")
except Exception as e:
    print(f"Error listing models: {e}")

print("\nTesting generate_content with gemini-3.6-flash...")
try:
    res = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Hello! Respond with JSON: {\"status\": \"ok\", \"message\": \"test\"}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    print("Success! Response:")
    print(res.text)
except Exception as e:
    print(f"Error with gemini-3.6-flash: {e}")
