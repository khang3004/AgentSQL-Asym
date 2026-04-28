import os
import httpx

api_key = os.environ.get("GROQ_API_KEY")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
payload = {
    "model": "llama-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.0,
    "max_tokens": 10
}
with httpx.Client() as client:
    resp = client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    print(resp.status_code)
    print(resp.text)
