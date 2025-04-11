from fireworks.client import Fireworks

client = Fireworks(api_key="<FIREWORKS_API_KEY>")
response = client.chat.completions.create(
model="accounts/fireworks/models/llama-v3p3-70b-instruct",
messages=[{
   "role": "user",
   "content": "Say this is a test",
}],
)

print(response.choices[0].message.content)