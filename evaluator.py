from fireworks.client import Fireworks

def Response(model, prompt):
    client = Fireworks(api_key="fw_3ZPqtKzG9NqJXzWPuxpjiYKo")
    response = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    )

    return response.choices[0].message.content

