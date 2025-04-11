from fireworks.client import Fireworks
import os
import numpy as np
from sklearn.preprocessing import normalize

def Response(model, prompt):
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY no encontrado en el archivo .env")
    
    client = Fireworks(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
        }],
    )

    return response.choices[0].message.content

def Vectorize(model, promt):
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY no encontrado en el archivo .env")
    
    client = Fireworks(api_key=api_key)
    response = client.embeddings.create(
        model="test-embeding-ada-002",
        input=promt,
    )
    return response.data[0].embedding

def vectorizar_texto(texto: str, modelo: str = "fireworks-ai/text-embedding") -> np.ndarray:
    response = Fireworks.client.Embedding.create(
        model=modelo,
        input=[texto]
    )
    vector = np.array(response.data[0].embedding)
    return normalize([vector])[0]  # Normalización L2

def main():
    # Configuración
    model_response = "gpt-3.5-turbo"  # Modelo para generar respuestas
    model_vectorize = "text-embedding-ada-002"  # Modelo para vectorizar
    prompt = ""

    try:
        # Obtener respuesta del modelo
        print("Obteniendo respuesta del modelo...")
        response_text = Response(model=model_response, prompt=prompt)
        print(f"Respuesta obtenida: {response_text}")

        # Vectorizar la respuesta
        print("Vectorizando la respuesta...")
        vector = Vectorize(model=model_vectorize, promt=response_text)
        print(f"Vector obtenido: {vector}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")

if __name__ == "__main__":
    main()