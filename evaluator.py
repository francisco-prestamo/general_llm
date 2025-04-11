from fireworks.client import Fireworks
import os
import numpy as np
from sklearn.preprocessing import normalize
from dotenv import load_dotenv  # Importar dotenv
import openai
# Cargar variables de entorno desde el archivo .env
load_dotenv()

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

def vectorizar_texto(model, promt) -> np.ndarray:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY no encontrado en el archivo .env")
    
    client = Fireworks(api_key=api_key)
    response = client.embeddings.create(
        model=model,
        input=[promt]
    )
    vector = np.array(response.data[0].embedding)
    return normalize([vector])[0]  # Normalización L2

def main():
    # Configuración
    model_response = "accounts/fireworks/models/llama-v3p3-70b-instruct"  # Modelo para generar respuestas
    model_vectorize = "nomic-ai/nomic-embed-text-v1.5"  # Modelo para vectorizar
    prompt = "Cual es el color del caballo blanco de maceo"

    try:
        # Obtener respuesta del modelo
        print("Obteniendo respuesta del modelo...")
        response_text = Response(model=model_response, prompt=prompt)
        print(f"Respuesta obtenida: {response_text}")

        # Vectorizar la respuesta
        print("Vectorizando la respuesta...")
        vector = vectorizar_texto(model=model_vectorize, promt=response_text)
        print(f"Vector obtenido: {vector}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")

if __name__ == "__main__":
    main()