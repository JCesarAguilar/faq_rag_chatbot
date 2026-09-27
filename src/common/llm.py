from openai import OpenAI
from .config import OPENAI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Genera un embedding por cada texto, en una sola llamada a la API.
    Devuelve la lista de vectores en el mismo orden que `texts`."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]

def generate_answer(prompt: str) -> str:
    """Envía el prompt al modelo de chat y devuelve el texto de la respuesta."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # bajo, porque queremos respuestas fieles al contexto, no creativas
    )
    return response.choices[0].message.content or "<no content returned>"