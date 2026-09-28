from openai import OpenAI
from .config import OPENAI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL
from .schemas import EvaluationResult

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
        temperature=0.0,  # bajo, porque queremos respuestas fieles al contexto, no creativas
    )
    return response.choices[0].message.content or "<no content returned>"

def evaluate_answer(prompt: str) -> EvaluationResult:
    """Envía el prompt de evaluación al modelo pidiendo salida estructurada
    (structured outputs), forzando que la respuesta matchee EvaluationResult
    en vez de solo pedirlo por texto (más robusto, ver notas de Módulo 1)."""
    completion = client.beta.chat.completions.parse(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=EvaluationResult,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError(
            "El modelo no devolvió una evaluación estructurada válida "
            f"(refusal: {completion.choices[0].message.refusal})"
        )
    return result