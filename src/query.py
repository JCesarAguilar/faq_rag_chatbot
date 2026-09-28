import json
import numpy as np
from common.config import INDEX_PATH, TOP_K, PROMPT_PATH
from common.llm import get_embeddings_batch, generate_answer
from common.schemas import ChunkWithEmbedding, RelatedChunk, QueryResponse


def load_index(path: str = INDEX_PATH) -> list[ChunkWithEmbedding]:
    """Carga el índice de chunks con embeddings generado por build_index.py."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No se encontró el índice en '{path}'. "
            "Corré primero: python3 src/build_index.py"
        )
    return [ChunkWithEmbedding(**item) for item in data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similitud coseno entre dos vectores: producto punto / (norma A x norma B)."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    return float(dot_product / (norm_a * norm_b))


def search_similar_chunks(question: str, chunks: list[ChunkWithEmbedding],
                           top_k: int = TOP_K) -> list[RelatedChunk]:
    """Genera el embedding de la pregunta y devuelve los top_k chunks
    más similares por similitud coseno, ordenados de mayor a menor score."""
    question_embedding = get_embeddings_batch([question])[0]

    scored_chunks = [
        RelatedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            score=cosine_similarity(question_embedding, chunk.embedding),
        )
        for chunk in chunks
    ]

    scored_chunks.sort(key=lambda c: c.score, reverse=True)
    return scored_chunks[:top_k]


def load_prompt_template(path: str = PROMPT_PATH) -> str:
    """Carga el template del prompt desde un archivo de texto plano."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(question: str, related_chunks: list[RelatedChunk]) -> str:
    """Arma el prompt final rellenando el template con el contexto
    recuperado (con procedencia) y la pregunta del usuario."""
    context = "\n\n".join(
        f"[Fragmento {i+1} - score {c.score:.3f}]\n{c.text}"
        for i, c in enumerate(related_chunks)
    )
    template = load_prompt_template()
    return template.format(context=context, question=question)


def answer_question(question: str) -> QueryResponse:
    """Orquesta el pipeline: carga índice → busca chunks relevantes →
    arma prompt → genera respuesta → devuelve el JSON final."""
    chunks = load_index()
    related_chunks = search_similar_chunks(question, chunks)
    prompt = build_prompt(question, related_chunks)
    answer = generate_answer(prompt)

    return QueryResponse(
        user_question=question,
        system_answer=answer,
        chunks_related=related_chunks,
    )


if __name__ == "__main__":
    question = input("Pregunta: ")
    response = answer_question(question)
    print(response.model_dump_json(indent=2))