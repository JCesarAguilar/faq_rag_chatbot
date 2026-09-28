import re
import json
import tiktoken
from common.config import MIN_CHUNK_WORDS, INDEX_PATH
from common.llm import get_embeddings_batch
from common.schemas import Chunk, ChunkWithEmbedding

encoder = tiktoken.encoding_for_model("gpt-4o-mini")

def load_document(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def chunk_by_section(text: str, source_doc_id: str,
                      min_words: int = MIN_CHUNK_WORDS) -> list[Chunk]:
    """Divide el texto en chunks por bloque temático (cada bloque separado
    por una línea en blanco: un título de sección o una pregunta frecuente).
    Los bloques se van acumulando hasta alcanzar un mínimo de palabras,
    para no generar chunks demasiado cortos (ej. solo un título). A
    diferencia de una ventana de tamaño fijo, cada chunk resultante
    respeta los límites naturales del documento: nunca mezcla dos
    secciones distintas en un mismo chunk."""

    raw_blocks = [
        (m.group().strip(), m.start(), m.end())
        for m in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text, re.DOTALL)
    ]

    chunks: list[Chunk] = []
    buffer_text = ""
    buffer_start = None
    chunk_index = 0

    for block_text, block_start, block_end in raw_blocks:
        if buffer_start is None:
            buffer_start = block_start
        buffer_text = f"{buffer_text}\n\n{block_text}".strip() if buffer_text else block_text

        if len(buffer_text.split()) >= min_words:
            chunks.append(Chunk(
                chunk_id=f"{source_doc_id}_chunk_{chunk_index:04d}",
                source_doc_id=source_doc_id,
                chunk_index=chunk_index,
                text=buffer_text,
                char_start=buffer_start,
                char_end=block_end,
                token_count=len(encoder.encode(buffer_text)),
            ))
            chunk_index += 1
            buffer_text = ""
            buffer_start = None

    if buffer_text:  # residual final, por si el documento no cierra justo en el mínimo
        assert buffer_start is not None
        chunks.append(Chunk(
            chunk_id=f"{source_doc_id}_chunk_{chunk_index:04d}",
            source_doc_id=source_doc_id,
            chunk_index=chunk_index,
            text=buffer_text,
            char_start=buffer_start,
            char_end=raw_blocks[-1][2],
            token_count=len(encoder.encode(buffer_text)),
        ))

    return chunks

def generate_embeddings_for_chunks(chunks: list[Chunk]) -> list[ChunkWithEmbedding]:
    texts = [c.text for c in chunks]
    embeddings = get_embeddings_batch(texts)
    return [
        ChunkWithEmbedding(**c.model_dump(), embedding=emb)
        for c, emb in zip(chunks, embeddings)
    ]

def save_index(chunks: list[ChunkWithEmbedding], path: str = INDEX_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks], f, ensure_ascii=False, indent=2)

def build_index(document_path: str) -> None:
    text = load_document(document_path)
    chunks = chunk_by_section(text, source_doc_id="faq_document")
    print(f"Generados {len(chunks)} chunks.")
    chunks_with_emb = generate_embeddings_for_chunks(chunks)
    print(f"Embeddings generados, dimensión: {len(chunks_with_emb[0].embedding)}")
    save_index(chunks_with_emb)
    print(f"Índice guardado en {INDEX_PATH}")

if __name__ == "__main__":
    build_index("data/faq_document.txt")