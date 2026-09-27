import json
import tiktoken
from common.config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, INDEX_PATH
from common.llm import get_embeddings_batch
from common.schemas import Chunk, ChunkWithEmbedding

encoder = tiktoken.encoding_for_model("gpt-4o-mini")

def load_document(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, source_doc_id: str,
                chunk_size: int = CHUNK_SIZE_WORDS,
                overlap: int = CHUNK_OVERLAP_WORDS) -> list[Chunk]:
    """
    TODO (esta la escribís vos):
    Sliding window sobre text.split(): ventanas de `chunk_size` palabras,
    avanzando de a (chunk_size - overlap) palabras cada vez.
    Por cada ventana arma un Chunk (chunk_id, chunk_index, text,
    char_start/char_end, token_count via `encoder`).
    """
    pass

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
    chunks = chunk_text(text, source_doc_id="faq_document")
    print(f"Generados {len(chunks)} chunks.")
    chunks_with_emb = generate_embeddings_for_chunks(chunks)
    print(f"Embeddings generados, dimensión: {len(chunks_with_emb[0].embedding)}")
    save_index(chunks_with_emb)
    print(f"Índice guardado en {INDEX_PATH}")

if __name__ == "__main__":
    build_index("data/faq_document.txt")