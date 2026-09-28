from typing import List, Annotated
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    source_doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    token_count: int


class ChunkWithEmbedding(Chunk):
    embedding: List[float]


class RelatedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float = Field(ge=-1.0, le=1.0)


class QueryResponse(BaseModel):
    user_question: str
    system_answer: str
    chunks_related: Annotated[List[RelatedChunk], Field(min_length=2, max_length=5)]


class EvaluationResult(BaseModel):
    score: int = Field(ge=0, le=10)
    reason: str = Field(min_length=50)
