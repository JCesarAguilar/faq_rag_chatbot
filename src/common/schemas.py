from pydantic import BaseModel, Field
from typing import List

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
    score: float

class QueryResponse(BaseModel):
    user_question: str
    system_answer: str
    chunks_related: List[RelatedChunk]

class EvaluationResult(BaseModel):
    score: int = Field(ge=0, le=10)
    reason: str = Field(min_length=50)