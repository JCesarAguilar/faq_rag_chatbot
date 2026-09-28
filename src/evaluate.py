import json
from common.config import EVALUATOR_PROMPT_PATH
from common.llm import evaluate_answer
from common.schemas import RelatedChunk, EvaluationResult


def load_evaluator_prompt_template(path: str = EVALUATOR_PROMPT_PATH) -> str:
    """Carga el template del prompt del evaluador desde un archivo de texto plano."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_evaluator_prompt(question: str, answer: str,
                            chunks_related: list[RelatedChunk]) -> str:
    """Arma el prompt para el agente evaluador con la pregunta original,
    la respuesta generada y los chunks de contexto usados."""
    context = "\n\n".join(
        f"[Fragmento {i+1} - score {c.score:.3f}]\n{c.text}"
        for i, c in enumerate(chunks_related)
    )
    template = load_evaluator_prompt_template()
    return template.format(question=question, answer=answer, context=context)


def evaluate_response(question: str, answer: str,
                       chunks_related: list[RelatedChunk]) -> EvaluationResult:
    """Orquesta la evaluación: arma el prompt y le pide al LLM que puntúe
    la respuesta con salida estructurada (EvaluationResult)."""
    prompt = build_evaluator_prompt(question, answer, chunks_related)
    return evaluate_answer(prompt)


if __name__ == "__main__":
    with open("outputs/sample_queries.json", "r", encoding="utf-8") as f:
        samples = json.load(f)

    for sample in samples:
        chunks = [RelatedChunk(**c) for c in sample["chunks_related"]]
        result = evaluate_response(
            sample["user_question"], sample["system_answer"], chunks
        )
        sample["evaluation"] = result.model_dump()
        print(f"Pregunta: {sample['user_question']}")
        print(f"Score: {result.score}/10")
        print(f"Razón: {result.reason}")
        print("-" * 60)

    with open("outputs/sample_queries.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print("Evaluaciones guardadas en outputs/sample_queries.json")