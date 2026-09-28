# Kené Alma Hostal — Chatbot FAQ con RAG

Chatbot de soporte basado en **RAG (Retrieval-Augmented Generation)** que responde preguntas frecuentes sobre Kené Alma Hostal, un hostal ficticio en Pucallpa, Perú. El sistema procesa un documento de FAQ en texto plano, lo divide en fragmentos (chunks), genera embeddings con la API de OpenAI y los almacena para poder recuperar la información relevante ante cada consulta del usuario, generando finalmente una respuesta en lenguaje natural con un LLM. Proyecto integrador del Módulo 2 (RAG) de la especialización AI Engineering de Henry.

## ¿Por qué RAG?

Un LLM por sí solo no conoce las políticas específicas de este hostal (no estaban en sus datos de entrenamiento) y, si se le pregunta igual, puede inventar una respuesta (alucinación). RAG resuelve esto **recuperando** la información real del documento de la empresa antes de generar la respuesta, y forzando al modelo a responder únicamente con ese contexto. Esto trae un beneficio clave: el conocimiento del chatbot se actualiza simplemente editando `data/faq_document.txt` y volviendo a correr la indexación — no hace falta reentrenar ni modificar el modelo.

## Instalación

1. Cloná el repositorio y entrá a la carpeta del proyecto:

```bash
   git clone <url-del-repo>
   cd faq_rag_chatbot
```

2. Creá y activá un entorno virtual:

```bash
   python3 -m venv venv
   source venv/bin/activate
```

3. Instalá las dependencias:

```bash
   pip install -r requirements.txt
```

4. Configurá tu API key de OpenAI:

```bash
   cp .env.example .env
   # editar .env y completar OPENAI_API_KEY=tu-key-real
```

## Uso

**1. Generar el índice de embeddings** (solo la primera vez, o cuando cambie el FAQ):

```bash
python3 src/build_index.py
```

Esto carga `data/faq_document.txt`, lo divide en chunks, genera un embedding por chunk con la API de OpenAI, y guarda todo en `data/index.json`.

**2. Hacer una consulta:**

```bash
python3 src/query.py
```

El script pide una pregunta por consola y devuelve un JSON como este:

```json
{
  "user_question": "¿Cuál es el horario de check-in y check-out?",
  "system_answer": "El horario de check-in es a partir de las 14:00 horas y el check-out debe realizarse antes de las 11:00 horas.",
  "chunks_related": [
    {
      "chunk_id": "faq_document_chunk_0003",
      "text": "CHECK-IN Y CHECK-OUT...",
      "score": 0.7134
    },
    {
      "chunk_id": "faq_document_chunk_0018",
      "text": "¿Puedo dejar mi equipaje después del check-out...",
      "score": 0.5684
    }
  ]
}
```

Más ejemplos de preguntas y respuestas en `outputs/sample_queries.json`.

## Estructura del proyecto

faq_rag_chatbot/
├── data/
│ ├── faq_document.txt # documento fuente del FAQ
│ └── index.json # chunks + embeddings (generado por build_index.py)
├── prompts/
│ └── main_prompt.txt # template del prompt usado para generar respuestas
├── src/
│ ├── common/
│ │ ├── config.py # configuración (rutas, modelos, parámetros)
│ │ ├── schemas.py # modelos Pydantic (validación de datos)
│ │ └── llm.py # wrappers a la API de OpenAI (embeddings y chat)
│ ├── build_index.py # pipeline de datos: carga → chunking → embeddings → guardado
│ └── query.py # pipeline de consultas: embedding → búsqueda → prompt → respuesta
├── outputs/
│ └── sample_queries.json # ejemplos de preguntas y respuestas
├── requirements.txt
├── .env.example
└── README.md

## Decisiones técnicas

### Chunking: por sección/párrafo, no por tamaño fijo

Se probaron dos estrategias:

1. **Ventana fija** (100 palabras, 20% overlap): generaba 21 chunks técnicamente válidos, pero al medir la calidad de recuperación, solo ~33% de los chunks recuperados eran relevantes — la ventana fija cortaba a mitad de un tema y mezclaba el final de una sección con el inicio de la siguiente.
2. **Por sección/párrafo** (estrategia final, `chunk_by_section()` en `build_index.py`): agrupa bloques de texto separados por líneas en blanco (títulos de sección o preguntas frecuentes) hasta alcanzar un mínimo de 30 palabras (`MIN_CHUNK_WORDS`), respetando los límites temáticos naturales del documento. Genera **20 chunks**, cada uno enfocado en un solo tema, y subió la relevancia medida a ~67%.

**Excepción documentada:** uno de los 20 chunks (la respuesta más breve del FAQ) queda en 44 tokens, por debajo del rango 50-500. Fusionarlo con su vecino lo arregla, pero baja el total a 19 chunks, incumpliendo el mínimo de 20 exigido. Se priorizó el mínimo de cantidad de chunks.

### Búsqueda vectorial: k-NN por fuerza bruta con similitud coseno

Se calcula la similitud coseno (manual, con `numpy`) entre el embedding de la pregunta y el de cada uno de los 20 chunks, devolviendo los `TOP_K=2` más similares. No se usó una base de datos vectorial dedicada (Pinecone/Chroma) porque, con un corpus de este tamaño (<10.000 vectores), la búsqueda exhaustiva es exacta (100% recall) y suficientemente rápida — una base vectorial dedicada solo aporta valor cuando el volumen de datos crece mucho más.

### Almacenamiento

Los embeddings se guardan en un archivo JSON (`data/index.json`) en vez de una base de datos, dado el tamaño reducido del proyecto (20 chunks). Cumple el mismo rol que una base vectorial para este caso de uso: persistencia de los vectores para consultarlos después.

## Limitación conocida: calidad de recuperación

Se midió la calidad de recuperación con 3 preguntas de prueba, obteniendo **~67% de chunks relevantes** (objetivo: ≥80%). La causa principal es el tamaño reducido del corpus (20 chunks, un solo dominio temático — todo sobre el mismo hostal), lo que genera un "piso" de similitud coseno entre chunks de temas distintos, ya que comparten vocabulario y contexto general.

Con `TOP_K=2` se prioriza cumplir el rango de 2-5 chunks por consulta exigido por la consigna, aceptando que en algunas preguntas el segundo chunk recuperado no sea el más relevante. El chunk correcto siempre apareció al menos en el top-2 en las pruebas realizadas, por lo que la respuesta final generada fue correcta en los 3 casos.

**Mejoras futuras posibles:** búsqueda híbrida (combinar similitud semántica con coincidencia de palabras clave) o un paso de reranking posterior a la recuperación inicial.

## Variables de entorno

| Variable          | Descripción                                                     |
| ----------------- | --------------------------------------------------------------- |
| `OPENAI_API_KEY`  | API key de OpenAI (requerida)                                   |
| `EMBEDDING_MODEL` | Modelo de embeddings (default: `text-embedding-3-small`)        |
| `CHAT_MODEL`      | Modelo de chat para generar respuestas (default: `gpt-4o-mini`) |
