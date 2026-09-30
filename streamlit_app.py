import os
import sys

import streamlit as st

# Hace que las importaciones de src/ funcionen igual que cuando corrés
# query.py directamente, sin necesidad de tocar su código.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# En Streamlit Cloud la API key vive en st.secrets, no en un .env local.
# Si está ahí, la copiamos a las variables de entorno ANTES de importar
# query.py (que la lee con os.getenv al importarse).
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # no hay secrets.toml (desarrollo local) -> se usa el .env normal

from query import answer_question  # noqa: E402 (import después del sys.path, a propósito)

MAX_QUERIES_PER_SESSION = 5

st.set_page_config(page_title="Kené Alma Hostal - Chatbot FAQ", page_icon="🏨")

st.title("🏨 Kené Alma Hostal — Chatbot FAQ")
st.caption(
    "Chatbot con RAG (Retrieval-Augmented Generation) construido a mano con "
    "OpenAI + Pydantic + numpy. [Ver código en GitHub](https://github.com/JCesarAguilar/faq_rag_chatbot)"
)

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

question = st.text_input(
    "Preguntame algo sobre el hostal (horarios, precios, mascotas, tours...)"
)

if st.button("Preguntar", type="primary"):
    if not question.strip():
        st.warning("Escribí una pregunta primero.")
    elif st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
        st.error(
            f"Llegaste al límite de {MAX_QUERIES_PER_SESSION} preguntas por sesión "
            "(demo con costo real de API). Recargá la página para reiniciar."
        )
    else:
        with st.spinner("Buscando en la base de conocimiento..."):
            try:
                response = answer_question(question)
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
            else:
                st.session_state.query_count += 1
                st.subheader("Respuesta")
                st.write(response.system_answer)

                with st.expander(
                    f"Ver los {len(response.chunks_related)} fragmentos usados como contexto"
                ):
                    for chunk in response.chunks_related:
                        st.markdown(f"**{chunk.chunk_id}** — score: `{chunk.score:.3f}`")
                        st.text(chunk.text)
                        st.divider()

st.caption(
    f"Preguntas usadas en esta sesión: {st.session_state.query_count}/{MAX_QUERIES_PER_SESSION}"
)