"""
הקליינט - Streamlit.
מציג את התשובה וגם את הקטעים שנשלפו (לאבחון: שליפה או יצירה?).
הרצה: streamlit run app.py
"""
import streamlit as st

import config
from rag import RAGEngine

st.set_page_config(page_title="בוט DevOps", page_icon="🤖")

# יישור לימין לעברית; קוד נשאר משמאל לימין
st.markdown(
    """
    <style>
    .stApp, .stChatMessage, .stMarkdown { direction: rtl; text-align: right; }
    code, pre { direction: ltr; text-align: left; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 בוט הידע של קורס DevOps")


@st.cache_resource   # טוען את המנוע פעם אחת, לא בכל שאלה
def load_engine():
    return RAGEngine()


try:
    engine = load_engine()
except Exception as e:
    st.error(f"לא ניתן לטעון את המערכת. האם הרצת קודם `python ingest.py`?\n\n{e}")
    st.stop()

with st.sidebar:
    st.header("⚙️ הגדרות המערכת")
    st.markdown(
        f"""
- **Embedding:** `{config.EMBED_MODEL}`
- **LLM:** `{config.LLM_MODEL}`
- **Vector DB:** Chroma (מקומי)
- **חיפוש:** היברידי (וקטורי + BM25)
- **Chunk:** {config.CHUNK_SIZE} טוקנים, חפיפה {config.CHUNK_OVERLAP}
- **Top-K:** {config.TOP_K}
- **מקורות:** 🟢 חומר הקורס · 🟡 חומר משלים
- **הרחבת שאלה:** ללא
"""
    )


def source_badge(source_type):
    # התגית נקבעת בקוד לפי המטא-דאטה - לא תלויה במודל
    if source_type == config.SOURCE_EXTRA:
        return "🟡 חומר משלים"
    return "🟢 חומר הקורס"


def show_sources(sources):
    with st.expander(f"📚 הקטעים שנשלפו ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(
                f"**{i}. {s['file']}** — {source_badge(s['source_type'])} — "
                f"`{s['header']}` — ציון: {s['score']:.3f}"
            )
            st.text(s["text"])
            st.divider()


if "history" not in st.session_state:
    st.session_state.history = []

# הצגת היסטוריית השיחה
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

question = st.chat_input("שאלי שאלה על חומר הקורס...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("מחפש בחומר ומנסח תשובה..."):
            try:
                answer, nodes = engine.ask(question)
            except Exception as e:
                answer, nodes = f"אירעה שגיאה: {e}", []

        sources = [
            {
                "file": n.node.metadata.get("file_name", "?"),
                "header": n.node.metadata.get("header_path", "/"),
                "source_type": n.node.metadata.get("source_type", config.SOURCE_COURSE),
                "score": n.score or 0.0,
                "text": n.node.get_content(),
            }
            for n in nodes
        ]
        # גיבוי בממשק: אם רוב הקטעים מחומר משלים - מציגים אזהרה גם אם המודל שכח לציין
        extra_count = sum(1 for s in sources if s["source_type"] == config.SOURCE_EXTRA)
        if sources and extra_count > len(sources) / 2 and "⚠️" not in answer:
            st.warning("רוב הקטעים שנשלפו הם מחומר משלים, ולא מחומר המרצה.")
        st.markdown(answer)
        if sources:
            show_sources(sources)

    st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})
