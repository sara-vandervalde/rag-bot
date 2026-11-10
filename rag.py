"""
חלק ב': מענה לשאלה (Query) - רץ בכל שאלה.

שאלה → Embedding (search_query) → חיפוש היברידי (וקטורי + BM25) → Top-4 → פרומפט → LLM → תשובה
(ללא הרחבת שאלה - לפי ההחלטה)
"""
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

# הודעת System - הוראות קבועות. מודלים נותנים להוראות מערכת משקל רב יותר מאשר לתוכן בהודעת המשתמש.
SYSTEM_PROMPT = """אתה עוזר שעונה על שאלות בנושא DevOps אך ורק על סמך הקטעים שתקבל.

כלל עליון - אסור להמציא:
- כל עובדה, הסבר, דוגמה, השוואה או משל בתשובה חייבים להופיע בקטעים.
- אל תשתמש בידע הכללי שלך כמקור תוכן, גם אם אתה יודע את התשובה.
- גם כשמבקשים "דוגמה", "פישוט", "הסבר נוסף" או "משל" - אל תמציא.
  אם אין דוגמה בקטעים, כתוב: "אין דוגמה לכך בחומר." והצג רק את מה שכן מופיע בקטעים.
- מותר לנסח מחדש ולפשט את הניסוח של מה שכתוב בקטעים, בלי להוסיף תוכן חדש.
- אם התשובה לא נמצאת בקטעים כלל, כתוב: "לא מצאתי מידע על כך בחומר."

ציון מקור - חובה:
- לכל קטע יש שדה source_type: "חומר הקורס" (חומר המרצה) או "חומר משלים" (לא מהמרצה).
- אם התשובה מבוססת על "חומר משלים", פתח בשורה:
  "⚠️ תשובה זו מבוססת על חומר משלים, ולא על חומר המרצה."
- אם התשובה משלבת את שני הסוגים, ציין איזה חלק הגיע מחומר משלים.
- אם בקטע מופיע שהסעיף לא נבדק מול מקור רשמי, ציין זאת בתשובה.

שפה: ענה בעברית. השאר מונחים טכניים, פקודות וקוד באנגלית."""

# הודעת המשתמש - הקטעים והשאלה. תזכורת קצרה בסוף, כי מודלים "זוכרים" טוב יותר את מה שקראו אחרון.
USER_TEMPLATE = """--- קטעים ---
{context}
--- סוף הקטעים ---

שאלה: {question}

(תזכורת: ענה רק ממה שכתוב בקטעים. בלי דוגמאות או מידע שאינם בקטעים.)"""


class RAGEngine:
    def __init__(self):
        # אותו מודל Embedding כמו בהכנה - הפעם input_type לשאלה
        embed_model = CohereEmbedding(
            api_key=config.COHERE_API_KEY,
            model_name=config.EMBED_MODEL,
            input_type="search_query",
        )

        # טעינת Chroma הקיים (לא בונים מחדש - זה תפקיד ingest.py)
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        collection = client.get_collection(config.COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

        # שני סוגי החיפוש
        vector_retriever = index.as_retriever(similarity_top_k=config.TOP_K)   # משמעות
        bm25_retriever = BM25Retriever.from_persist_dir(config.BM25_DIR)        # מילות מפתח

        # מודל התשובה - לפי הספק שנבחר ב-config.py.
        # temperature=0 → פחות "יצירתיות", פחות נטייה להוסיף תוכן מעצמו.
        if config.LLM_PROVIDER == "cohere":
            from llama_index.llms.cohere import Cohere
            self.llm = Cohere(model=config.LLM_MODEL, api_key=config.COHERE_API_KEY, temperature=0)
        else:
            from llama_index.llms.groq import Groq
            self.llm = Groq(model=config.LLM_MODEL, api_key=config.GROQ_API_KEY, temperature=0)

        # חיפוש היברידי: מאחד את שתי הרשימות לדירוג אחד ומחזיר Top-4.
        # num_queries=1 → בלי הרחבת שאלה (ברירת המחדל הייתה מייצרת שאלות נוספות עם ה-LLM!)
        self.retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=config.TOP_K,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
            llm=self.llm,
        )

    def ask(self, question: str):
        # שליפה
        nodes = self.retriever.retrieve(question)

        # בניית פרומפט - כל קטע נשלח עם המטא-דאטה שלו (שם קובץ + כותרת)
        context = "\n\n---\n\n".join(
            n.node.get_content(metadata_mode=MetadataMode.LLM) for n in nodes
        )
        user_message = USER_TEMPLATE.format(context=context, question=question)

        # יצירת תשובה - קריאה אחת ל-LLM, דרך ממשק ה-Chat (עובד גם ב-Cohere וגם ב-Groq)
        response = self.llm.chat(
            [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_message),
            ]
        )

        # מחזירים גם את הקטעים - כדי לאבחן: "האם הקטע הנכון נמצא?"
        return response.message.content, nodes
