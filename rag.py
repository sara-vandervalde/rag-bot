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

PROMPT = """אתה עוזר שעונה על שאלות בנושא DevOps על סמך חומר הקורס בלבד.

הוראות:
- ענה רק על סמך המידע שבקטעים למטה. אל תוסיף ידע מבחוץ.
- אם התשובה לא נמצאת בקטעים, כתוב: "לא מצאתי מידע על כך בחומר הקורס."
- ענה בעברית. השאר מונחים טכניים, פקודות וקוד באנגלית.

--- קטעים מחומר הקורס ---
{context}
--- סוף הקטעים ---

שאלה: {question}

תשובה:"""


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

        # מודל התשובה - לפי הספק שנבחר ב-config.py
        if config.LLM_PROVIDER == "cohere":
            from llama_index.llms.cohere import Cohere
            self.llm = Cohere(model=config.LLM_MODEL, api_key=config.COHERE_API_KEY)
        else:
            from llama_index.llms.groq import Groq
            self.llm = Groq(model=config.LLM_MODEL, api_key=config.GROQ_API_KEY)

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
        prompt = PROMPT.format(context=context, question=question)

        # יצירת תשובה - קריאה אחת ל-LLM, דרך ממשק ה-Chat (עובד גם ב-Cohere וגם ב-Groq)
        response = self.llm.chat([ChatMessage(role="user", content=prompt)])

        # מחזירים גם את הקטעים - כדי לאבחן: "האם הקטע הנכון נמצא?"
        return response.message.content, nodes
