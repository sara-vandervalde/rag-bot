"""
חלק א': הכנת הידע (Indexing) - רץ פעם אחת, ובכל פעם שהמסמכים מתעדכנים.

מסמכים → חלוקה לפי כותרות → פיצול סעיפים ארוכים → Embedding → Chroma
                                                    ↘ אינדקס BM25
"""
import os
import shutil
import time

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


def load_folder(folder: str, source_label: str):
    """טוען את קבצי ה-MD מתיקייה ומסמן כל מסמך בסוג המקור שלו."""
    if not os.path.isdir(folder) or not any(
        f.endswith(".md") for _, _, files in os.walk(folder) for f in files
    ):
        print(f"התיקייה {folder} ריקה או לא קיימת - מדלג")
        return []
    docs = SimpleDirectoryReader(folder, required_exts=[".md"], recursive=True).load_data()
    for d in docs:
        d.metadata["source_type"] = source_label
        # התווית נשלחת ל-LLM (כדי שידע לציין את המקור), אבל לא נכנסת לווקטור
        d.excluded_embed_metadata_keys.append("source_type")
    print(f"נטענו {len(docs)} מסמכים מ-{folder} ({source_label})")
    return docs


def main():
    # 1. טעינת מסמכים משתי תיקיות, כל אחת עם תווית מקור במטא-דאטה
    documents = load_folder(config.DOCS_DIR, config.SOURCE_COURSE) + load_folder(
        config.EXTRA_DOCS_DIR, config.SOURCE_EXTRA
    )
    if not documents:
        print("לא נמצאו קבצי MD. בדקי את התיקיות docs ו-docs_extra.")
        return

    # 2א. חלוקה מבוססת מבנה: כל כותרת (#, ##, ###) מתחילה קטע חדש.
    #     נתיב הכותרות נשמר כמטא-דאטה (header_path), למשל "/Docker/Volumes/"
    section_nodes = MarkdownNodeParser().get_nodes_from_documents(documents)

    # 2ב. פיצול סעיף ארוך מ-500 טוקנים לפי משפטים, עם חפיפה של 15%.
    #     סעיף קצר נשאר שלם - ולכן אין חפיפה בין סעיפים שונים, רק בתוך סעיף שפוצל.
    #     המטא-דאטה (כולל הכותרת) עובר לכל תת-קטע.
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(section_nodes)
    print(f"נוצרו {len(nodes)} קטעים (Chunks)")

    # 3. מודל Embedding - input_type="search_document" כי אלה מסמכים (לשאלה נשתמש ב-search_query)
    embed_model = CohereEmbedding(
        api_key=config.COHERE_API_KEY,
        model_name=config.EMBED_MODEL,
        input_type="search_document",
        embed_batch_size=96,   # עד 96 קטעים בכל קריאת API (המקסימום של Cohere) - פחות קריאות מהמכסה
    )

    # 4. שמירה ב-Chroma. מוחקים אוסף קודם כדי שהרצה חוזרת לא תכפיל קטעים.
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # המפתח החינמי מוגבל ל-100,000 טוקנים בדקה, ולכן שולחים בקבוצות עם הפסקה ביניהן.
    index = VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)
    batch = config.INGEST_BATCH_NODES
    for start in range(0, len(nodes), batch):
        index.insert_nodes(nodes[start:start + batch])
        done = min(start + batch, len(nodes))
        print(f"Embedding: {done}/{len(nodes)} קטעים")
        if done < len(nodes):
            print(f"   ממתין {config.INGEST_PAUSE_SECONDS} שניות (מגבלת המפתח החינמי)...")
            time.sleep(config.INGEST_PAUSE_SECONDS)
    print("הווקטורים נשמרו ב-Chroma")

    # 5. אינדקס מילות מפתח (BM25) - החצי השני של החיפוש ההיברידי. לא דורש API.
    shutil.rmtree(config.BM25_DIR, ignore_errors=True)
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=config.TOP_K)
    bm25.persist(config.BM25_DIR)
    print("אינדקס BM25 נשמר")

    print("✅ ההכנה הסתיימה. אפשר להריץ: streamlit run app.py")


if __name__ == "__main__":
    main()
