"""
כל ההחלטות שקיבלנו בשלב התכנון - במקום אחד.
שינוי החלטה = שינוי שורה אחת כאן.
"""
import os

# אימות תעודות SSL דרך מאגר התעודות של Windows (ולא של פייתון).
# נדרש ברשת עם סינון/אנטי-וירוס שבודק תעבורה מוצפנת - אחרת פייתון 3.13 דוחה את התעודה.
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

# --- מפתחות API (נקראים מקובץ .env) ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- הידע הארגוני ---
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")   # חומר המרצה (git clone של ריפו הקורס) - לא עולה לגיטהב
EXTRA_DOCS_DIR = "./docs_extra"              # חומר משלים (למשל concepts.md) - כן עולה לגיטהב

# תוויות מקור - נשמרות במטא-דאטה של כל קטע ומוצגות ל-LLM ובממשק
SOURCE_COURSE = "חומר הקורס"
SOURCE_EXTRA = "חומר משלים"

# --- החלטה: Vector DB מקומי ---
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "devops_knowledge"
BM25_DIR = "./bm25_index"                    # אינדקס מילות המפתח (חצי מהחיפוש ההיברידי)

# --- החלטה: חלוקה ---
CHUNK_SIZE = 500                             # טוקנים
CHUNK_OVERLAP = 75                           # 15% מ-500

# --- קצב שליחה ל-Cohere (מפתח ניסיון: 100,000 טוקנים בדקה) ---
INGEST_BATCH_NODES = 150                     # 150 × עד 500 טוקנים ≤ 75,000 - מתחת למגבלה
INGEST_PAUSE_SECONDS = 61                    # הפסקה בין קבוצות

# --- החלטה: Top-K ---
TOP_K = 4                                    # 4 × 500 ≈ 2000 טוקנים לכל שאלה

# --- החלטה: מודלים ---
EMBED_MODEL = "embed-multilingual-v3.0"      # Cohere, רב-לשוני (עברית + מונחים באנגלית)

# ספק ה-LLM: "cohere" או "groq".
# Groq חסום כרגע בנטפרי (נתיב ה-chat). אם תאושר פתיחה - לשנות ל-"groq".
LLM_PROVIDER = "cohere"
LLM_MODEL_COHERE = "command-a-03-2025"       # Command A, רב-לשוני
LLM_MODEL_GROQ = "llama-3.1-8b-instant"      # Llama 3.1 8B
LLM_MODEL = LLM_MODEL_COHERE if LLM_PROVIDER == "cohere" else LLM_MODEL_GROQ
