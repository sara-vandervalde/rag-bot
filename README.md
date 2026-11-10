# בוט RAG לחומר קורס DevOps

## מבנה הפרויקט

| קובץ | תפקיד | מתי רץ |
|---|---|---|
| `config.py` | כל החלטות התכנון במקום אחד | — |
| `ingest.py` | חלק א': טעינה → חלוקה → Embedding → Chroma + BM25 | פעם אחת, ובכל עדכון מסמכים |
| `rag.py` | חלק ב': שאלה → חיפוש היברידי → פרומפט → LLM | בכל שאלה |
| `app.py` | ממשק Streamlit (תשובה + הקטעים שנשלפו) | — |

## הפעלה

1. התקנה:
   ```
   pip install -r requirements.txt
   ```
2. מפתחות API:
   - Cohere (מפתח ניסיון חינמי): https://dashboard.cohere.com - חובה (Embedding + תשובות)
   - Groq (שכבה חינמית): https://console.groq.com - רק אם `LLM_PROVIDER = "groq"`
   - העתיקי את `.env.example` ל-`.env` ומלאי את המפתחות.
3. הורדת חומר הקורס מ-GitHub לתיקייה `docs`:
   ```
   git clone <כתובת-הריפו> docs
   ```
4. הכנת הידע (פעם אחת):
   ```
   python ingest.py
   ```
5. הפעלת הבוט:
   ```
   streamlit run app.py
   ```

כשהמסמכים בריפו מתעדכנים: `git pull` בתוך `docs`, ואז שוב `python ingest.py`.

## ההחלטות ואיפה הן בקוד

| החלטה | בחירה | איפה |
|---|---|---|
| חלוקה | לפי כותרות MD, פיצול סעיף ארוך | `ingest.py` שלבים 2א-2ב |
| Chunk / Overlap | 500 / 75 (15%) | `config.py` |
| Embedding | Cohere embed-multilingual-v3.0 | `config.py` |
| Vector DB | Chroma מקומי | `ingest.py` שלב 4 |
| חיפוש | היברידי (וקטורי + BM25) | `rag.py` – `QueryFusionRetriever` |
| Top-K | 4 | `config.py` |
| הרחבת שאלה | ללא | `rag.py` – `num_queries=1` |
| LLM | Cohere Command A (חלופה: Llama 3.1 8B דרך Groq) | `config.py` – `LLM_PROVIDER` |

## למה Cohere ולא Groq?

ההחלטה המקורית הייתה Llama 3.1 8B דרך Groq. בפועל, סינון האינטרנט (נטפרי) חוסם את נתיב ה-chat של Groq,
ולכן עברנו ל-Cohere. ההחלפה דרשה שינוי בשורה אחת ב-`config.py` בלבד – בלי לגעת ב-Embedding, ב-DB או בחיפוש.

## תקלות שנפתרו בדרך

| תקלה | סיבה | פתרון |
|---|---|---|
| `SSL: CERTIFICATE_VERIFY_FAILED` | סינון אינטרנט מציג תעודה משלו; Python 3.13 קפדני | `truststore` ב-`config.py` |
| `429 rate limit` בהכנה | מפתח ניסיון: 100,000 טוקנים לדקה | שליחה בקבוצות עם הפסקה (`ingest.py`) |
| `Generate API was removed` | הממשק הישן של Cohere נסגר | שימוש ב-`llm.chat` במקום `llm.complete` |

## מגבלה ידועה

BM25 מחפש מילים זהות. בעברית, תחיליות כמו ה/ב/ו/ל ("הקונטיינר" ≠ "קונטיינר") מחלישות את ההתאמה.
המונחים באנגלית (Docker, ArgoCD, pipeline) לא מושפעים – ובשבילם בדיוק בחרנו בחיפוש ההיברידי.
