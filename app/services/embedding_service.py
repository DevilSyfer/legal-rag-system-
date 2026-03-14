from sentence_transformers import SentenceTransformer
from app.core.config import GROQ_API_KEY

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts: list[str]) ->list[list[float]]:
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()