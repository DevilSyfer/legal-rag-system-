from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import QDRANT_URL

client = QdrantClient(url=QDRANT_URL)


def create_collection(collection_name:str)->str:    
    return client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    


def store_chunks(collection_name: str, chunks:list[dict], embeddings: list[list[float]]):
    points = [
        PointStruct(
            id=chunk["chunk_index"],
            vector=embedding,
            payload=chunk
        )
        for chunk,embedding in zip(chunks, embeddings)
    ]
    
    Add_vectors = client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    return {"stored": len(points)}


def search_results(collection_name:str, query_embedding:list[float], limit=3):
    search_result = client.query_points(
        collection_name=collection_name,
        query=list(query_embedding),
        limit=limit
    ).points
    
    return search_result