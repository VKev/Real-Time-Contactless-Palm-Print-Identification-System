from typing import List, Optional, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.models import Filter, FilterSelector, ScoredPoint
import random

client = QdrantClient(
    host="localhost",
    port=6333,
    grpc_port=6334,
    prefer_grpc=True,
)

def ensure_collection(
    collection_name: str,
    vector_size: int,
    distance: Distance = Distance.EUCLID,
) -> None:
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

def insert_vectors(
    collection_name: str,
    vectors: List[List[float]],
    ids: Optional[List[int]] = None,
    payloads: Optional[List[Dict]] = None,
) -> None:
    points = []
    for idx, vec in enumerate(vectors):
        point_id = ids[idx] if ids is not None else None
        point_payload = payloads[idx] if payloads is not None else None
        points.append(
            PointStruct(id=point_id, vector=vec, payload=point_payload)
        )
    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )
    print(f"Inserted {len(points)} points into '{collection_name}'")

def list_vectors(collection_name: str, batch_size: int = 100) -> None:
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        for point in points:
            print(f"ID: {point.id}\n  Payload: {point.payload}\n  Vector: {point.vector}\n")
        
        if next_offset is None:
            break

        offset = next_offset

def get_total_count(
    collection_name: str,
    exact: bool = True
) -> int:

    result = client.count(
        collection_name=collection_name,
        exact=exact
    )
    total = result.count
    print(f"Total points in '{collection_name}': {total}")
    return total

def delete_all_points(
    collection_name: str,
) -> None:

    selector = FilterSelector(filter=Filter(must=[]))

    client.delete(
        collection_name=collection_name,
        points_selector=selector,
        wait=True,
    )

    print(f"Deleted all points from '{collection_name}'")

def search_vectors(
    collection_name: str,
    query_vector: List[float],
    top_k: int = 10
) -> List[PointStruct]:
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=True
    )
    pts: List[ScoredPoint] = results.points
    print(f"{'ID':>6}  {'Score':>8}  {'Label'}")
    print("-" * 28)

    for p in pts:
        label = p.payload.get("label", "")
        print(f"{p.id:6}  {p.score:8.4f}  {label}")

    return results

def delete_collection(collection_name: str) -> None:
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' has been deleted.")
    else:
        print(f"Collection '{collection_name}' does not exist.")

if __name__ == "__main__":
    COL = "palm_vectors"
    DIM = 128

    delete_collection(COL)
    
    ensure_collection(COL, vector_size=DIM, distance=Distance.EUCLID)

    vec= [random.random() for _ in range(DIM)]
    sample_vectors = [
        vec,
        [10 * x for x in vec]
    ]
    sample_ids     = [101, 102]
    sample_payload = [{"label": "palm_101"}, {"label": "palm_102"}]

    insert_vectors(
        collection_name=COL,
        vectors=sample_vectors,
        ids=sample_ids,
        payloads=sample_payload,
    )

    list_vectors(COL)
    
    get_total_count(COL)
    
    query = [20 * x for x in vec]
    search_vectors(COL, query_vector=query, top_k=5)
