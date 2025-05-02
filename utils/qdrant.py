from __future__ import annotations

import random
from typing import List, Optional, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FilterSelector,
    ScoredPoint,
)


class QdrantHelper:
    """Thin wrapper around one QdrantClient instance."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
    ) -> None:
        self.client = QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
        )


    def ensure_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.EUCLID,
    ) -> None:
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )

    def delete_collection(self, collection_name: str) -> None:
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            print(f"[INFO] Collection '{collection_name}' deleted.")

    # ---------- point helpers ---------- #

    def insert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: Optional[List[int]] = None,
        payloads: Optional[List[Dict]] = None,
    ) -> None:
        points = [
            PointStruct(
                id=ids[idx] if ids else None,
                vector=vec,
                payload=payloads[idx] if payloads else None,
            )
            for idx, vec in enumerate(vectors)
        ]
        self.client.upsert(collection_name, points=points, wait=True)
        print(f"[INFO] Inserted {len(points)} points into '{collection_name}'")

    def list_vectors(self, collection_name: str, batch_size: int = 100) -> None:
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break
            for p in points:
                print(f"ID: {p.id}\n  Payload: {p.payload}\n  Vector: {p.vector}\n")
            if offset is None:
                break

    def count(self, collection_name: str, exact: bool = True) -> int:
        total = self.client.count(collection_name, exact=exact).count
        print(f"[INFO] Total points in '{collection_name}': {total}")
        return total

    def delete_all_points(self, collection_name: str) -> None:
        self.client.delete(
            collection_name,
            points_selector=FilterSelector(filter=Filter(must=[])),
            wait=True,
        )
        print(f"[INFO] All points deleted from '{collection_name}'")

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
    ) -> List[ScoredPoint]:
        result = self.client.query_points(
            collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        print(f"{'ID':>6}  {'Score':>8}  Label")
        print("-" * 30)
        for p in result.points:
            label = p.payload.get("label", "")
            print(f"{p.id:6}  {p.score:8.4f}  {label}")
        return result.points


if __name__ == "__main__":
    COL = "palm_vectors"
    DIM = 128

    qd = QdrantHelper()

    # qd.delete_collection(COL)
    qd.ensure_collection(COL, vector_size=DIM)

    base_vec = [random.random() for _ in range(DIM)]
    sample_vectors = [base_vec, [10 * x for x in base_vec]]
    sample_ids = [101, 102]
    sample_payloads = [{"label": "palm_101"}, {"label": "palm_102"}]

    qd.insert_vectors(COL, sample_vectors, ids=sample_ids, payloads=sample_payloads)
    qd.list_vectors(COL)
    qd.count(COL)
    
    query_vec = [20 * x for x in base_vec]
    qd.search(COL, query_vec, top_k=5)
