import numpy as np
from typing import List, Dict, Any, Tuple

class VectorIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids = []
        self.metadata = []
        self.norms = np.empty((0,), dtype=np.float32)
        self.deleted = np.empty((0,), dtype=bool)
        self.id_map: Dict[str, int] = {}

    def add_batch(self, ids: List[str], vectors: List[List[float]], metadata: List[Dict[str, Any]]):
        if not vectors:
            return
        
        vec_array = np.array(vectors, dtype=np.float32)
        if vec_array.shape[1] != self.dimension:
            raise ValueError(f"Expected vector dimension {self.dimension}, got {vec_array.shape[1]}")
        
        start_idx = len(self.ids)
        for i, doc_id in enumerate(ids):
            if doc_id in self.id_map:
                old_idx = self.id_map[doc_id]
                self.deleted[old_idx] = True
            self.id_map[doc_id] = start_idx + i
        
        norms = np.linalg.norm(vec_array, axis=1)
        # Avoid division by zero
        norms[norms == 0] = 1e-10

        self.vectors = np.vstack([self.vectors, vec_array])
        self.norms = np.concatenate([self.norms, norms])
        self.deleted = np.concatenate([self.deleted, np.zeros(len(ids), dtype=bool)])
        self.ids.extend(ids)
        self.metadata.extend(metadata)

    def delete_batch(self, ids: List[str]):
        if not ids:
            return
        for doc_id in ids:
            if doc_id in self.id_map:
                idx = self.id_map[doc_id]
                self.deleted[idx] = True
                del self.id_map[doc_id]

    def prune_deleted(self):
        if len(self.deleted) == 0:
            return
        keep_mask = ~self.deleted
        if keep_mask.all():
            return
            
        self.vectors = self.vectors[keep_mask]
        self.norms = self.norms[keep_mask]
        self.deleted = self.deleted[keep_mask]
        
        new_ids = []
        new_metadata = []
        self.id_map.clear()
        
        for i, keep in enumerate(keep_mask):
            if keep:
                new_idx = len(new_ids)
                doc_id = self.ids[i]
                new_ids.append(doc_id)
                new_metadata.append(self.metadata[i])
                self.id_map[doc_id] = new_idx
        
        self.ids = new_ids
        self.metadata = new_metadata

    def search(self, query_vector: List[float], k: int, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if len(self.ids) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        if q_vec.shape[0] != self.dimension:
            raise ValueError(f"Expected query vector dimension {self.dimension}, got {q_vec.shape[0]}")

        # Filter indices
        valid_indices = []
        for i, meta in enumerate(self.metadata):
            if self.deleted[i]:
                continue
            match = True
            if filter_metadata:
                for key, val in filter_metadata.items():
                    if meta.get(key) != val:
                        match = False
                        break
            if match:
                valid_indices.append(i)
        
        if not valid_indices:
            return []
        
        target_vectors = self.vectors[valid_indices]
        target_norms = self.norms[valid_indices]

        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1e-10

        # Cosine similarity: dot product / (norm(a) * norm(b))
        dot_products = np.dot(target_vectors, q_vec)
        similarities = dot_products / (target_norms * q_norm)

        # Get top k
        k = min(k, len(similarities))
        
        # argpartition is faster than argsort for top k
        if k < len(similarities):
            top_k_idx = np.argpartition(similarities, -k)[-k:]
            # sort the top k by similarity
            top_k_idx = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]]
        else:
            top_k_idx = np.argsort(similarities)[::-1]

        results = []
        for idx in top_k_idx:
            original_idx = valid_indices[idx]
            results.append({
                "id": self.ids[original_idx],
                "score": float(similarities[idx]),
                "metadata": self.metadata[original_idx]
            })

        return results

    def save(self, filepath: str):
        np.savez_compressed(
            filepath,
            vectors=self.vectors,
            norms=self.norms,
            ids=np.array(self.ids, dtype=object),
            metadata=np.array(self.metadata, dtype=object)
        )

    def load(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        self.vectors = data['vectors']
        self.norms = data['norms']
        self.ids = data['ids'].tolist()
        self.metadata = data['metadata'].tolist()
        self.deleted = np.zeros(len(self.ids), dtype=bool)
        self.id_map = {doc_id: i for i, doc_id in enumerate(self.ids)}

