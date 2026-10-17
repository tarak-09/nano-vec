import numpy as np
from typing import List, Dict, Any, Tuple

class VectorIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids = []
        self.metadata = []
        self.norms = np.empty((0,), dtype=np.float32)

    def add_batch(self, ids: List[str], vectors: List[List[float]], metadata: List[Dict[str, Any]]):
        if not vectors:
            return
        
        vec_array = np.array(vectors, dtype=np.float32)
        if vec_array.shape[1] != self.dimension:
            raise ValueError(f"Expected vector dimension {self.dimension}, got {vec_array.shape[1]}")
        
        norms = np.linalg.norm(vec_array, axis=1)
        # Avoid division by zero
        norms[norms == 0] = 1e-10

        self.vectors = np.vstack([self.vectors, vec_array])
        self.norms = np.concatenate([self.norms, norms])
        self.ids.extend(ids)
        self.metadata.extend(metadata)

    def search(self, query_vector: List[float], k: int, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if len(self.ids) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        if q_vec.shape[0] != self.dimension:
            raise ValueError(f"Expected query vector dimension {self.dimension}, got {q_vec.shape[0]}")

        # Filter indices
        valid_indices = None
        if filter_metadata:
            valid_indices = []
            for i, meta in enumerate(self.metadata):
                match = True
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
        else:
            target_vectors = self.vectors
            target_norms = self.norms
            valid_indices = np.arange(len(self.ids))

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
            original_idx = valid_indices[idx] if filter_metadata else idx
            results.append({
                "id": self.ids[original_idx],
                "score": float(similarities[idx]),
                "metadata": self.metadata[original_idx]
            })

        return results
