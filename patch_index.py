import re

with open('src/vector_db/index.py', 'r') as f:
    content = f.read()

content = content.replace(
    'def search(self, query_vector: List[float], k: int, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:',
    'def search(self, query_vector: List[float], k: int, filter_metadata: Dict[str, Any] = None, metric: str = "cosine") -> List[Dict[str, Any]]:'
)

search_logic_old = """        # Cosine similarity: dot product / (norm(a) * norm(b))
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
            top_k_idx = np.argsort(similarities)[::-1]"""

search_logic_new = """        if metric == "cosine":
            # Cosine similarity: dot product / (norm(a) * norm(b))
            dot_products = np.dot(target_vectors, q_vec)
            scores = dot_products / (target_norms * q_norm)
            
            k = min(k, len(scores))
            if k < len(scores):
                top_k_idx = np.argpartition(scores, -k)[-k:]
                top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
            else:
                top_k_idx = np.argsort(scores)[::-1]
        elif metric == "euclidean":
            # Euclidean distance: smaller is better, so score is -distance for consistent sorting
            diffs = target_vectors - q_vec
            distances = np.linalg.norm(diffs, axis=1)
            scores = -distances
            
            k = min(k, len(scores))
            if k < len(scores):
                top_k_idx = np.argpartition(scores, -k)[-k:]
                top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
            else:
                top_k_idx = np.argsort(scores)[::-1]
        else:
            raise ValueError(f"Unknown metric: {metric}")"""

content = content.replace(search_logic_old, search_logic_new)

content = content.replace(
    '                "score": float(similarities[idx]),',
    '                "score": float(scores[idx]),'
)

with open('src/vector_db/index.py', 'w') as f:
    f.write(content)

