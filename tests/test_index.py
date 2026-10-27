import pytest
import numpy as np
from vector_db.index import VectorIndex

def test_index_initialization():
    index = VectorIndex(3)
    assert index.dimension == 3
    assert len(index.ids) == 0

def test_add_batch():
    index = VectorIndex(3)
    index.add_batch(
        ["1", "2"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [{"category": "A"}, {"category": "B"}]
    )
    assert len(index.ids) == 2
    assert index.vectors.shape == (2, 3)

def test_search_exact_match():
    index = VectorIndex(3)
    index.add_batch(
        ["1", "2", "3"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [{}, {}, {}]
    )
    
    results = index.search([1.0, 0.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0]["id"] == "1"
    assert results[0]["score"] == pytest.approx(1.0)

def test_search_with_filter():
    index = VectorIndex(3)
    index.add_batch(
        ["1", "2", "3"],
        [[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.0, 1.0, 0.0]],
        [{"color": "red"}, {"color": "blue"}, {"color": "red"}]
    )
    
    results = index.search([1.0, 0.0, 0.0], k=2, filter_metadata={"color": "red"})
    assert len(results) == 2
    ids = [r["id"] for r in results]
    assert "1" in ids
    assert "3" in ids
    assert "2" not in ids

def test_invalid_dimension():
    index = VectorIndex(3)
    with pytest.raises(ValueError):
        index.add_batch(["1"], [[1.0, 0.0]], [{}])
        
    index.add_batch(["1"], [[1.0, 0.0, 0.0]], [{}])
    with pytest.raises(ValueError):
        index.search([1.0, 0.0], k=1)

def test_delete_batch():
    index = VectorIndex(3)
    index.add_batch(
        ["1", "2", "3"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [{}, {}, {}]
    )
    index.delete_batch(["2"])
    
    results = index.search([0.0, 1.0, 0.0], k=3)
    ids = [r["id"] for r in results]
    assert "2" not in ids
    assert len(results) == 2

def test_prune_deleted():
    index = VectorIndex(3)
    index.add_batch(
        ["1", "2", "3"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [{}, {}, {}]
    )
    index.delete_batch(["1", "3"])
    index.prune_deleted()
    
    assert len(index.ids) == 1
    assert index.ids[0] == "2"
    assert index.vectors.shape == (1, 3)
    assert len(index.deleted) == 1
    assert not index.deleted[0]
