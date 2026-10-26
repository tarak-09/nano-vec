import numpy as np

vectors = np.array([[1.0, 2.0], [3.0, 4.0]])
ids = ["a", "b"]
metadata = [{"foo": "bar"}, {"baz": "qux"}]

np.savez("test.npz", vectors=vectors, ids=np.array(ids, dtype=object), metadata=np.array(metadata, dtype=object))

data = np.load("test.npz", allow_pickle=True)
print(data['vectors'])
print(data['ids'].tolist())
print(data['metadata'].tolist())
