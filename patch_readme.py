import re

with open('README.md', 'r') as f:
    content = f.read()

content = content.replace(
    'Find the closest matches using exact KNN, optionally filtered by metadata:',
    'Find the closest matches using exact KNN (supports `cosine` and `euclidean` metrics), optionally filtered by metadata:'
)

content = content.replace(
    '  "k": 5,\n  "filter": {\n    "tag": "news"\n  }\n}\'',
    '  "k": 5,\n  "filter": {\n    "tag": "news"\n  },\n  "metric": "cosine"\n}\''
)

content = content.replace(
    'Fast exact KNN search based on cosine similarity, backed by NumPy.',
    'Fast exact KNN search based on cosine similarity or Euclidean distance, backed by NumPy.'
)

content = content.replace(
    'A lightweight, in-memory vector database exposing an HTTP API for exact K-Nearest Neighbors (KNN) search via cosine similarity, with Write-Ahead Log (WAL) durability.',
    'A lightweight, in-memory vector database exposing an HTTP API for exact K-Nearest Neighbors (KNN) search via cosine or Euclidean distance, with Write-Ahead Log (WAL) durability.'
)

with open('README.md', 'w') as f:
    f.write(content)

