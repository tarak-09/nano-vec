# Vector DB

A lightweight, in-memory vector database exposing an HTTP API for exact K-Nearest Neighbors (KNN) search via cosine similarity, with Write-Ahead Log (WAL) durability. 

## Features
- **In-Memory Index**: Fast exact KNN search based on cosine similarity, backed by NumPy.
- **WAL Durability**: Data is persisted sequentially to an append-only JSONL Write-Ahead Log. On startup, the WAL is replayed to reconstruct the in-memory index.
- **Metadata Filtering**: Supports attaching metadata to vectors and exactly matching subset fields during search.
- **Batch Insertion & Deletion**: Efficient HTTP endpoints to insert multiple vectors at once, or soft-delete vectors by ID (pruned during snapshots).

## Installation

This project manages dependencies via `uv`. Ensure `uv` is installed, then run:

```bash
# Create a virtual environment and install dependencies
uv venv
uv pip install -e .
```

For development (including tests):
```bash
uv pip install -e ".[dev]"
```

## Usage

### 1. Start the Server

Run the server module as a script, specifying the vector dimension:

```bash
python -m vector_db.server --dim 128 --port 8080 --data-dir ./data --snapshot-interval 3600
```
- `--dim`: Vector dimension (required)
- `--port`: Port to listen on (default: 8080)
- `--data-dir`: Directory for WAL and snapshots (default: data)
- `--snapshot-interval`: Interval in seconds for background snapshotting (default: 0, disabled)

### 2. Insert Vectors

Use the `/insert` endpoint to add batch vectors:

```bash
curl -X POST http://localhost:8080/insert -H "Content-Type: application/json" -d '{
  "items": [
    {
      "id": "item1",
      "vector": [1.0, 0.5, 0.2, 0.0],
      "metadata": {"category": "sports", "tag": "news"}
    },
    {
      "id": "item2",
      "vector": [0.1, 0.9, 0.3, 0.1],
      "metadata": {"category": "tech", "tag": "news"}
    }
  ]
}'
```
*(Ensure vectors have length matching `--dim` parameter, e.g., 4 in this example).*

### 3. Search Vectors

Find the closest matches using exact KNN, optionally filtered by metadata:

```bash
curl -X POST http://localhost:8080/search -H "Content-Type: application/json" -d '{
  "vector": [1.0, 0.0, 0.0, 0.0],
  "k": 5,
  "filter": {
    "tag": "news"
  }
}'
```

### 4. Delete Vectors

Use the `/delete` endpoint to soft-delete vectors by ID:

```bash
curl -X POST http://localhost:8080/delete -H "Content-Type: application/json" -d '{
  "ids": ["item1"]
}'
```
Deleted items are filtered out of search results immediately and permanently pruned from disk during the next snapshot.

### 5. Create a Snapshot

Force an immediate compaction of the in-memory index to disk and truncate the Write-Ahead Log:

```bash
curl -X POST http://localhost:8080/snapshot
```
This is useful to speed up recovery times and prevent infinite WAL growth. Background snapshotting can also be enabled using the `--snapshot-interval` flag.

## Running Tests

Tests use `pytest` and `pytest-asyncio`. Run them with:

```bash
uv run pytest tests/
```
