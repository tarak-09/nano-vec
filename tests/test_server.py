import pytest
import pytest_asyncio
import os
import tempfile
import json
from aiohttp import web
from vector_db.server import VectorDBServer

@pytest_asyncio.fixture
async def cli(aiohttp_client):
    with tempfile.TemporaryDirectory() as tmpdir:
        server = VectorDBServer(dimension=2, data_dir=tmpdir)
        app = await server.init_app()
        client = await aiohttp_client(app)
        yield client

@pytest.mark.asyncio
async def test_health(cli):
    resp = await cli.get('/health')
    assert resp.status == 200
    text = await resp.text()
    data = json.loads(text)
    assert data["status"] == "ok"
    assert data["index_size"] == 0

@pytest.mark.asyncio
async def test_insert_and_search(cli):
    # Insert
    insert_payload = {
        "items": [
            {"id": "v1", "vector": [1.0, 0.0], "metadata": {"label": "x"}},
            {"id": "v2", "vector": [0.0, 1.0], "metadata": {"label": "y"}}
        ]
    }
    resp = await cli.post('/insert', json=insert_payload)
    assert resp.status == 200
    
    # Search
    search_payload = {
        "vector": [1.0, 0.0],
        "k": 1
    }
    resp = await cli.post('/search', json=search_payload)
    assert resp.status == 200
    data = await resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["id"] == "v1"
    
@pytest.mark.asyncio
async def test_search_with_filter(cli):
    insert_payload = {
        "items": [
            {"id": "v1", "vector": [1.0, 0.1], "metadata": {"label": "x"}},
            {"id": "v2", "vector": [1.0, 0.0], "metadata": {"label": "y"}},
            {"id": "v3", "vector": [0.0, 1.0], "metadata": {"label": "x"}}
        ]
    }
    await cli.post('/insert', json=insert_payload)
    
    search_payload = {
        "vector": [1.0, 0.0],
        "k": 2,
        "filter": {"label": "x"}
    }
    resp = await cli.post('/search', json=search_payload)
    assert resp.status == 200
    data = await resp.json()
    assert len(data["results"]) == 2
    ids = [r["id"] for r in data["results"]]
    assert "v1" in ids
    assert "v3" in ids
    assert "v2" not in ids

@pytest.mark.asyncio
async def test_wal_replay(aiohttp_client):
    with tempfile.TemporaryDirectory() as tmpdir:
        # First server instance
        server1 = VectorDBServer(dimension=2, data_dir=tmpdir)
        app1 = await server1.init_app()
        client1 = await aiohttp_client(app1)
        
        insert_payload = {
            "items": [
                {"id": "v1", "vector": [1.0, 0.0], "metadata": {}}
            ]
        }
        await client1.post('/insert', json=insert_payload)
        await server1.wal.close()
        
        # Second server instance (replays WAL)
        server2 = VectorDBServer(dimension=2, data_dir=tmpdir)
        app2 = await server2.init_app()
        client2 = await aiohttp_client(app2)
        
        resp = await client2.get('/health')
        data = await resp.json()
        assert data["index_size"] == 1
        await server2.wal.close()

@pytest.mark.asyncio
async def test_snapshot(aiohttp_client):
    with tempfile.TemporaryDirectory() as tmpdir:
        # First server instance
        server1 = VectorDBServer(dimension=2, data_dir=tmpdir)
        app1 = await server1.init_app()
        client1 = await aiohttp_client(app1)
        
        insert_payload = {
            "items": [
                {"id": "v1", "vector": [1.0, 0.0], "metadata": {}},
                {"id": "v2", "vector": [0.0, 1.0], "metadata": {}}
            ]
        }
        await client1.post('/insert', json=insert_payload)
        
        # Trigger snapshot
        resp = await client1.post('/snapshot')
        if resp.status != 200:
            print(await resp.json())
        assert resp.status == 200
        
        # Check that snapshot file was created
        assert os.path.exists(os.path.join(tmpdir, "snapshot.npz"))
        
        # Check that WAL file is empty
        wal_path = os.path.join(tmpdir, "wal.jsonl")
        assert os.path.getsize(wal_path) == 0
        
        # Insert another item
        insert_payload2 = {
            "items": [
                {"id": "v3", "vector": [0.7, 0.7], "metadata": {}}
            ]
        }
        await client1.post('/insert', json=insert_payload2)
        assert os.path.getsize(wal_path) > 0
        await server1.wal.close()
        
        # Second server instance (loads snapshot + replays WAL)
        server2 = VectorDBServer(dimension=2, data_dir=tmpdir)
        app2 = await server2.init_app()
        client2 = await aiohttp_client(app2)
        
        resp = await client2.get('/health')
        data = await resp.json()
        assert data["index_size"] == 3
        
        # Verify search works with loaded data
        search_payload = {"vector": [1.0, 0.0], "k": 1}
        resp = await client2.post('/search', json=search_payload)
        data = await resp.json()
        assert data["results"][0]["id"] == "v1"
        
        await server2.wal.close()
