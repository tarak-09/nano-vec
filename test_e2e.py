import asyncio
from vector_db.server import VectorDBServer
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

async def main():
    server = VectorDBServer(dimension=4, data_dir="data_e2e")
    app = await server.init_app()
    
    # We will use aiohttp TestClient if we want, but we can also just call handlers directly
    # To keep it simple, we'll just test the handlers manually
    class DummyRequest:
        def __init__(self, data):
            self._data = data
        async def json(self):
            return self._data
            
    # Insert v1
    req1 = DummyRequest({
        "items": [
            {"id": "item1", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {"ver": 1}}
        ]
    })
    res1 = await server.handle_insert(req1)
    print("Insert v1:", res1.status)
    
    # Upsert v2
    req2 = DummyRequest({
        "items": [
            {"id": "item1", "vector": [0.0, 1.0, 0.0, 0.0], "metadata": {"ver": 2}}
        ]
    })
    res2 = await server.handle_insert(req2)
    print("Upsert v2:", res2.status)
    
    # Search
    req_search = DummyRequest({
        "vector": [0.0, 1.0, 0.0, 0.0],
        "k": 5
    })
    res_search = await server.handle_search(req_search)
    import json
    body = json.loads(res_search.text)
    print("Search results:", body)
    
    await server.cleanup(app)

if __name__ == "__main__":
    asyncio.run(main())
