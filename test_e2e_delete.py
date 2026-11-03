import asyncio
from vector_db.server import VectorDBServer

async def main():
    server = VectorDBServer(dimension=4, data_dir="data_e2e2")
    app = await server.init_app()
    
    class DummyRequest:
        def __init__(self, data):
            self._data = data
        async def json(self):
            return self._data
            
    # Insert
    req1 = DummyRequest({
        "items": [
            {"id": "item1", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {"ver": 1}}
        ]
    })
    await server.handle_insert(req1)
    
    # Delete
    req2 = DummyRequest({
        "ids": ["item1"]
    })
    res2 = await server.handle_delete(req2)
    print("Delete status:", res2.status)
    
    # Search
    req_search = DummyRequest({
        "vector": [1.0, 0.0, 0.0, 0.0],
        "k": 5
    })
    res_search = await server.handle_search(req_search)
    import json
    body = json.loads(res_search.text)
    print("Search results after delete:", body)
    
    await server.cleanup(app)

if __name__ == "__main__":
    asyncio.run(main())
