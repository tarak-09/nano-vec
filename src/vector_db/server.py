import asyncio
from aiohttp import web
from pydantic import BaseModel, ValidationError, Field
from typing import List, Dict, Any, Optional
import os

from vector_db.index import VectorIndex
from vector_db.wal import WAL

class InsertItem(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class InsertRequest(BaseModel):
    items: List[InsertItem]

class SearchRequest(BaseModel):
    vector: List[float]
    k: int = 10
    filter: Optional[Dict[str, Any]] = None

class VectorDBServer:
    def __init__(self, dimension: int, data_dir: str = "data"):
        self.dimension = dimension
        self.data_dir = data_dir
        self.index = VectorIndex(dimension)
        
        os.makedirs(self.data_dir, exist_ok=True)
        self.wal_path = os.path.join(self.data_dir, "wal.jsonl")
        self.wal = WAL(self.wal_path)

    async def init_app(self) -> web.Application:
        # Replay WAL
        print(f"Replaying WAL from {self.wal_path}...")
        count = 0
        for entry in self.wal.read_all():
            if entry["type"] == "insert_batch":
                ids = entry["ids"]
                vectors = entry["vectors"]
                metadata = entry["metadata"]
                self.index.add_batch(ids, vectors, metadata)
                count += len(ids)
        print(f"Replayed {count} items from WAL.")

        await self.wal.open()

        app = web.Application()
        app.router.add_post('/insert', self.handle_insert)
        app.router.add_post('/search', self.handle_search)
        app.router.add_get('/health', self.handle_health)
        
        # Ensure wal is closed on exit
        app.on_cleanup.append(self.cleanup)
        return app

    async def cleanup(self, app):
        await self.wal.close()

    async def handle_insert(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            req = InsertRequest(**data)
        except ValidationError as e:
            return web.json_response({"error": "Validation error", "details": e.errors()}, status=400)
        except Exception as e:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        ids = []
        vectors = []
        metadata = []

        for item in req.items:
            if len(item.vector) != self.dimension:
                return web.json_response(
                    {"error": f"Vector dimension mismatch. Expected {self.dimension}, got {len(item.vector)} for id {item.id}"}, 
                    status=400
                )
            ids.append(item.id)
            vectors.append(item.vector)
            metadata.append(item.metadata)

        # Write to WAL first
        wal_entry = {
            "type": "insert_batch",
            "ids": ids,
            "vectors": vectors,
            "metadata": metadata
        }
        await self.wal.append(wal_entry)

        # Update index
        self.index.add_batch(ids, vectors, metadata)

        return web.json_response({"status": "success", "inserted": len(ids)})

    async def handle_search(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            req = SearchRequest(**data)
        except ValidationError as e:
            return web.json_response({"error": "Validation error", "details": e.errors()}, status=400)
        except Exception as e:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        if len(req.vector) != self.dimension:
            return web.json_response(
                {"error": f"Vector dimension mismatch. Expected {self.dimension}, got {len(req.vector)}"}, 
                status=400
            )

        try:
            results = self.index.search(req.vector, req.k, req.filter)
            return web.json_response({"results": results})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "index_size": len(self.index.ids)})

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vector DB Server")
    parser.add_argument("--dim", type=int, required=True, help="Vector dimension")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for WAL")
    
    args = parser.parse_args()
    
    server = VectorDBServer(dimension=args.dim, data_dir=args.data_dir)
    
    async def run_server():
        app = await server.init_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', args.port)
        await site.start()
        print(f"Server started at http://0.0.0.0:{args.port}")
        
        # Keep running
        while True:
            await asyncio.sleep(3600)
            
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
