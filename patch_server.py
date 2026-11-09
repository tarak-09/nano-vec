import re

with open('src/vector_db/server.py', 'r') as f:
    content = f.read()

content = content.replace(
    '    filter: Optional[Dict[str, Any]] = None',
    '    filter: Optional[Dict[str, Any]] = None\n    metric: str = "cosine"'
)

content = content.replace(
    '            results = self.index.search(req.vector, req.k, req.filter)',
    '            results = self.index.search(req.vector, req.k, req.filter, req.metric)'
)

with open('src/vector_db/server.py', 'w') as f:
    f.write(content)

