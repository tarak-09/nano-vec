import json
import os
import aiofiles

class WAL:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file = None

    async def open(self):
        # Create file if it doesn't exist
        if not os.path.exists(self.filepath):
            open(self.filepath, 'a').close()
        self._file = await aiofiles.open(self.filepath, mode='a')

    async def append(self, entry: dict):
        if self._file is None:
            await self.open()
        line = json.dumps(entry) + '\n'
        await self._file.write(line)
        await self._file.flush()

    async def close(self):
        if self._file:
            await self._file.close()
            self._file = None

    def read_all(self):
        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath, 'r') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
