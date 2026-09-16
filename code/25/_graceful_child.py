# skip
"""The server for 10_graceful_shutdown: one slow endpoint, a chosen grace period."""
import asyncio
import sys

import uvicorn
from fastapi import FastAPI

port, grace, work = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
app = FastAPI()


@app.get("/ask")
async def ask() -> dict:
    await asyncio.sleep(work)
    return {"answer": "done"}


@app.get("/ready")
async def ready() -> dict:
    return {"ok": True}


uvicorn.run(app, host="127.0.0.1", port=port, log_level="error", timeout_graceful_shutdown=grace)
