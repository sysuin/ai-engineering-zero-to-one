# skip
"""Capture one real JSON-RPC exchange for figure 19.4. Not a listing; a data source."""
import json
import subprocess
import sys

FRAMES = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize",
     "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                "clientInfo": {"name": "clarity-client", "version": "0.11.0"}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
     "params": {"name": "query_warehouse",
                "arguments": {"question": "revenue in 2024 Q3"}}},
]

server = subprocess.Popen([sys.executable, "code/clarity/v0_11/mcp_server.py"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, bufsize=1)
exchange = []
for frame in FRAMES:
    server.stdin.write(json.dumps(frame) + "\n")
    server.stdin.flush()
    exchange.append({"direction": "->", "frame": frame})
    if "id" in frame:
        exchange.append({"direction": "<-", "frame": json.loads(server.stdout.readline())})
server.kill()
json.dump(exchange, open("code/19/_wire.json", "w"), indent=1)
print(f"captured {len(exchange)} frames")
