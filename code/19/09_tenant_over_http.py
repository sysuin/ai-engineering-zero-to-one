# timeout: 300
# Per-tenant scoping over HTTP. The tenant arrives with the request, from a verified
# token — never as a tool argument, because a tool argument is a value a model can
# invent. No model is involved here: the client plays the part of a persuaded one.

import asyncio
import json
import socket
import sqlite3
import threading
import time

import httpx2
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer

# Two tenants, one token each. In production the verifier checks a signed token from
# an identity provider; the point here is only where the tenant comes from.
TOKENS = {"token-for-northeast": "Northeast", "token-for-west": "West"}


class Verifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        tenant = TOKENS.get(token)
        return AccessToken(token=token, client_id="analyst", scopes=["read"],
                           subject=tenant) if tenant else None


with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    PORT = probe.getsockname()[1]
URL = f"http://127.0.0.1:{PORT}"

server = MCPServer(name="scoped", token_verifier=Verifier(), log_level="ERROR",
                   auth=AuthSettings(issuer_url="https://auth.example.com",
                                     resource_server_url=f"{URL}/mcp"))
DB = "file:data/meridian/warehouse/meridian.db?mode=ro"


@server.tool(description="Revenue for a year, for the caller's region.")
def revenue(year: int) -> dict:
    region = get_access_token().subject          # from the verified token, not the model
    with sqlite3.connect(DB, uri=True) as con:
        (value,) = con.execute("SELECT ROUND(SUM(revenue), 2) FROM v_sales "
                               "WHERE region = ? AND year = ?", (region, year)).fetchone()
    return {"region": region, "year": year, "revenue": value}


@server.tool(description="Revenue for a year and a named region.")
def revenue_for_region(year: int, region: str) -> dict:
    # The version that trusts its arguments. Any caller can name any region.
    with sqlite3.connect(DB, uri=True) as con:
        (value,) = con.execute("SELECT ROUND(SUM(revenue), 2) FROM v_sales "
                               "WHERE region = ? AND year = ?", (region, year)).fetchone()
    return {"region": region, "year": year, "revenue": value}


app = server.streamable_http_app()
web = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error"))
threading.Thread(target=web.run, daemon=True).start()
while not web.started:
    time.sleep(0.05)


async def call(token: str | None, tool: str, arguments: dict) -> str:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with httpx2.AsyncClient(headers=headers) as http:
            async with streamable_http_client(f"{URL}/mcp", http_client=http) as streams:
                async with ClientSession(*streams[:2]) as session:
                    await session.initialize()
                    result = await session.call_tool(tool, arguments)
                    answer = json.loads(result.content[0].text)
                    return f"{answer['region']} {answer['year']}: {answer['revenue']:,.2f}"
    except Exception as error:                              # noqa: BLE001
        cause = error
        while getattr(cause, "exceptions", None):           # unwrap task-group errors
            cause = cause.exceptions[0]
        return f"refused ({type(cause).__name__})"


async def main() -> None:
    cases = [
        ("Northeast token, scoped tool", "token-for-northeast", "revenue", {"year": 2025}),
        ("West token, scoped tool", "token-for-west", "revenue", {"year": 2025}),
        ("West token, asks for Northeast", "token-for-west", "revenue_for_region",
         {"year": 2025, "region": "Northeast"}),
        ("no token at all", None, "revenue", {"year": 2025}),
    ]
    for label, token, tool, arguments in cases:
        print(f"{label:31} -> {await call(token, tool, arguments)}")

    # What the refusal looks like on the wire, and where it tells a client to go.
    async with httpx2.AsyncClient() as http:
        reply = await http.post(f"{URL}/mcp", json={"jsonrpc": "2.0", "id": 1,
                                                    "method": "tools/list"})
        print(f"\nunauthenticated POST /mcp -> HTTP {reply.status_code}")
        challenge = reply.headers.get("www-authenticate", "")
        print("  WWW-Authenticate: " + challenge.split(", resource_metadata")[0]
              .replace(", ", ",\n                    "))
        where = challenge.split('resource_metadata="')[-1].rstrip('"').replace(URL, "")
        print(f"  ... and resource_metadata points to {where}")
        metadata = (await http.get(URL + where)).json()
        print(f"  which names the resource {metadata['resource'].replace(URL, '')!r}")
        print(f"  and the authorization server {metadata['authorization_servers'][0]}")


asyncio.run(main())
web.should_exit = True
