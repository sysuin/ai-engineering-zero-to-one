"""
Clarity v0.11 — the same tools, behind a protocol.

Nothing in here implements anything. Every tool delegates to the functions Chapter 16
built and Chapters 17 and 18 have been calling ever since, because the moment an MCP
server grows its own copy of the logic you have two systems that disagree on Fridays.

Three rules, and the first one has cost more people an afternoon than the other two
together:

  stdout is the wire      a stray print() before serving puts junk in the protocol
  the schema is the API   a description is read by a model, so write it for one
  expose no more than     the server is a front door, not a back door: same tools,
  the app exposes         same scoping, same refusals
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.v0_6.retrieve import Retriever            # noqa: E402
from clarity.v0_7.warehouse import Warehouse           # noqa: E402
from clarity.v0_8.tools import ToolError, build_tools  # noqa: E402
from meridian_index import SOURCES, load_index         # noqa: E402


def log(message: str) -> None:
    """
    The only safe way for a stdio server to say anything.

    stdout carries JSON-RPC frames. Anything you write there that is not a frame is a
    protocol violation, and the client's parser is under no obligation to be kind
    about it. §19.7 shows exactly what happens.
    """
    print(message, file=sys.stderr, flush=True)


def build(name: str = "clarity") -> MCPServer:
    chunks, vectors = load_index()
    tools = {t.name: t for t in build_tools(Retriever(chunks, vectors), Warehouse())}
    server = MCPServer(name=name, version="0.11.0", log_level="WARNING",
                       instructions="Meridian's documents and sales warehouse. "
                                    "Every figure comes from a tool; nothing is "
                                    "remembered between calls.")

    # The type hints are not decoration: the SDK turns them into the JSON Schema the
    # client advertises to a model. A missing annotation is advertised as a string.
    @server.tool(description=tools["search_documents"].description)
    def search_documents(query: str, limit: int = 5) -> list[dict]:
        return tools["search_documents"].run(query=query, limit=limit)

    @server.tool(description=tools["query_warehouse"].description)
    def query_warehouse(question: str) -> dict:
        # ToolError is Clarity's "the model should read this and try again". Letting it
        # escape as an exception is right: §19.8 shows what the protocol does with it,
        # and why that is different from the server being broken.
        return tools["query_warehouse"].run(question=question)

    @server.tool(description=tools["arithmetic"].description)
    def arithmetic(expression: str) -> float:
        return tools["arithmetic"].run(expression=expression)

    @server.tool(description=tools["today"].description)
    def today(offset_days: int = 0) -> str:
        return tools["today"].run(offset_days=offset_days)

    # ---------------------------------------------------------------- resources
    # A tool is a verb; a resource is a noun. The client decides when to read one,
    # and reading it has no side effects — which is why a host can put resources in
    # front of a person to pick from, and cannot do that with tools.
    # Exactly the folders the retriever indexes, taken from the retriever rather
    # than retyped. A first draft of this line was `rglob("*.md")`, which quietly
    # published the prompt-injection corpus Chapter 29 keeps on disk. A front
    # door that exposes more than the app is not a front door.
    documents = sorted(path for _, folder in SOURCES
                       for path in folder.glob("*.md"))

    @server.resource("meridian://documents",
                     description="Every Meridian document available to Clarity.",
                     mime_type="application/json")
    def document_index() -> str:
        return json.dumps([{"uri": f"meridian://documents/{d.stem}",
                            "name": d.stem, "bytes": d.stat().st_size}
                           for d in documents], indent=1)

    @server.resource("meridian://documents/{name}",
                     description="One Meridian document, as written.",
                     mime_type="text/markdown")
    def document(name: str) -> str:
        for path in documents:
            if path.stem == name:
                return path.read_text()
        # Not a ToolError: no model is going to retry this. It is a bad URI.
        raise ValueError(f"No document named {name!r}.")

    # ---------------------------------------------------------------- prompts
    # The third primitive, and the one people forget: a prompt is a named, versioned
    # piece of expertise the server can ship so that every client asks the same way.
    @server.prompt(description="Investigate a movement in a metric, the way "
                               "Meridian's analysts do.")
    def investigate(metric: str, period: str) -> str:
        return (f"Establish {metric} for {period} from the warehouse, then find the "
                f"passage in the documents that explains the movement. Give the "
                f"figure first, then the cause, then say what you could not verify. "
                f"Never state a number you did not obtain from a tool.")

    return server


if __name__ == "__main__":
    log("clarity mcp server starting on stdio")
    build().run(transport="stdio")
