# timeout: 600
# The two primitives that are not tools, and why the distinction is not pedantry.

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PARAMS = StdioServerParameters(command=sys.executable,
                               args=["code/clarity/v0_11/mcp_server.py"])


async def main() -> None:
    with open("data/meridian/server.log", "w") as errlog:
        async with stdio_client(PARAMS, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                print("--- resources: things, not actions ---")
                listed = await session.list_resources()
                templates = await session.list_resource_templates()
                for resource in listed.resources:
                    print(f"  {resource.uri}")
                    print(f"      {resource.description}")
                for template in templates.resource_templates:
                    print(f"  {template.uri_template}   (a template)")
                    print(f"      {template.description}")

                index = await session.read_resource("meridian://documents")
                documents = json.loads(index.contents[0].text)
                print(f"\n  the index lists {len(documents)} documents — the same "
                      f"ones the")
                print(f"  retriever indexes, and no others")

                one = await session.read_resource("meridian://documents/qbr-2024-Q3")
                body = one.contents[0].text
                heading = next(l for l in body.splitlines() if l.startswith("#"))
                print(f"  reading qbr-2024-Q3 -> {len(body):,} bytes, "
                      f"beginning {heading[:34]!r}")

                print("\n--- prompts: expertise the server ships ---")
                prompts = await session.list_prompts()
                for prompt in prompts.prompts:
                    args = ", ".join(a.name for a in (prompt.arguments or []))
                    print(f"  {prompt.name}({args})")
                    print(f"      {prompt.description}")

                filled = await session.get_prompt(
                    "investigate", {"metric": "margin", "period": "2025 Q1"})
                text = filled.messages[0].content.text
                print(f"\n  get_prompt investigate(margin, 2025 Q1):")
                for line in (text[:150] + "…").split(". "):
                    print(f"    {line.strip()}")


asyncio.run(main())
print()
print("Three primitives, and the difference between them is who decides and what")
print("happens when they are wrong.")
print()
print("  a tool      the model chooses it, and it acts. Getting it wrong changes")
print("              something, so it needs budgets, approval and an audit trail.")
print("  a resource  the client chooses it, and reading it changes nothing. A host")
print("              can show a person a list and let them pick. It is a file.")
print("  a prompt    a person chooses it. It is a way for whoever knows the domain")
print("              to ship that knowledge to every client that connects.")
print()
print("Clarity's `investigate` prompt is the phrasing §17.4 arrived at after a bad")
print("afternoon — figure first, then cause, then what could not be verified. It was")
print("in a file only this repository had. Now it arrives with the connection, and")
print("every client asks the way the analysts here ask.")
print()
print("The mistake is exposing a document-reader as a tool. Then the model decides")
print("when to read, you pay for its guesses, and a person never sees the list.")
