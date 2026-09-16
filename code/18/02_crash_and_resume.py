# timeout: 1200
# Kill the process mid-run. Start a new one. Continue from where it stopped.

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

sys.path.insert(0, "code")
from clarity.v0_10.graph import build                    # noqa: E402
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from meridian_index import load_index                    # noqa: E402

STORE = Path("data/meridian/crash.db")
STORE.unlink(missing_ok=True)
THREAD = "crash-demo"

print("--- first process ---")
child = subprocess.run([sys.executable, "code/18/_crash_child.py", str(STORE), THREAD],
                       capture_output=True, text=True, timeout=300)
print(child.stdout.strip())
print(f"exit code: {child.returncode}   (killed, not raised \u2014 no cleanup ran)")

connection = sqlite3.connect(STORE, check_same_thread=False)
rows = connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
print(f"checkpoints left behind: {rows}")

print("\n--- second process ---")
chunks, vectors = load_index()
app = build(build_tools(Retriever(chunks, vectors), Warehouse())).compile(
    checkpointer=SqliteSaver(connection))
config = {"configurable": {"thread_id": THREAD}}

snapshot = app.get_state(config)
taken = snapshot.values["steps"]
print(f"recovered state: {taken} step{'s' if taken != 1 else ''} already taken, "
      f"{len(snapshot.values['messages'])} messages")
print(f"next node to run: {snapshot.next}")

# Passing None means "carry on from the checkpoint" rather than "start again".
answer = None
for event in app.stream(None, config, stream_mode="updates"):
    for node, update in event.items():
        if node == "act":
            print(f"  act    <- {update['messages'][0].content[:62]}")
        elif node == "think":
            m = update["messages"][0]
            if m.tool_calls:
                for call in m.tool_calls:
                    print(f"  think  -> {call['name']}({str(call['args'])[:40]})")
            elif m.content:
                answer = m.content

print(f"\nA: {(answer or '')[:300]}")

print("\n--- the same crash, with durability='sync' ---")
SYNC = Path("data/meridian/crash-sync.db")
SYNC.unlink(missing_ok=True)
subprocess.run([sys.executable, "code/18/_crash_child.py", str(SYNC), THREAD, "sync"],
               capture_output=True, text=True, timeout=300)
synced = build(build_tools(Retriever(chunks, vectors), Warehouse())).compile(
    checkpointer=SqliteSaver(sqlite3.connect(SYNC, check_same_thread=False))).get_state(config)
print(f"recovered state: {synced.values['steps']} step(s), "
      f"{len(synced.values['messages'])} messages, next node {synced.next}")
act_again = "act" in snapshot.next
print(f"the default run {'lost' if act_again else 'kept'} the finished act and "
      f"{'ran it again' if act_again else 'did not repeat it'}; the sync run "
      f"{'kept it' if 'act' not in synced.next else 'lost it too'}")
SYNC.unlink(missing_ok=True)

json.dump({"exit_code": child.returncode, "checkpoints": rows,
           "recovered_steps": snapshot.values["steps"],
           "recovered_messages": len(snapshot.values["messages"]),
           "next_node": list(snapshot.next)},
          open("code/18/_crash.json", "w"), indent=1)
print()
print("The first process died. It did not save anything on the way out, it did not")
print("catch a signal, and it did not finish. What had been checkpointed survived. By")
print("default a checkpoint is stored while the next step starts, so the step that had")
print("just finished can be lost with the process; with durability='sync' it is stored")
print("before the next step begins, at the cost of waiting for the write.")
print()
print("Chapter 17's ninety lines cannot do this, and adding it is not a small change:")
print("the loop has to stop being a `while` and start being a sequence of transitions")
print("over serialisable state. That is the same thing as writing the framework.")
