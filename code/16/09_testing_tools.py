# Test the tools without a model, and the loop with a fake one. Nothing here
# calls an API, needs a key or touches the network. Each test_ function would
# run unchanged under pytest.

import sys
import time
from types import SimpleNamespace

sys.path.insert(0, "code")
from clarity.v0_8.tools import (Tool, ToolError, ToolRunner,  # noqa: E402
                                build_tools)


class EmptyRetriever:
    def search(self, query, k=5):
        return []


class RefusingWarehouse:
    last_refusal = "no metric for headcount"

    def ask(self, question):
        return None


TOOLS = build_tools(EmptyRetriever(), RefusingWarehouse())
run_tool = {t.name: t.run for t in TOOLS}


def call(id_, name, arguments):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=id_, function=function)


def reply(*calls, content=None):
    return SimpleNamespace(content=content, tool_calls=list(calls) or None)


class FakeClient:
    """Replays scripted replies in order and records every request sent."""

    def __init__(self, *script):
        self.script, self.requests = list(script), []
        completions = SimpleNamespace(create=self.create)
        self.chat = SimpleNamespace(completions=completions)

    def create(self, **request):
        self.requests.append([*request["messages"]])
        message = self.script.pop(0) if self.script else self.last
        self.last = message
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


# --------------------------------------------------------- the tools, alone
def test_arithmetic_refuses_a_function_call():
    try:
        run_tool["arithmetic"]("__import__('os').system('rm -rf /')")
    except ToolError as error:
        assert "Call" in str(error) and "Retry" in str(error)
    else:
        raise AssertionError("evaluated a function call")


def test_arithmetic_refuses_attribute_access():
    try:
        run_tool["arithmetic"]("(1).__class__")
    except ToolError:
        return
    raise AssertionError("allowed attribute access")


def test_arithmetic_computes():
    result = run_tool["arithmetic"]("(8461842 - 5661205) / 8461842")
    assert result == 0.330972


def test_warehouse_refusal_says_where_to_look():
    try:
        run_tool["query_warehouse"]("How many people work in Columbus?")
    except ToolError as error:
        assert "search_documents" in str(error)
    else:
        raise AssertionError("no refusal")


# --------------------------------------------------------- the loop, faked
def test_results_are_paired_by_id():
    fake = FakeClient(reply(call("a", "arithmetic", '{"expression": "2+2"}'),
                            call("b", "arithmetic", '{"expression": "3*3"}')),
                      reply(content="done"))
    ToolRunner(TOOLS, client=fake).run("q")
    sent = {m["tool_call_id"]: m["content"] for m in fake.requests[1]
            if isinstance(m, dict) and m.get("role") == "tool"}
    assert sent == {"a": "4.0", "b": "9.0"}, sent


def test_round_cap_stops_a_model_that_never_stops():
    bad = call("x", "arithmetic", '{"expression": "import os"}')
    fake = FakeClient(reply(bad))              # repeated for ever
    answer, trace = ToolRunner(TOOLS, client=fake, max_rounds=6).run("q")
    assert len(trace) == 6 and "step budget" in answer


def test_unknown_tool_is_reported_to_the_model():
    fake = FakeClient(reply(call("x", "delete_everything", "{}")),
                      reply(content="ok"))
    _, trace = ToolRunner(TOOLS, client=fake).run("q")
    assert trace[0].failed and "Available" in trace[0].result


def test_arguments_that_are_not_json_do_not_crash_the_loop():
    fake = FakeClient(reply(call("x", "arithmetic", '{"expression": "2+')),
                      reply(content="ok"))
    _, trace = ToolRunner(TOOLS, client=fake).run("q")
    assert trace[0].failed and "not a JSON object" in trace[0].result


def test_a_hung_tool_returns_on_time():
    slow = Tool("slow", "Never returns in time.",
                {"type": "object", "properties": {}},
                lambda: time.sleep(2), timeout=0.2)
    fake = FakeClient(reply(call("x", "slow", "{}")), reply(content="ok"))
    started = time.perf_counter()
    _, trace = ToolRunner([slow], client=fake).run("q")
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0 and "abandoned" in trace[0].result


tests = [(n, fn) for n, fn in globals().items() if n.startswith("test_")]
failures = 0
for name, fn in tests:
    label = name[5:].replace("_", " ")
    try:
        fn()
        print(f"  pass  {label}")
    except AssertionError as error:
        failures += 1
        print(f"  FAIL  {label}: {error}")
print(f"\n{len(tests) - failures} of {len(tests)} passed, "
      "with no model, no key and no network")
