"""
Clarity v1.0 — the command line.

Four verbs, and each one is a thing somebody actually needs to do:

    ask        answer one question, with its sources and its cost
    serve      run the API
    eval       run the suite against the golden set
    demo       the eight-minute tour, scripted

The CLI exists because the fastest way to check that a system still works should not
involve a browser. It is also, in practice, how you will use your own system most days.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.v1_0.clarity import Clarity                        # noqa: E402


def cmd_ask(args) -> int:
    engine = Clarity(cache=not args.no_cache, role=args.role)
    answer = engine.ask(args.question)
    print(answer.text)
    print()
    print(f"  tools    {', '.join(answer.tools) or 'none'}")
    print(f"  sources  {', '.join(answer.sources) or 'none'}")
    print(f"  cost     {answer.tokens:,} tokens in {answer.seconds:.1f}s"
          f"{'  (cached)' if answer.cached else ''}")
    if answer.trace_id:
        print(f"  trace    {answer.trace_id}")
    # A refusal is a successful run. Exiting non-zero on one would make every CI job
    # that asks an unanswerable question fail, which is the opposite of the point.
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    uvicorn.run("clarity.v1_0.service:app", host=args.host, port=args.port)
    return 0


def cmd_eval(args) -> int:
    from clarity.evals import suite
    engine = Clarity(cache=False)

    def answer_fn(question: str):
        result = engine.ask(question)
        return result.text, result.tools, ""

    layers = suite.run(answer_fn, judge_sample=args.judge)
    print(suite.report(layers))
    return 0 if suite.green(layers) else 1


def cmd_demo(args) -> int:
    from clarity.v1_0 import demo
    return demo.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clarity")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--role", default="analyst")
    ask.add_argument("--no-cache", action="store_true")
    ask.set_defaults(fn=cmd_ask)

    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(fn=cmd_serve)

    ev = sub.add_parser("eval", help="run the suite")
    ev.add_argument("--judge", type=int, default=12)
    ev.set_defaults(fn=cmd_eval)

    dem = sub.add_parser("demo", help="the eight-minute tour")
    dem.set_defaults(fn=cmd_demo)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
