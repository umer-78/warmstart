"""python -m warmstart bench              run every measurement, print the tables, write results/bench.json
python -m warmstart gate               rerun and fail if the cache got less precise, leakier or less useful
python -m warmstart serve --upstream URL   the caching proxy in front of an OpenAI-compatible API
python -m warmstart demo               rebuild the demo page's data in docs/"""
import argparse
import json
import sys

from . import bench

BASELINE = bench.RESULTS / "baseline.json"


def gate(now, base):
    problems = []
    b, n = base["embedders"][base["best"]], now["embedders"][now["best"]]
    if n["held_out"]["wrong_rate"] > 0.01:
        problems.append(f"held-out wrong hits {n['held_out']['wrong_rate']:.2%}, above the 1% ship gate")
    if n["replay"]["wrong_rate"] > 0.01:
        problems.append(f"replay wrong semantic hits {n['replay']['wrong_rate']:.2%}, above the 1% ship gate")
    if n["replay"]["leaks"] or n["replay"]["stale"]:
        problems.append(f"{n['replay']['leaks']} leaks and {n['replay']['stale']} stale answers served")
    if now["prompt_change"]["served_from_before"]:
        problems.append("a prompt change did not empty the cache")
    if n["replay"]["hit_rate"] < b["replay"]["hit_rate"] - 0.02:
        problems.append(f"hit rate {n['replay']['hit_rate']:.1%}, more than 2 points under the baseline {b['replay']['hit_rate']:.1%}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(prog="warmstart")
    ap.add_argument("command", choices=["bench", "gate", "serve", "demo"])
    ap.add_argument("--upstream", help="serve: the OpenAI-compatible base URL to forward misses to")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args(argv)
    if args.command == "serve":
        import uvicorn
        from .proxy import create_app
        if not args.upstream:
            ap.error("serve needs --upstream")
        uvicorn.run(create_app(args.upstream), host="0.0.0.0", port=args.port)
        return 0
    if args.command == "demo":
        from . import demo
        demo.build()
        return 0
    # the gate checks the embedder the cache ships with; bench also measures the alternatives
    result = bench.run(embedders=("bge-small",) if args.command == "gate" else bench.EMBEDDERS)
    print(bench.report(result))
    print(f"\n({result['seconds']} s)")
    if args.command == "bench":
        bench.save(result)
    if args.command == "gate":
        problems = gate(result, json.loads(BASELINE.read_text()))
        for p in problems:
            print(f"GATE: {p}", file=sys.stderr)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
