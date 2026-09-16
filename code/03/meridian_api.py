# skip
"""
A small HTTP API over the Meridian warehouse, for Chapter 3.

Every listing in this chapter makes a real HTTP request over a real socket and gets a real
status code back. None of them touches the internet, and none of them needs a key. That
matters for three reasons: the examples still work in ten years, they work on a plane, and
you can make the server fail on demand — which is the only way to learn what to do when it
does.

Start it inside a listing:

    from meridian_api import serve
    BASE = serve()          # returns e.g. http://127.0.0.1:53219

Endpoints, chosen to be the ones you will meet in the wild:

    GET /health                       200
    GET /quarters                     200  every quarter with data
    GET /revenue?year=&quarter=       200 / 400 (bad params) / 404 (no such quarter)
    GET /account                      401 unless a bearer token is sent
    GET /flaky?key=&fail=             503 the first `fail` times for that key, then 200
    GET /ratelimited                  429, with a Retry-After header
    GET /slow?seconds=                200, eventually
    GET /boom                         500
    POST /orders                      201, and honours an Idempotency-Key header

And the ones the longer sections of the chapter need:

    GET /feed?offset=&limit=          newest-first, paged by offset
    GET /feed?cursor=&limit=          newest-first, paged by cursor
    POST /feed                        publish new items at the top of the feed
    GET /limited                      200, or 429 once a token bucket is empty
    GET /trickle?chunks=&interval=    200, one byte at a time, slowly
    GET /events?count=&interval=      a stream of server-sent events
    GET /revenue-v2?year=&quarter=    the same data, in a shape that quietly changed
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DB = "data/meridian/warehouse/meridian.db"
VALID_TOKEN = "meridian-demo-key"

_attempts: dict[str, int] = defaultdict(int)
_orders: list[dict] = []
_idempotency: dict[str, dict] = {}
_lock = threading.Lock()

# The feed: item 1 is the oldest. Newest-first is how most feeds are served, and it is
# what makes offset pagination misbehave when something new arrives.
_feed: list[dict] = [{"id": i, "title": f"ticket {i}"} for i in range(1, 101)]

# A token bucket: CAPACITY requests at once, refilled at RATE per second.
BUCKET_CAPACITY, BUCKET_RATE = 5, 5.0
_bucket = {"tokens": float(BUCKET_CAPACITY), "at": time.monotonic()}


def _query(sql: str, args: tuple = ()) -> list[dict]:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, args).fetchall()]
    finally:
        con.close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass                                    # keep listing output clean

    def _send(self, status: int, payload: dict, extra_headers: dict | None = None):
        body = json.dumps(payload, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Meridian-Api", "demo")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        route = url.path

        if route == "/health":
            return self._send(200, {"status": "ok", "service": "meridian-api"})

        if route == "/quarters":
            rows = _query("SELECT DISTINCT year, quarter FROM v_sales "
                          "ORDER BY year, quarter")
            return self._send(200, {"quarters": [f"{r['year']}-Q{r['quarter']}"
                                                 for r in rows]})

        if route == "/revenue":
            if "year" not in params or "quarter" not in params:
                return self._send(400, {
                    "error": "missing_parameter",
                    "message": "Both 'year' and 'quarter' are required.",
                    "example": "/revenue?year=2024&quarter=3"})
            try:
                year, quarter = int(params["year"]), int(params["quarter"])
            except ValueError:
                return self._send(400, {
                    "error": "invalid_parameter",
                    "message": "'year' and 'quarter' must be whole numbers."})

            rows = _query("""SELECT region, ROUND(SUM(revenue), 2) AS revenue,
                                    COUNT(DISTINCT order_id) AS orders
                             FROM v_sales WHERE year = ? AND quarter = ?
                             GROUP BY region ORDER BY revenue DESC""", (year, quarter))
            if not rows:
                return self._send(404, {
                    "error": "not_found",
                    "message": f"No data for {year} Q{quarter}.",
                    "hint": "Try /quarters to see what exists."})
            return self._send(200, {"year": year, "quarter": quarter, "regions": rows,
                                    "total_revenue": round(sum(r["revenue"] for r in rows), 2)})

        if route == "/account":
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {VALID_TOKEN}":
                return self._send(401, {
                    "error": "unauthorized",
                    "message": "Send an Authorization header: 'Bearer <your key>'."})
            return self._send(200, {"plan": "demo", "requests_remaining": 4998})

        if route == "/flaky":
            key = params.get("key", "default")
            fail_times = int(params.get("fail", 2))
            with _lock:
                _attempts[key] += 1
                n = _attempts[key]
            if n <= fail_times:
                return self._send(503, {
                    "error": "service_unavailable",
                    "message": f"Temporarily unavailable (attempt {n}).",
                    "attempt": n})
            return self._send(200, {"ok": True, "succeeded_on_attempt": n})

        if route == "/ratelimited":
            return self._send(429, {
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Slow down."},
                {"Retry-After": "2", "X-RateLimit-Limit": "60",
                 "X-RateLimit-Remaining": "0"})

        if route == "/slow":
            time.sleep(float(params.get("seconds", 3)))
            return self._send(200, {"ok": True, "slept": params.get("seconds", "3")})

        if route == "/boom":
            return self._send(500, {
                "error": "internal_error",
                "message": "Something broke on our side. Not your fault."})

        if route == "/orders":
            return self._send(200, {"count": len(_orders), "orders": _orders})

        if route == "/feed":
            limit = int(params.get("limit", 10))
            with _lock:
                newest_first = sorted(_feed, key=lambda item: -item["id"])
            if "cursor" in params:
                # The cursor is the id of the last item the client saw. "Items older than
                # this one" means the same thing however many new items have arrived.
                cursor = int(params["cursor"])
                page = [item for item in newest_first if item["id"] < cursor][:limit]
            else:
                offset = int(params.get("offset", 0))
                page = newest_first[offset:offset + limit]
            next_cursor = page[-1]["id"] if len(page) == limit else None
            return self._send(200, {"items": page, "next_cursor": next_cursor})

        if route == "/limited":
            with _lock:
                now = time.monotonic()
                _bucket["tokens"] = min(BUCKET_CAPACITY,
                                        _bucket["tokens"] + (now - _bucket["at"]) * BUCKET_RATE)
                _bucket["at"] = now
                if _bucket["tokens"] < 1:
                    wait = (1 - _bucket["tokens"]) / BUCKET_RATE
                    return self._send(429, {"error": "rate_limit_exceeded"},
                                      {"Retry-After": f"{wait:.2f}"})
                _bucket["tokens"] -= 1
            return self._send(200, {"ok": True})

        if route == "/trickle":
            chunks = int(params.get("chunks", 8))
            interval = float(params.get("interval", 0.5))
            body = b"x" * chunks
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            for i in range(chunks):
                time.sleep(interval)
                self.wfile.write(body[i:i + 1])
                self.wfile.flush()
            return

        if route == "/events":
            count = int(params.get("count", 5))
            interval = float(params.get("interval", 0.2))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")    # length unknown in advance
            self.end_headers()

            def chunk(data: bytes):
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            words = "Midwest revenue fell after the largest account did not renew".split()
            for i in range(count):
                time.sleep(interval)
                event = {"index": i, "delta": words[i % len(words)] + " "}
                chunk(f"data: {json.dumps(event)}\n\n".encode())
            chunk(b"event: done\ndata: {}\n\n")
            self.wfile.write(b"0\r\n\r\n")                    # the end of the body
            return

        if route == "/revenue-v2":
            year, quarter = int(params.get("year", 0)), int(params.get("quarter", 0))
            rows = _query("""SELECT region, ROUND(SUM(revenue), 2) AS revenue
                             FROM v_sales WHERE year = ? AND quarter = ?
                             GROUP BY region ORDER BY revenue DESC""", (year, quarter))
            # A provider "tidied up" its response: renamed a field, and started sending
            # money as strings so that no precision is lost. Nothing returns an error.
            return self._send(200, {"year": year, "quarter": quarter,
                                    "regions": [{"region_name": r["region"],
                                                 "revenue": f"{r['revenue']:.2f}"}
                                                for r in rows]})

        return self._send(404, {"error": "not_found",
                                "message": f"No route {route}."})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/feed":
            length = int(self.headers.get("Content-Length", 0))
            count = int(json.loads(self.rfile.read(length) or b"{}").get("count", 1))
            with _lock:
                top = max(item["id"] for item in _feed)
                new = [{"id": top + i, "title": f"ticket {top + i}"}
                       for i in range(1, count + 1)]
                _feed.extend(new)
            return self._send(201, {"published": [item["id"] for item in new]})
        if url.path != "/orders":
            return self._send(404, {"error": "not_found",
                                    "message": f"No route {url.path}."})

        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        key = self.headers.get("Idempotency-Key")

        with _lock:
            # A repeat of a request we have already carried out returns the original
            # result rather than doing it a second time. That is the whole idea.
            if key and key in _idempotency:
                return self._send(200, {**_idempotency[key], "replayed": True})

            order = {"order_id": len(_orders) + 1, "sku": payload.get("sku"),
                     "qty": payload.get("qty")}
            _orders.append(order)
            if key:
                _idempotency[key] = order

        # Some of the time, the order is created and the response never arrives.
        if payload.get("simulate_lost_response"):
            self.close_connection = True
            return
        return self._send(201, order)


def serve() -> str:
    """Start the API on a free port in a background thread. Returns its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    return f"http://{host}:{port}"


def reset():
    """Forget how many times each /flaky key has been tried, and drop any orders."""
    with _lock:
        _attempts.clear()
        _orders.clear()
        _idempotency.clear()
