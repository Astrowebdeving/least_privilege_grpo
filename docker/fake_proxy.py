from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer


RESPONSES = {
    "/status": "status=green\n",
    "/count": "count=12\n",
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = RESPONSES.get(self.path, "not_found\n")
        code = 200 if self.path in RESPONSES else 404
        print(f"GET {self.path}", flush=True)
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
