"""Minimal loopback callback for SmartAPI app registration; it never reads tokens."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PAGE = b"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Northstar callback</title></head>
<body><main><h1>Northstar callback is ready</h1>
<p>This endpoint does not capture or store authentication tokens. Return to the local app and use its daily-login form.</p>
</main></body></html>"""


class CallbackHandler(BaseHTTPRequestHandler):
    server_version = "NorthstarCallback"
    sys_version = ""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(PAGE)

    def do_POST(self) -> None:
        self.send_error(405)

    def log_message(self, _format: str, *_args) -> None:
        return


def make_server(host: str = "127.0.0.1", port: int = 8011) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "::1"}:
        raise ValueError("The callback stub must bind to loopback only")
    return ThreadingHTTPServer((host, port), CallbackHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    server = make_server(port=args.port)
    print(f"Token-discarding callback listening on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()