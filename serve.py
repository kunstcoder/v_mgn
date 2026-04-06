"""간단한 정적 웹 서버.

- /              -> index.html
- /mgn-easy      -> mgn_easy.html
- /mgn-easy/     -> mgn_easy.html
- 그 외 정적 파일 -> 현재 디렉터리 기준 제공
"""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "0.0.0.0"
PORT = 8000
ROOT = Path(__file__).resolve().parent


class DemoRequestHandler(SimpleHTTPRequestHandler):
    def _normalize_demo_path(self) -> None:
        if self.path in {"/", "/index.html"}:
            self.path = "/index.html"
        elif self.path in {"/mgn-easy", "/mgn-easy/"}:
            self.path = "/mgn_easy.html"

    def do_GET(self) -> None:  # noqa: N802 (표준 라이브러리 시그니처 준수)
        self._normalize_demo_path()
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802 (표준 라이브러리 시그니처 준수)
        self._normalize_demo_path()
        super().do_HEAD()


def run() -> None:
    handler = partial(DemoRequestHandler, directory=str(ROOT))
    server = ThreadingHTTPServer((HOST, PORT), handler)
    print(f"Serving static demo on http://localhost:{PORT}")
    print("- easy MGN page: /mgn-easy")
    print("- existing demo : /")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
