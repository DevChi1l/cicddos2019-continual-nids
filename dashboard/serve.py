"""Serve the dashboard folder on http://localhost:8000 and open it in the browser (standard library only).

Use this if your browser blocks the page's background worker when index.html is opened from file://.

  python dashboard/serve.py               # http://localhost:8000/index.html
  python dashboard/serve.py --port 8080   # another port
  python dashboard/serve.py --no-browser  # serve only

Serves this folder only, on the loopback interface; stop with Ctrl+C.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import threading
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    a = ap.parse_args()
    if not (HERE / "index.html").exists():
        raise SystemExit("dashboard/index.html not found; build it with python dashboard/build_dashboard.py")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
    with http.server.ThreadingHTTPServer(("localhost", a.port), handler) as srv:
        url = f"http://localhost:{a.port}/index.html"
        print(f"serving {HERE} at {url}  (Ctrl+C to stop)", flush=True)
        if not a.no_browser:
            threading.Timer(0.5, webbrowser.open, (url,)).start()
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
