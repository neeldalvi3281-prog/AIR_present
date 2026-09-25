"""Download the MediaPipe Hand Landmarker .task model (one-time, setup only)."""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config


def main():
    cfg = load_config()
    path = cfg["model"]["path"]
    url = cfg["model"]["url"]
    if os.path.exists(path):
        print(f"Model already present: {path} ({os.path.getsize(path)} bytes)")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading Hand Landmarker model...\n  {url}\n  -> {path}")
    urllib.request.urlretrieve(url, path)
    print(f"Done: {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
