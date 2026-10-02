"""Download the public ARC-AGI-1 tasks (Apache-2.0, github.com/fchollet/ARC-AGI) into data/arc/."""
import io
import urllib.request
import zipfile
from pathlib import Path

DEST = Path(__file__).parent / "data" / "arc"
URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"

if __name__ == "__main__":
    if (DEST / "training").exists():
        print("already present:", DEST)
    else:
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))
        for n in z.namelist():
            if "/data/" in n and n.endswith(".json"):
                rel = n.split("/data/", 1)[1]
                (DEST / rel).parent.mkdir(parents=True, exist_ok=True)
                (DEST / rel).write_bytes(z.read(n))
        print("downloaded to", DEST)
