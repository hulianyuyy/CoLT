import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

ROOT = Path( "./LLaMA-Factory/data")    # root dir
MAX_WORKERS = 8     # threads

def unzip_one(zip_path: Path):
    out_dir = zip_path.parent
    base_name = zip_path.stem

    if (out_dir / base_name).exists():
        print(f"Skip: {zip_path}")
        return

    print(f"Unzipping: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(out_dir)
    except Exception as e:
        print(f"Error: {zip_path} -> {e}")

def main():
    zips = list(ROOT.rglob("*.zip"))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        pool.map(unzip_one, zips)

if __name__ == "__main__":
    main()
