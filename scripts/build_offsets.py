"""
Build byte offset index for 15M metadata.jsonl on Modal Volume.
Enables sub-millisecond random access lookup with zero startup RAM overhead.
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy>=1.26")
volume = modal.Volume.from_name("voice-rag-index", create_if_missing=True)
app = modal.App("voice-rag-build-offsets", image=image)


@app.cls(volumes={"/index": volume}, timeout=1200)
class OffsetBuilder:

    @modal.method()
    def run(self):
        import os
        import time
        import numpy as np

        meta_path = "/index/metadata.jsonl"
        offset_path = "/index/metadata_offsets.npy"

        print(f"Reading line offsets from {meta_path}...")
        t0 = time.perf_counter()
        offsets = []

        with open(meta_path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                offsets.append(pos)
                if len(offsets) % 1_000_000 == 0:
                    print(f"  Indexed {len(offsets):,} lines ({time.perf_counter() - t0:.1f}s)...")

        arr = np.array(offsets, dtype=np.int64)
        print(f"Saving {len(arr):,} offsets to {offset_path}...")
        np.save(offset_path, arr)
        volume.commit()

        duration = time.perf_counter() - t0
        print(f"✅ Offset index built in {duration:.1f}s ({os.path.getsize(offset_path)/(1024*1024):.1f} MB)")


@app.local_entrypoint()
def main():
    builder = OffsetBuilder()
    builder.run.remote()
