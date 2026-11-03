"""Lossless per-control-step archive: all worlds, bounded memory, atomic chunks."""
from pathlib import Path
import io
import json
import os
import shutil
import time
import numpy as np
import zstandard as zstd


class Recorder:
    def __init__(self, directory, nenv, dt, stage, min_free_gb=3, kind="actual_training"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.nenv, self.dt, self.stage = nenv, dt, stage
        self.min_free = int(min_free_gb*1024**3)
        self.kind = kind
        self.compressor = zstd.ZstdCompressor(level=2)
        self.manifest = self.directory / "manifest.jsonl"
        self.index = len(self.manifest.read_text().splitlines()) if self.manifest.exists() else 0

    def append(self, records, start_step, update):
        if shutil.disk_usage(self.directory).free < self.min_free:
            raise OSError("Recording disk reserve reached; stopping without dropping history")
        arrays = {k: np.asarray(v) for k,v in records.items()}
        length = arrays["qpos"].shape[0]
        assert arrays["qpos"].shape[1] == self.nenv
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        encoded = self.compressor.compress(buffer.getbuffer())
        name = f"chunk_{self.index:07d}.npz.zst"
        tmp = self.directory / (name+".tmp")
        with tmp.open("wb") as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        tmp.rename(self.directory / name)
        entry = dict(file=name, stage=self.stage, update=update,
            start_control_step=int(start_step), end_control_step=int(start_step+length*self.nenv),
            frames_per_world=length, worlds=self.nenv, dt=self.dt,
            bytes=len(encoded), written_at=time.time(), kind=self.kind)
        with self.manifest.open("a") as f:
            f.write(json.dumps(entry)+"\n")
            f.flush()
            os.fsync(f.fileno())
        self.index += 1
        return entry


def load_chunk(path):
    content = zstd.ZstdDecompressor().decompress(Path(path).read_bytes())
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
