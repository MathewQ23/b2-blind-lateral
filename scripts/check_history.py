#!/usr/bin/env python3
"""Check that every recorded PPO collection interval is present exactly once."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stairs.recording import load_chunk

p=argparse.ArgumentParser()
p.add_argument("run",type=Path)
p.add_argument("--read-all",action="store_true")
a=p.parse_args()
end=0
chunks=0
for manifest in sorted((a.run/"recordings").glob("stage_*/manifest.jsonl")):
    for line in manifest.read_text().splitlines():
        entry=json.loads(line)
        assert entry["start_control_step"]==end,(end,entry)
        path=manifest.parent/entry["file"]
        assert path.stat().st_size==entry["bytes"]
        if a.read_all:
            content=load_chunk(path)
            assert content["qpos"].shape[:2]==(entry["frames_per_world"],entry["worlds"])
            assert content["done"].shape==content["qpos"].shape[:2]
        end=entry["end_control_step"]
        chunks+=1
print(json.dumps(dict(continuous=True,chunks=chunks,recorded_control_samples=end)))
