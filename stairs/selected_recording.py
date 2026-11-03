"""Full 50 Hz fields for fixed worlds, plus bounded extra abnormal episodes."""
from collections import defaultdict
import io
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import zstandard as zstd


def write_archive(path, arrays):
    buffer=io.BytesIO()
    np.savez(buffer,**arrays)
    encoded=zstd.ZstdCompressor(level=3).compress(buffer.getbuffer())
    tmp=Path(str(path)+".tmp")
    with tmp.open("wb") as stream:
        stream.write(encoded);stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)
    return len(encoded)


class SelectedRecorder:
    def __init__(self, root, stage, num_envs, worlds=4, extra_worlds=8):
        self.root=Path(root)
        self.directory=self.root/f"recordings/stage_{stage:02d}"
        self.directory.mkdir(parents=True,exist_ok=True)
        self.manifest=self.directory/"manifest.jsonl"
        self.stage,self.num_envs,self.worlds=stage,num_envs,worlds
        self.extra_worlds=extra_worlds
        entries=[json.loads(s) for s in self.manifest.read_text().splitlines()] if self.manifest.exists() else []
        self.index=len(entries)
        self.expected=entries[-1]["end_control_step"] if entries else None
        self.buffers=[[] for _ in range(extra_worlds)]
        self.complete=[False]*extra_worlds
        self.counts=defaultdict(int)
        self.anomalies=self.root/"anomalies"
        self.anomalies.mkdir(exist_ok=True)
        if (self.anomalies/"manifest.jsonl").exists():
            for line in (self.anomalies/"manifest.jsonl").read_text().splitlines():
                e=json.loads(line)
                if e["stage"]==stage:self.counts[e["reason"]]+=1

    def append(self, records, start_step, update):
        if self.expected is not None and start_step!=self.expected:
            raise ValueError(f"Checkpoint/archive disagreement: {start_step} vs {self.expected}; refusing to overwrite history")
        if shutil.disk_usage(self.root).free<5*1024**3:
            raise OSError("5 GiB disk reserve reached")
        horizon=records["qpos"].shape[0]
        selected={k:v[:,:self.worlds] for k,v in records.items()}
        filename=f"chunk_{self.index:07d}.npz.zst"
        size=write_archive(self.directory/filename,selected)
        entry=dict(file=filename,stage=self.stage,update=update,kind="actual_training_selected_worlds",
            start_control_step=start_step,end_control_step=start_step+horizon*self.num_envs,
            frames_per_world=horizon,worlds=self.worlds,world_ids=list(range(self.worlds)),
            total_training_worlds=self.num_envs,dt=.02,bytes=size,written_at=time.time(),fields=list(selected))
        with self.manifest.open("a") as stream:
            stream.write(json.dumps(entry)+"\n");stream.flush();os.fsync(stream.fileno())
        self.expected=entry["end_control_step"];self.index+=1
        self._extra_episodes(records,start_step)
        return entry

    def _extra_episodes(self,records,start_step):
        for i in range(self.extra_worlds):
            world=self.worlds+i
            for t in range(len(records["qpos"])):
                row={k:v[t,world].copy() for k,v in records.items()}
                if int(row["episode_step"])==1:
                    self.buffers[i]=[];self.complete[i]=True
                self.buffers[i].append(row)
                # Episodes are at most 4500 steps. Partial buffers never claim completeness.
                if len(self.buffers[i])>4500:
                    self.buffers[i]=[];self.complete[i]=False
                if not bool(row["done"]):continue
                reason=int(row["reason"])
                if not reason and int(row["max_flight"])>=10:reason=5
                counts=row["contact_count"] / max(int(row["climb_frames"]),1)
                if not reason and int(row["climb_frames"])>100 and counts.min()<.12:reason=6
                if not reason and bool(row["timeout"]):reason=7
                # 12 complete examples per reason and curriculum stage, from extra worlds 4..11.
                if reason and self.complete[i] and self.counts[reason]<12:
                    name=f"stage{self.stage:02d}_reason{reason}_case{self.counts[reason]:03d}.npz.zst"
                    trace={k:np.stack([r[k] for r in self.buffers[i]]) for k in records}
                    size=write_archive(self.anomalies/name,trace)
                    event=dict(file=name,stage=self.stage,world=world,reason=reason,complete_episode=True,
                        episode_id=int(row["episode_id"]),frames=len(self.buffers[i]),bytes=size,
                        end_training_sample=start_step+(t+1)*self.num_envs)
                    with (self.anomalies/"manifest.jsonl").open("a") as f:
                        f.write(json.dumps(event)+"\n");f.flush();os.fsync(f.fileno())
                    self.counts[reason]+=1
                self.buffers[i]=[];self.complete[i]=False

    def state_dict(self):
        return dict(buffers=self.buffers,complete=self.complete)

    def restore(self,state):
        self.buffers,self.complete=state["buffers"],state["complete"]


def archive_end(root):
    end=0
    for manifest in sorted((Path(root)/"recordings").glob("stage_*/manifest.jsonl")):
        for line in manifest.read_text().splitlines():
            entry=json.loads(line)
            if entry["start_control_step"]!=end:
                raise ValueError("Noncontinuous recording manifest")
            if (manifest.parent/entry["file"]).stat().st_size!=entry["bytes"]:
                raise ValueError("Recording size mismatch")
            end=entry["end_control_step"]
    return end
