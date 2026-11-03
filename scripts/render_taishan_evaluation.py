#!/usr/bin/env python3
"""Replay one complete, explicitly labelled held-out Taishan evaluation episode."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json
from pathlib import Path
import subprocess
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    p=argparse.ArgumentParser()
    p.add_argument('evaluation',type=Path)
    p.add_argument('--task',choices=['ascent','descent','traverse','stand'],default='traverse')
    p.add_argument('--episode',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    summary=json.loads((a.evaluation/'summary.json').read_text())
    indices=[i for i,r in enumerate(summary['results']) if r['task']==a.task]
    index=indices[a.episode];row=summary['results'][index]
    with np.load(a.evaluation/'episodes.npz') as z:
        trace={k:z[k][:,index].copy() for k in ('qpos','qvel','ctrl','foot_contact')}
    frames=int(round(row['duration_s']/.02))
    m=mujoco.MjModel.from_xml_path(str(a.evaluation/'scene/scene.xml'));d=mujoco.MjData(m)
    camera=mujoco.MjvCamera();camera.distance=3.;camera.elevation=-23
    direction=np.sign(row['goal_x']-trace['qpos'][0,0]) or 1
    camera.azimuth=70 if direction>0 else 250
    option=mujoco.MjvOption();option.geomgroup[3]=0
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','640x360','-r','50','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','24',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)],stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(m,height=360,width=640) as renderer:
            for t in range(frames):
                for key in ('qpos','qvel','ctrl'):getattr(d,key)[:]=trace[key][t]
                d.time=(t+1)*.02;mujoco.mj_forward(m,d)
                camera.lookat[:]=d.qpos[:3]+np.array([direction*.35,0,0])
                renderer.update_scene(d,camera=camera,scene_option=option)
                im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
                caption=(f"HELD-OUT EVALUATION | {a.task.upper()} | {summary['height']*100:.0f}cm stairs | 1x\n"
                    f"Single learned policy | seed {row['seed']} | t={d.time:.2f}s | "
                    f"feet {''.join(str(int(v)) for v in trace['foot_contact'][t])}\n"
                    f"Episode accepted: {row['accepted']} | task rate: {summary['accepted_rates'][a.task]:.0%} / {summary['episodes_per_task']} trials")
                draw.rectangle((0,0,640,57),fill='black');draw.text((6,3),caption,font=font,fill='white')
                encoder.stdin.write(np.asarray(im).tobytes())
    finally:
        encoder.stdin.close();code=encoder.wait()
    if code:raise SystemExit(code)
    a.output.with_suffix('.json').write_text(json.dumps(dict(kind='held_out_evaluation',source=str(a.evaluation.resolve()),
        task=a.task,episode=a.episode,frames=frames,original_speed=True,result=row),indent=2))
    print(a.output)


if __name__=='__main__':main()
