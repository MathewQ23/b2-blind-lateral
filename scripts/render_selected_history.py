#!/usr/bin/env python3
"""Replay every saved 50 Hz frame of a selected world, including failures."""
import os
os.environ.setdefault("MUJOCO_GL","egl")
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from stairs.recording import load_chunk
from stairs.selected_recording import archive_end

RECORD_HZ=50
# Azimuth is relative to the direction of travel: 0 looks over the robot's back, 180 faces it.
CAMERAS={
    "legacy":dict(azimuth=70,elevation=-23,distance=None,lead=.35,height=0.),
    "chase":dict(azimuth=25,elevation=-30,distance=3.2,lead=.3,height=.25),
    "front":dict(azimuth=180,elevation=-18,distance=3.,lead=.1,height=.1),
    "three_quarter":dict(azimuth=145,elevation=-20,distance=2.8,lead=.1,height=.1),
    "scenic_ascent":dict(azimuth=110,elevation=-18,distance=4.6,lead=.3,height=.18,fixed=True),
    "scenic_descent":dict(azimuth=70,elevation=-18,distance=4.6,lead=-.3,height=.18,fixed=True),
    "scenic_traverse":dict(azimuth=90,elevation=-18,distance=4.8,lead=.3,height=.18,fixed=True),
    "side":dict(azimuth=90,elevation=-12,distance=3.2,lead=.2,height=.05),
}


def collision_signature(model):
    return {model.geom(i).name:(np.round(model.geom_pos[i],6).tolist(),np.round(model.geom_size[i],6).tolist())
            for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]}


def forward(azimuth,elevation):
    az,el=np.radians(azimuth),np.radians(elevation)
    return np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])


def camera_view(preset, progress, direction, motion="static"):
    """World-space camera on the clear south side; gently arc and push in like a walking operator."""
    view=dict(preset)
    if motion=="news_follow":
        progress=float(np.clip(progress,0,1));ease=progress*progress*(3-2*progress)
        view.update(azimuth=(110+25*ease if direction>0 else 70-25*ease),
                    elevation=-16-6*ease,distance=4.4-.5*ease,lead=.16,height=.16,fixed=True)
    return view


class Framing:
    """Keep cameras above terrain; legacy decorative hiding is explicitly opt-in."""
    def __init__(self,model,hide_props=False):
        world=[i for i in range(model.ngeom) if model.geom_bodyid[i]==0]
        solid=[i for i in world if model.geom_contype[i] or model.geom_conaffinity[i]]
        self.boxes=np.array([[model.geom_pos[i,0]-model.geom_size[i,0],model.geom_pos[i,0]+model.geom_size[i,0],
            model.geom_pos[i,1]-model.geom_size[i,1],model.geom_pos[i,1]+model.geom_size[i,1],model.geom_pos[i,2]+model.geom_size[i,2]]
            for i in solid if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX]).reshape(-1,5)
        planes=[model.geom_pos[i,2] for i in solid if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE]
        self.floor=max(planes) if planes else -np.inf
        b=self.boxes
        def skin(i):
            # Textured overlays hugging a step or platform belong to the terrain's appearance; keep them.
            x,y,z=model.geom_pos[i]
            return model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH and bool(((b[:,0]-.05<=x)&(x<=b[:,1]+.05)&(b[:,2]-.05<=y)&(y<=b[:,3]+.05)&(z<=b[:,4]+.06)).any())
        def in_path(i):
            x,y,_=model.geom_pos[i]
            return bool(((b[:,0]<=x)&(x<=b[:,1])&(b[:,2]<=y)&(y<=b[:,3])).any())
        decor=[i for i in world if i not in solid and model.geom_rbound[i]>0 and not skin(i)]
        # Legacy hiding is opt-in. Presentation scenes must move whole props off the route.
        self.props=np.array([i for i in decor if in_path(i)],dtype=int)
        if hide_props:model.geom_group[self.props]=5
        self.scenery=np.array([i for i in decor if not in_path(i)],dtype=int)
        self.groups=model.geom_group[self.scenery].copy()

    def ground(self,x,y):
        b=self.boxes;inside=(b[:,0]<=x)&(x<=b[:,1])&(b[:,2]<=y)&(y<=b[:,3])
        return max(self.floor,b[inside,4].max()) if inside.any() else self.floor

    def visible(self,eye,target,margin=.05):
        # Steps are solid and never hidden: the sight line must pass above every tread it crosses.
        for s in np.linspace(0,1,41)[:-1]:
            p=eye+s*(target-eye)
            if p[2]<self.ground(p[0],p[1])+margin:return False
        return True

    def elevation(self,lookat,azimuth,elevation,distance,clearance=.5,targets=()):
        while elevation>-75:
            eye=lookat-distance*forward(azimuth,elevation)
            if eye[2]-self.ground(eye[0],eye[1])>=clearance and all(self.visible(eye,t) for t in targets):break
            elevation-=5
        return elevation

    def hide_occluders(self,model,data,eye,targets,margin=.25,spread=.3):
        if not len(self.scenery):return
        model.geom_group[self.scenery]=self.groups
        centers=data.geom_xpos[self.scenery];radius=model.geom_rbound[self.scenery]+margin
        hidden=np.zeros(len(self.scenery),bool)
        for target in targets:
            # A cone that widens toward the robot keeps foliage from crowding the subject as well as blocking it.
            ray=target-eye;s=np.clip((centers-eye)@ray/(ray@ray),0,1)
            hidden|=np.linalg.norm(centers-(eye+s[:,None]*ray),axis=1)<radius+spread*s*np.linalg.norm(ray)
        model.geom_group[self.scenery[hidden]]=5


def main():
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("--world",type=int,default=0)
    p.add_argument("--stride",type=int,default=1,help="playback speed multiple relative to real time")
    p.add_argument("--max-frames",type=int,default=0)
    p.add_argument("--start-sample",type=int,default=0,help="first training sample (as captioned) to include")
    p.add_argument("--end-sample",type=int,default=0,help="last training sample to include; 0 = archive end")
    p.add_argument("--episode",type=int,help="only this episode id of the selected world; combine with a sample range for speed")
    p.add_argument("--width",type=int,default=640)
    p.add_argument("--height",type=int,default=360)
    p.add_argument("--fps",type=int,default=RECORD_HZ,help="output frame rate; frames are dropped to keep real-time speed")
    p.add_argument("--crf",type=int,default=25)
    p.add_argument("--camera",choices=sorted(CAMERAS),default="legacy")
    p.add_argument("--camera-motion",choices=["static","news_follow"],default="static",help="smooth side-to-front arc and gentle push-in through a bounded recorded clip")
    p.add_argument("--fovy",type=float,help="vertical field of view in degrees")
    p.add_argument("--smooth",type=float,default=0.,help="camera follow time constant in seconds; 0 follows rigidly")
    p.add_argument("--clean",action="store_true",help="no caption overlay")
    p.add_argument("--hide-scenery",action="store_true",help="legacy per-geometry occlusion hiding; off by default")
    p.add_argument("--scene",type=Path,help="render with this MJCF; refused unless its terrain and robot match the training stage")
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.world not in range(4) or a.stride<1:p.error("world must be 0..3; stride >=1")
    if a.end_sample and a.end_sample<a.start_sample:p.error("end sample precedes start sample")
    if a.stride*RECORD_HZ%a.fps:p.error("stride*50 must be divisible by fps")
    if a.camera_motion!="static" and not a.end_sample:p.error("camera motion requires --end-sample")
    step=a.stride*RECORD_HZ//a.fps
    archive_end(a.run)
    clip=bool(a.start_sample or a.end_sample or a.episode is not None)
    suffix=(f"_ep{a.episode}" if a.episode is not None else "")+(f"_{a.start_sample}-{a.end_sample or 'end'}" if a.start_sample or a.end_sample else "")
    output=a.output or a.run/f"videos/world{a.world}_{'original' if a.stride==1 else str(a.stride)+'x'}{suffix}.mp4"
    output.parent.mkdir(exist_ok=True,parents=True)
    partial=output.with_name(output.stem+".partial.mp4")
    encoder=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
        "-s",f"{a.width}x{a.height}","-r",str(a.fps),"-i","pipe:0","-an","-c:v","libx264","-preset","veryfast",
        "-crf",str(a.crf),"-pix_fmt","yuv420p","-movflags","+faststart",str(partial)],stdin=subprocess.PIPE)
    scale=a.width/640
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",round(12*scale))
    preset=CAMERAS[a.camera];view=preset
    frames=seen=0;complete=passed=False;stages=[];tasks=set();outcome=None
    try:
        for manifest in sorted((a.run/"recordings").glob("stage_*/manifest.jsonl")):
            if passed:break
            entries=[json.loads(line) for line in manifest.read_text().splitlines()]
            if clip and not any(e["start_control_step"]+e["frames_per_world"]*e["total_training_worlds"]>=a.start_sample
                                and not (a.end_sample and e["start_control_step"]+e["total_training_worlds"]>a.end_sample) for e in entries):continue
            stage=int(manifest.parent.name.split("_")[-1])
            scene=a.run/f"scenes/stage_{stage:02d}/scene.xml"
            metadata=json.loads(scene.with_name('metadata.json').read_text())
            model=mujoco.MjModel.from_xml_path(str(scene))
            if a.scene:
                staged=model;model=mujoco.MjModel.from_xml_path(str(a.scene))
                if (model.nq,model.nv,model.nu)!=(staged.nq,staged.nv,staged.nu) or collision_signature(model)!=collision_signature(staged):
                    raise SystemExit(f"{a.scene} does not match the stage {stage} training terrain/robot; refusing to misrepresent it")
            data=mujoco.MjData(model)
            model.vis.global_.offwidth=max(model.vis.global_.offwidth,a.width)
            model.vis.global_.offheight=max(model.vis.global_.offheight,a.height)
            if a.fovy:model.vis.global_.fovy=a.fovy
            camera=mujoco.MjvCamera();camera.elevation=view["elevation"]
            camera.distance=view["distance"] or (4.1 if metadata.get('robot')=='Unitree B2' else 3.1)
            option=mujoco.MjvOption();option.geomgroup[3]=0
            framing=None if a.camera=="legacy" else Framing(model,hide_props=a.hide_scenery)
            if framing:model.vis.quality.shadowsize=max(model.vis.quality.shadowsize,8192)
            target=previous_episode=elevation=None
            with mujoco.Renderer(model,height=a.height,width=a.width) as renderer:
                stages.append(stage)
                for entry in entries:
                    worlds=entry["total_training_worlds"]
                    if entry["start_control_step"]+entry["frames_per_world"]*worlds<a.start_sample:continue
                    if a.end_sample and entry["start_control_step"]+worlds>a.end_sample:passed=True;break
                    trace=load_chunk(manifest.parent/entry["file"])
                    for t in range(entry["frames_per_world"]):
                        samples=entry["start_control_step"]+(t+1)*worlds
                        if samples<a.start_sample or (a.end_sample and samples>a.end_sample):continue
                        episode=int(trace["episode_id"][t,a.world])
                        if a.episode is not None:
                            if episode>a.episode:passed=True;break
                            if episode!=a.episode:continue
                        if trace["done"][t,a.world]:
                            outcome="success" if trace["success"][t,a.world] else ("timeout" if trace["timeout"][t,a.world] else f"failure_reason_{int(trace['reason'][t,a.world])}")
                        seen+=1
                        if (seen-1)%step:continue
                        if a.max_frames and frames>=a.max_frames:break
                        if frames%1000==0 and shutil.disk_usage(output.parent).free<2*1024**3:
                            raise OSError("Video disk reserve reached; raw trajectory retained")
                        for field in ("qpos","qvel","ctrl"):
                            getattr(data,field)[:]=trace[field][t,a.world]
                        data.time=float(trace["sim_time"][t,a.world]);mujoco.mj_forward(model,data)
                        direction=float(trace["direction"][t,a.world])
                        progress=(samples-a.start_sample)/max(1,a.end_sample-a.start_sample)
                        view=camera_view(preset,progress,direction,a.camera_motion)
                        camera.distance=view["distance"] or (4.1 if metadata.get("robot")=="Unitree B2" else 3.1)
                        camera.azimuth=view["azimuth"] if view.get("fixed") else (0 if direction>0 else 180)+view["azimuth"]
                        goal=data.qpos[:3]+np.array([view["lead"]*direction,0,view["height"]])
                        # Snap on a new episode (the robot is teleported); otherwise ease toward the body.
                        snap=target is None or episode!=previous_episode or not a.smooth
                        ease=1. if snap else 1-np.exp(-step/RECORD_HZ/a.smooth)
                        target=goal if snap else target+(goal-target)*ease
                        previous_episode=episode;camera.lookat[:]=target
                        if framing:
                            body=data.qpos[:3]
                            desired=framing.elevation(target,camera.azimuth,view["elevation"],camera.distance,
                                targets=[body+np.array([0,0,.3]),body-np.array([0,0,.2])])
                            elevation=desired if snap else min(elevation+(desired-elevation)*ease,desired)
                            camera.elevation=elevation
                            eye=target-camera.distance*forward(camera.azimuth,camera.elevation)
                            if a.hide_scenery:
                                framing.hide_occluders(model,data,eye,[target,data.qpos[:3]+np.array([0,0,.3])])
                        renderer.update_scene(data,camera=camera,scene_option=option)
                        img=Image.fromarray(renderer.render())
                        task=("ASCENT","DESCENT","TRAVERSE","STAND")[int(trace["task"][t,a.world])];tasks.add(task.lower())
                        if not a.clean:
                            caption=(f"ACTUAL TRAINING | world {a.world} | {task} | {a.stride}x | stage {stage}\n"
                                f"samples {samples:,} | episode {episode} | "
                                f"contact {''.join(str(int(v)) for v in trace['foot_contact'][t,a.world])}")
                            if 'payload_kg' in metadata:caption+=f"\nB2 | payload {metadata['payload_kg']:.1f} kg | step {metadata['height']*100:.0f} cm"
                            if trace["done"][t,a.world]:
                                caption+="\nEND: "+("SUCCESS" if trace["success"][t,a.world] else "FAILURE / TIMEOUT")
                            draw=ImageDraw.Draw(img)
                            draw.rectangle((0,0,a.width,round(58*scale)),fill="black");draw.text((round(7*scale),round(4*scale)),caption,font=font,fill="white")
                        encoder.stdin.write(np.asarray(img).tobytes());frames+=1
                    if a.max_frames and frames>=a.max_frames or passed:break
            if a.max_frames and frames>=a.max_frames:break
        complete=True
    finally:
        encoder.stdin.close();code=encoder.wait()
        report=dict(world=a.world,frames=frames,stride=a.stride,kind="actual_training_clip" if clip else "actual_training_selected_world",
            start_sample=a.start_sample,end_sample=a.end_sample or None,episode=a.episode,
            full_history=complete and not a.max_frames and not clip and a.stride==1 and a.fps==RECORD_HZ and code==0,
            complete_export=complete and code==0,encoder_exit=code,source=str(a.run.resolve()),time=time.time())
        if (a.width,a.height,a.fps,a.camera,a.clean,a.scene)!=(640,360,RECORD_HZ,"legacy",False,None):
            report.update(width=a.width,height=a.height,fps=a.fps,camera=a.camera,clean=a.clean,
                render_scene=str(a.scene.resolve()) if a.scene else None,stages=stages,tasks=sorted(tasks),last_outcome=outcome,
                hide_scenery=a.hide_scenery,camera_motion=a.camera_motion,fovy=float(model.vis.global_.fovy),smooth=a.smooth)
        output.with_suffix(".json").write_text(json.dumps(report,indent=2))
        if complete and code==0:partial.replace(output)
    if code:raise SystemExit(code)
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
