"""Reward-only terrain-aligned posture shaping for the unchanged B2 dynamics."""
import json
import jax.numpy as jp
import mujoco
import numpy as np
from stairs.b2_env import B2SwingEnv


def posture_costs(rotation,desired_pitch,clearances):
    pitch_up=jp.arctan2(rotation[2,0],jp.sqrt(rotation[0,0]**2+rotation[1,0]**2)+1e-6)
    roll=jp.arctan2(rotation[2,1],rotation[2,2])
    orientation=-6.*roll**2-2.*(pitch_up-desired_pitch)**2
    collision_warning=-1.5*jp.clip((.08-jp.min(clearances))/.08,0.,2.)**2
    return orientation,collision_warning


class B2PostureEnv(B2SwingEnv):
    def __init__(self,directory,stage=0,seed=20261028):
        super().__init__(directory,stage,seed)
        points=[]
        for name in ('torso_1','torso_10'):
            geom=self.mj_model.geom(name)
            if int(geom.type[0])!=int(mujoco.mjtGeom.mjGEOM_BOX):
                raise ValueError(f'Expected official B2 box geometry for {name}')
            rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,geom.quat)
            for sx in (-1.,1.):
                for sy in (-1.,1.):
                    local=np.array([sx*geom.size[0],sy*geom.size[1],-geom.size[2]])
                    points.append(rotation.reshape(3,3)@local+geom.pos)
        self._body_bottom_points=jp.asarray(np.array(points))
        self.reward_keys+=['body_clearance']
        self.scene_metadata.update(task_version='b2_payload_v3',reward_parts=self.reward_keys,
            reward_revision='Stronger roll stability, terrain-aligned pitch and pre-contact belly/head clearance warning; acceptance and physics unchanged',
            body_clearance_margin_m=.08,terrain_pitch_fraction=.6,maximum_target_pitch_rad=.35)
        (self.directory/'metadata.json').write_text(json.dumps(self.scene_metadata,indent=2)+'\n')

    def step(self,state,action):
        following=super().step(state,action)
        data=following.data;info=dict(following.info);rewards=dict(following.metrics)
        x,y=data.qpos[:2];direction=info['direction']
        fore=self.terrain(x+.35*direction,y);rear=self.terrain(x-.35*direction,y)
        desired_pitch=jp.clip(.6*jp.arctan2(fore-rear,.7),-.35,.35)
        rotation=data.xmat[self._base_id]
        world=self._body_bottom_points@rotation.T+data.xpos[self._base_id]
        clearances=world[:,2]-self.terrain(world[:,0],world[:,1])
        orientation,warning=posture_costs(rotation,desired_pitch,clearances)
        reward=following.reward+(orientation-rewards['orientation']+warning)*self.dt
        rewards['orientation']=orientation;rewards['body_clearance']=warning
        valid=jp.all(jp.isfinite(data.qpos))&jp.all(jp.isfinite(data.qvel))
        reward=jp.where(valid,reward,-1.)
        info['episode_return']=state.info['episode_return']+reward
        return following.replace(info=info,metrics=rewards,reward=reward)
