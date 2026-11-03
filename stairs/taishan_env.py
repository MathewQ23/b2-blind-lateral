"""One terrain-aware Go1 task for ascent, descent, traversal and stopping."""
from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base, joystick
from stairs.taishan import TaishanConfig, StairRoute, build_scene


HEIGHTS = (.02, .04, .06, .08, .10, .12, .14, .16, .18, .20, .22)
TASKS = ("ascent", "descent", "traverse", "stand")


class TaishanEnv(joystick.Joystick):
    def __init__(self, directory, stage=0, seed=20261028, height_override=None):
        self.directory = Path(directory)
        self.stage = stage
        self.height = HEIGHTS[stage] if height_override is None else height_override
        fraction = (self.height-.02)/.20
        terrain_config = TaishanConfig(rise_m=self.height,
            tread_m=.32-.08*fraction, rise_jitter_m=min(.015,self.height*.1),
            tread_jitter_m=.01+.01*fraction, seed=seed+stage, scenery=False)
        self.route = StairRoute(terrain_config)
        scene, self.scene_metadata = build_scene(directory, terrain_config)
        config = joystick.default_config()
        config.impl = "warp"
        config.njmax = 512
        config.naconmax = 32768
        config.action_scale = .65
        config.noise_config.level = .25
        config.reward_config.scales.pose = .2
        config.reward_config.scales.feet_height = -.1
        config.reward_config.scales.feet_clearance = -.5
        config.reward_config.scales.feet_air_time = .0
        config.reward_config.scales.orientation = -3.
        config.reward_config.max_foot_height = .07+self.height*.7
        base.Go1Env.__init__(self, str(scene), config)
        self._post_init()
        self.episode_length = 4500
        self._all_foot_contacts = jp.array([[int(self.mj_model.sensor_adr[self.mj_model.sensor(n).id])
            for n in names] for names in self.scene_metadata["foot_contact_sensors"].values()])
        self._trunk_contact = jp.array([int(self.mj_model.sensor_adr[self.mj_model.sensor(n).id])
            for n in self.scene_metadata["trunk_contact_sensors"]])
        self._scan_x = jp.repeat(jp.linspace(-.4,1.,8),3)
        self._scan_y = jp.tile(jp.array([-.25,0,.25]),8)
        self._starts = jp.array([self.mj_model.key(n).qpos for n in self.scene_metadata["keyframes"]])
        summit = self.route.segment("summit")
        self.summit_center = (summit["x0"]+summit["x1"])/2
        self.left_goal = float(self._starts[0,0])
        self.right_goal = float(self._starts[2,0])
        self.reward_keys = list(config.reward_config.scales) + [
            "body_height", "heading", "lateral", "long_air", "flight", "gait", "stand", "success", "failure"]
        self.scene_metadata.update(task_integration="stairs.train_taishan",stage=stage,
            reward_parts=self.reward_keys,task_names=TASKS,observation="proprioception + local height scan + contact + phase + goal",
            success="Goal distance < .22 m, speed < .12 m/s, angular speed < .3 rad/s, upright > .9, >=3 feet, hold 1 s")
        (self.directory/"metadata.json").write_text(json.dumps(self.scene_metadata,indent=2)+"\n")

    def terrain(self, x, y=0.):
        return self.route.height_at(x,y,array_module=jp)

    def _get_obs(self, data, info):
        observations = super()._get_obs(data,info)
        w,x,y,z=data.qpos[3:7]
        yaw=jp.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        sx=data.qpos[0]+jp.cos(yaw)*self._scan_x-jp.sin(yaw)*self._scan_y
        sy=data.qpos[1]+jp.sin(yaw)*self._scan_x+jp.cos(yaw)*self._scan_y
        scan=jp.clip(self.terrain(sx,sy)-data.qpos[2],-1.2,.8)
        heading=info["heading"]-yaw
        extra=jp.concatenate((scan,info["last_contact"].astype(jp.float32),
            jp.array([jp.sin(info["phase"]),jp.cos(info["phase"]),jp.sin(heading),jp.cos(heading),
                      jp.clip((info["goal_x"]-data.qpos[0])*info["direction"],-1,3)])))
        return {k:jp.concatenate((v,extra)) for k,v in observations.items()}

    def reset(self, key, task_override=-1):
        key,kt,kr,kx,ky,kv,kp=jax.random.split(key,7)
        draw=jax.random.uniform(kt)
        task=jp.where(draw<.4,0,jp.where(draw<.8,1,jp.where(draw<.95,2,3)))
        task=jp.where(task_override>=0,task_override,task).astype(jp.int32)
        reverse=jax.random.bernoulli(kr)
        descending=task==1
        index=jp.where(reverse,jp.where(descending,3,2),jp.where(descending,1,0))
        qpos=self._starts[index]
        qpos=qpos.at[0].add(jax.random.uniform(kx,(),minval=-.1,maxval=.1))
        qpos=qpos.at[1].set(jax.random.uniform(ky,(),minval=-.05,maxval=.05))
        heading=jp.where(reverse,jp.pi,0.)
        yaw=heading+jax.random.uniform(kp,(),minval=-.06,maxval=.06)
        qpos=qpos.at[3:7].set(jp.array([jp.cos(yaw/2),0.,0.,jp.sin(yaw/2)]))
        direction=jp.where(reverse,-1.,1.)
        goal=jp.where(task==0,self.summit_center,jp.where(reverse,self.left_goal,self.right_goal))
        goal=jp.where(task==3,qpos[0],goal)
        cruise=jax.random.uniform(kv,(),minval=.25,maxval=.45)
        data=mjx_env.make_data(self.mj_model,qpos=qpos,qvel=jp.zeros(self.mj_model.nv),
            ctrl=self._default_pose,impl="warp",naconmax=32768,njmax=512)
        data=mjx.forward(self.mjx_model,data)
        zero=jp.array(0.,jp.float32);counter=jp.array(0,jp.int32)
        info=dict(rng=key,command=jp.array([jp.where(task==3,0.,cruise),0.,0.]),
            last_act=jp.zeros(12),last_last_act=jp.zeros(12),feet_air_time=jp.zeros(4),
            last_contact=jp.zeros(4,dtype=bool),swing_peak=jp.zeros(4),
            steps_since_last_pert=counter,steps_until_next_pert=jp.array(100000,jp.int32),
            episode_steps=counter,episode_id=counter,episode_return=zero,task=task,
            heading=heading,direction=direction,goal_x=goal,cruise=cruise,phase=zero,
            success=jp.array(False),failure=jp.array(False),timeout=jp.array(False),
            hold=counter,flight_steps=counter,max_flight=counter,contact_count=jp.zeros(4),
            climb_frames=counter,airborne_count=counter,reason=counter)
        metrics={k:zero for k in self.reward_keys}
        return mjx_env.State(data,self._get_obs(data,info),zero,zero,metrics,info)

    def _cost_feet_clearance(self,data):
        feet=data.site_xpos[self._feet_site_id]
        clearance=feet[:,2]-self.terrain(feet[:,0],feet[:,1])
        vel=data.sensordata[self._foot_linvel_sensor_adr]
        return jp.sum(jp.abs(clearance-self._config.reward_config.max_foot_height)*jp.linalg.norm(vel[:,:2],axis=-1))

    def step(self,state,action):
        info=dict(state.info)
        targets=jp.clip(self._default_pose+action*self._config.action_scale,self._lowers,self._uppers)
        data=mjx_env.step(self.mjx_model,state.data,targets,self.n_substeps)
        contact=jp.any(data.sensordata[self._all_foot_contacts]>0,axis=-1)
        first_contact=(info["feet_air_time"]>0)&(contact|info["last_contact"])
        info["feet_air_time"]+=self.dt
        feet=data.site_xpos[self._feet_site_id]
        foot_height=feet[:,2]-self.terrain(feet[:,0],feet[:,1])
        info["swing_peak"]=jp.maximum(info["swing_peak"],foot_height)
        clearance=data.qpos[2]-jp.mean(self.terrain(feet[:,0],feet[:,1]))
        up=self.get_upvector(data)[2]
        invalid=~jp.all(jp.isfinite(data.qpos))|~jp.all(jp.isfinite(data.qvel))
        torso=jp.any(data.sensordata[self._trunk_contact]>0)
        offpath=(jp.abs(data.qpos[1])>.5)|(data.qpos[0]<self.left_goal-.8)|(data.qpos[0]>self.right_goal+.8)
        fallen=(up<.45)|(clearance<.12)
        failure=invalid|torso|offpath|fallen
        info["reason"]=jp.where(invalid,4,jp.where(torso,2,jp.where(offpath,3,jp.where(fallen,1,0)))).astype(jp.int32)
        distance=(info["goal_x"]-data.qpos[0])*info["direction"]
        w,x,y,z=data.qpos[3:7]
        yaw=jp.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        heading_error=jp.arctan2(jp.sin(info["heading"]-yaw),jp.cos(info["heading"]-yaw))
        speed=jp.where(jp.abs(distance)<.18,0.,jp.clip(distance*1.2,-.15,info["cruise"]))
        info["command"]=jp.array([speed,0.,jp.clip(heading_error*1.5,-.6,.6)])
        stable=(jp.abs(distance)<.22)&(jp.linalg.norm(self.get_local_linvel(data))<.12)&(
            jp.linalg.norm(self.get_gyro(data))<.3)&(up>.9)&(jp.sum(contact)>=3)&~failure
        info["hold"]=jp.where(stable,info["hold"]+1,0)
        success=(info["hold"]>=50)&~failure
        info["episode_steps"]+=1
        limit=jp.where(info["task"]==2,4500,jp.where(info["task"]==3,500,2400))
        timeout=info["episode_steps"]>=limit
        moving=jp.abs(speed)>.05
        info["phase"]=jp.mod(info["phase"]+2*jp.pi*1.6*self.dt*moving,2*jp.pi)
        # Soft diagonal timing prior, plus explicit long-air/flight penalties.
        desired=jp.sin(info["phase"]+jp.array([0.,jp.pi,jp.pi,0.]))<=0
        gait=jp.mean(jp.where(contact==desired,1.,0.))*moving
        flight=~jp.any(contact)
        info["flight_steps"]=jp.where(flight,info["flight_steps"]+1,0)
        info["max_flight"]=jp.maximum(info["max_flight"],info["flight_steps"])
        info["contact_count"]+=contact.astype(jp.float32)*moving
        info["climb_frames"]+=moving.astype(jp.int32)
        info["airborne_count"]+=(flight&moving).astype(jp.int32)
        rewards=self._get_reward(data,action,info,state.metrics,failure,first_contact,contact)
        rewards={k:v*self._config.reward_config.scales[k] for k,v in rewards.items()}
        rewards.update(body_height=-2*jp.square(clearance-.28),heading=-.4*heading_error**2,
            lateral=-jp.square(data.qpos[1]),long_air=-2*jp.sum(jp.square(jp.maximum(info["feet_air_time"]-.45,0))),
            flight=-1.5*flight,gait=.4*gait,stand=1.0*stable,
            success=2.*success/self.dt,failure=-1.*failure/self.dt)
        reward=jp.where(invalid,-1.,sum(rewards.values())*self.dt)
        info.update(last_last_act=info["last_act"],last_act=action,last_contact=contact,
            feet_air_time=info["feet_air_time"]*(~contact),swing_peak=info["swing_peak"]*(~contact),
            success=success,failure=failure,timeout=timeout,episode_return=info["episode_return"]+reward)
        return state.replace(data=data,obs=self._get_obs(data,info),info=info,metrics=rewards,
            reward=reward,done=(failure|success|timeout).astype(jp.float32))
