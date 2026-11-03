"""Terrain-aware B2 PPO task with genuine fixed payload inertias and torque PD."""
from pathlib import Path
import json
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from mujoco_playground._src import mjx_env
from stairs.b2_scene import build_b2_scene,STAGES,FEET,SETTINGS


class B2PayloadEnv:
    dt=.02
    sim_dt=.004
    n_substeps=5
    episode_length=4500
    action_scale=.6

    def __init__(self,directory,stage=0,seed=20261028):
        self.directory=Path(directory);self.stage=stage
        self.height,self.payload_kg=STAGES[stage]
        self.xml_path,self.scene_metadata,self.route=build_b2_scene(directory,stage,seed)
        self.mj_model=mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.mjx_model=mjx_env.put_model(self.mj_model,impl='warp')
        m=self.mj_model
        self._qadr=jp.array([int(m.jnt_qposadr[m.joint(n).id]) for n in self.scene_metadata['actuator_joint_order']])
        self._vadr=jp.array([int(m.jnt_dofadr[m.joint(n).id]) for n in self.scene_metadata['actuator_joint_order']])
        limits=np.array([m.joint(n).range for n in self.scene_metadata['actuator_joint_order']])
        self._lowers,self._uppers=jp.array(limits[:,0]),jp.array(limits[:,1])
        self._torque_limits=jp.array(m.actuator_ctrlrange[:,1])
        self._default_pose=jp.tile(jp.array([0.,.8,-1.6]),4)
        self._feet_site_id=np.array([m.site(f).id for f in FEET])
        self._foot_linvel_sensor_adr=jp.array([[int(m.sensor_adr[m.sensor(f'{f}_velocity').id])+i for i in range(3)] for f in FEET])
        self._all_foot_contacts=jp.array([[int(m.sensor_adr[m.sensor(f'{f}_contact').id])] for f in FEET])
        self._trunk_contact=jp.array([int(m.sensor_adr[m.sensor(n).id]) for n in self.scene_metadata['trunk_contact_sensors']])
        self._base_id=m.body('base_link').id
        self._starts=jp.array([m.key(n).qpos for n in self.scene_metadata['keyframes']])
        self.left_goal=float(self._starts[0,0]);self.right_goal=float(self._starts[2,0])
        summit=self.route.segment('summit');self.summit_center=(summit['x0']+summit['x1'])/2
        self._scan_x=jp.repeat(jp.linspace(-.6,1.2,8),3)
        self._scan_y=jp.tile(jp.array([-.35,0.,.35]),8)
        self._landings=jp.array([[(s['x0']+s['x1'])/2 for s in self.route.segments if s['name'].startswith(prefix)]
                                for prefix in ('up_landing_','down_landing_')])
        self.reward_keys=['tracking','progress','orientation','vertical_velocity','angular_velocity','pose',
            'energy','torque','action_rate','slip','body_height','heading','lateral','long_air','flight',
            'gait','stand','time_cost','success','failure','stall']
        self.scene_metadata.update(reward_parts=self.reward_keys,observation_size=82,
            task_names=['ascent','descent','traverse','stand'],
            success='Goal distance < .22 m, linear speed < .12 m/s, angular speed < .3 rad/s, upright > .9, >=3 feet for 1 s')
        (self.directory/'metadata.json').write_text(json.dumps(self.scene_metadata,indent=2)+'\n')

    def terrain(self,x,y=0.):
        return self.route.height_at(x,y,array_module=jp)

    def get_upvector(self,data):
        return data.xmat[self._base_id,:,2]

    def get_local_linvel(self,data):
        return data.xmat[self._base_id].T@data.qvel[:3]

    def get_gyro(self,data):
        start=int(self.mj_model.sensor_adr[self.mj_model.sensor('imu_gyro').id])
        return data.sensordata[start:start+3]

    def _obs(self,data,info):
        key,noise_key=jax.random.split(info['rng']);info['rng']=key
        rotation=data.xmat[self._base_id]
        base=jp.concatenate((self.get_local_linvel(data),self.get_gyro(data),rotation.T@jp.array([0.,0.,-1.]),
            data.qpos[self._qadr]-self._default_pose,data.qvel[self._vadr],info['last_act'],info['command']))
        noise=jp.concatenate((jp.full(3,.025),jp.full(3,.05),jp.full(3,.0125),jp.full(12,.0075),jp.full(12,.375),jp.zeros(15)))
        base=base+jax.random.uniform(noise_key,base.shape,minval=-1.,maxval=1.)*noise
        yaw=jp.arctan2(rotation[1,0],rotation[0,0])
        sx=data.qpos[0]+jp.cos(yaw)*self._scan_x-jp.sin(yaw)*self._scan_y
        sy=data.qpos[1]+jp.sin(yaw)*self._scan_x+jp.cos(yaw)*self._scan_y
        scan=jp.clip(self.terrain(sx,sy)-data.qpos[2],-1.4,.8)
        heading=info['heading']-yaw
        extra=jp.concatenate((scan,info['last_contact'].astype(jp.float32),
            jp.array([jp.sin(info['phase']),jp.cos(info['phase']),jp.sin(heading),jp.cos(heading),
                jp.clip((info['goal_x']-data.qpos[0])*info['direction'],-1,3),self.payload_kg/30.])))
        return {'state':jp.concatenate((base,extra))}

    def reset(self,key,task_override=-1):
        key,kt,kr,kx,ky,kv,kp,kl=jax.random.split(key,8)
        draw=jax.random.uniform(kt)
        task=jp.where(draw<.4,0,jp.where(draw<.8,1,jp.where(draw<.95,2,3)))
        task=jp.where(task_override>=0,task_override,task).astype(jp.int32)
        reverse=jax.random.bernoulli(kr);descending=task==1
        index=jp.where(reverse,jp.where(descending,3,2),jp.where(descending,1,0))
        q=self._starts[index]
        partial=(task_override<0)&(task<2)&(jax.random.uniform(kl)<.2)&(self.stage>=2)
        side=jp.where(task==0,jp.where(reverse,1,0),jp.where(reverse,0,1))
        landing=self._landings[side,jax.random.randint(kp,(),0,2)]
        q=q.at[0].set(jp.where(partial,landing,q[0])+jax.random.uniform(kx,(),minval=-.06,maxval=.06))
        q=q.at[1].set(jax.random.uniform(ky,(),minval=-.04,maxval=.04))
        q=q.at[2].set(self.terrain(q[0],q[1])+.55)
        heading=jp.where(reverse,jp.pi,0.);direction=jp.where(reverse,-1.,1.)
        yaw=heading+jax.random.uniform(kp,(),minval=-.05,maxval=.05)
        q=q.at[3:7].set(jp.array([jp.cos(yaw/2),0.,0.,jp.sin(yaw/2)]))
        goal=jp.where(task==0,self.summit_center,jp.where(reverse,self.left_goal,self.right_goal))
        goal=jp.where(task==3,q[0],goal)
        cruise=jax.random.uniform(kv,(),minval=.25,maxval=.4)
        data=mjx_env.make_data(self.mj_model,qpos=q,qvel=jp.zeros(self.mj_model.nv),ctrl=jp.zeros(12),
            impl='warp',naconmax=32768,njmax=512)
        data=mjx.forward(self.mjx_model,data)
        zero=jp.array(0.,jp.float32);counter=jp.array(0,jp.int32)
        info=dict(rng=key,command=jp.array([jp.where(task==3,0.,cruise),0.,0.]),
            last_act=jp.zeros(12),last_contact=jp.zeros(4,dtype=bool),feet_air_time=jp.zeros(4),
            episode_steps=counter,episode_id=counter,episode_return=zero,task=task,partial=partial,
            heading=heading,direction=direction,goal_x=goal,cruise=cruise,phase=zero,payload_kg=jp.array(self.payload_kg),
            success=jp.array(False),failure=jp.array(False),timeout=jp.array(False),hold=counter,
            flight_steps=counter,max_flight=counter,contact_count=jp.zeros(4),climb_frames=counter,
            airborne_count=counter,reason=counter,best_distance=jp.abs(goal-q[0]),stagnant_steps=counter)
        return mjx_env.State(data,self._obs(data,info),zero,zero,{k:zero for k in self.reward_keys},info)

    def step(self,state,action):
        info=dict(state.info)
        targets=jp.clip(self._default_pose+action*self.action_scale,self._lowers,self._uppers)
        def physics(data,_):
            torque=SETTINGS['joint_pd_kp']*(targets-data.qpos[self._qadr])-SETTINGS['joint_pd_kd']*data.qvel[self._vadr]
            data=data.replace(ctrl=jp.clip(torque,-self._torque_limits,self._torque_limits))
            return mjx.step(self.mjx_model,data),None
        data=jax.lax.scan(physics,state.data,None,length=5)[0]
        contact=jp.any(data.sensordata[self._all_foot_contacts]>0,axis=-1)
        air=(info['feet_air_time']+self.dt)*(~contact)
        feet=data.site_xpos[self._feet_site_id];velocity=data.sensordata[self._foot_linvel_sensor_adr]
        clearance=data.qpos[2]-jp.mean(self.terrain(feet[:,0],feet[:,1]))
        up=self.get_upvector(data)[2]
        invalid=~jp.all(jp.isfinite(data.qpos))|~jp.all(jp.isfinite(data.qvel))
        torso=jp.any(data.sensordata[self._trunk_contact]>0)
        offpath=(jp.abs(data.qpos[1])>self.route.config.path_width_m/2-.25)|(data.qpos[0]<self.left_goal-.9)|(data.qpos[0]>self.right_goal+.9)
        fallen=(up<.45)|(clearance<.24)
        failure=invalid|torso|offpath|fallen
        reason=jp.where(invalid,4,jp.where(torso,2,jp.where(offpath,3,jp.where(fallen,1,0)))).astype(jp.int32)
        distance=(info['goal_x']-data.qpos[0])*info['direction'];remaining=jp.abs(distance)
        rotation=data.xmat[self._base_id];yaw=jp.arctan2(rotation[1,0],rotation[0,0])
        heading_error=jp.arctan2(jp.sin(info['heading']-yaw),jp.cos(info['heading']-yaw))
        speed=jp.where(remaining<.18,0.,jp.clip(distance*1.2,-.15,info['cruise']))
        info['command']=jp.array([speed,0.,jp.clip(heading_error*1.5,-.6,.6)])
        local=self.get_local_linvel(data);gyro=self.get_gyro(data)
        stable=(remaining<.22)&(jp.linalg.norm(local)<.12)&(jp.linalg.norm(gyro)<.3)&(up>.9)&(jp.sum(contact)>=3)&~failure
        info['hold']=jp.where(stable,info['hold']+1,0);success=(info['hold']>=50)&~failure
        info['episode_steps']+=1
        timeout=info['episode_steps']>=jp.where(info['task']==2,4500,jp.where(info['task']==3,500,2400))
        moving=(remaining>.25)&(info['task']!=3)
        improvement=info['best_distance']-remaining
        info['stagnant_steps']=jp.where(moving&(improvement<.025),info['stagnant_steps']+1,0)
        info['best_distance']=jp.where(improvement>=.025,remaining,info['best_distance'])
        stalled=(info['stagnant_steps']>=300)&moving&~failure&~success
        unused=jp.any(air>2.)&moving&~failure&~success
        reason=jp.where(stalled,8,jp.where(unused,9,reason));failure|=stalled|unused
        phase=jp.mod(info['phase']+2*jp.pi*1.2*self.dt*moving,2*jp.pi)
        desired=jp.sin(phase+jp.array([0.,jp.pi,jp.pi,0.]))<=0
        flight=~jp.any(contact)
        info['flight_steps']=jp.where(flight,info['flight_steps']+1,0)
        info['max_flight']=jp.maximum(info['max_flight'],info['flight_steps'])
        info['contact_count']+=contact.astype(jp.float32)*moving
        info['climb_frames']+=moving.astype(jp.int32);info['airborne_count']+=(flight&moving).astype(jp.int32)
        old_distance=jp.abs(state.info['goal_x']-state.data.qpos[0])
        progress=(old_distance-remaining)/self.dt
        rewards=dict(tracking=jp.exp(-jp.sum(jp.square(info['command'][:2]-local[:2]))/.08)-1.,
            progress=2.*jp.clip(progress/jp.maximum(info['cruise'],.2),-1.,1.25)*moving,
            orientation=-3*jp.sum(jp.square(self.get_upvector(data)[:2])),vertical_velocity=-.5*local[2]**2,
            angular_velocity=-.05*jp.sum(jp.square(gyro[:2]))-.3*gyro[2]**2,
            pose=-.1*jp.sum(jp.square(data.qpos[self._qadr]-self._default_pose)),
            energy=-.00015*jp.sum(jp.abs(data.actuator_force*data.qvel[self._vadr])),
            torque=-.1*jp.sum(jp.square(data.actuator_force/self._torque_limits)),
            action_rate=-.01*jp.sum(jp.square(action-info['last_act'])),
            slip=-.15*jp.sum(jp.sum(jp.square(velocity[:,:2]),axis=-1)*contact),
            body_height=-3*(clearance-.5)**2,heading=-.4*heading_error**2,lateral=-data.qpos[1]**2,
            long_air=-.6*jp.sum(jp.square(jp.clip(air-.55,0,1))),flight=-1.5*flight,
            gait=.4*(jp.mean((contact==desired).astype(jp.float32))-.5)*moving,
            stand=jp.where(info['task']==3,.3,1.)*stable,time_cost=-.1*moving,
            success=2.*success/self.dt,failure=-1.*failure/self.dt,stall=-1.*stalled/self.dt)
        reward=jp.where(invalid,-1.,sum(rewards.values())*self.dt)
        info.update(last_act=action,last_contact=contact,feet_air_time=air,phase=phase,
            success=success,failure=failure,timeout=timeout,reason=reason,episode_return=info['episode_return']+reward)
        return state.replace(data=data,info=info,obs=self._obs(data,info),metrics=rewards,
            reward=reward,done=(failure|success|timeout).astype(jp.float32))


class B2SwingEnv(B2PayloadEnv):
    """Add a bounded, smooth swing-clearance cost; acceptance and physics unchanged."""
    def __init__(self,directory,stage=0,seed=20261028):
        super().__init__(directory,stage,seed)
        self.reward_keys+=['swing_clearance']
        self.scene_metadata.update(task_version='b2_payload_v2',reward_parts=self.reward_keys,
            reward_revision='Smooth diagonal swing-height cost, terrain-relative; no change to task/gait acceptance')
        (self.directory/'metadata.json').write_text(json.dumps(self.scene_metadata,indent=2)+'\n')

    def step(self,state,action):
        following=super().step(state,action)
        info=dict(following.info);rewards=dict(following.metrics)
        moving=(jp.abs(info['goal_x']-following.data.qpos[0])>.25)&(info['task']!=3)
        phase=info['phase']+jp.array([0.,jp.pi,jp.pi,0.])
        swing=jp.maximum(jp.sin(phase),0.)
        feet=following.data.site_xpos[self._feet_site_id]
        clearance=feet[:,2]-self.terrain(feet[:,0],feet[:,1])-.032
        target=.06+.5*self.height
        error=jp.clip((clearance-target*swing)/target,-2.,2.)
        rewards['swing_clearance']=-.8*jp.mean(jp.square(error)*(swing>.1))*moving
        reward=following.reward+rewards['swing_clearance']*self.dt
        info['episode_return']=state.info['episode_return']+reward
        return following.replace(info=info,metrics=rewards,reward=reward)
