"""Versioned recovery task: progress-dependent reward and finer terrain curriculum."""
import json
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env
from stairs.taishan_env import TaishanEnv

RECOVERY_HEIGHTS=tuple(.12+i*.01 for i in range(11))


class RecoveryEnv(TaishanEnv):
    def __init__(self,directory,stage=0,seed=20261028):
        super().__init__(directory,stage,seed,height_override=RECOVERY_HEIGHTS[stage])
        self.reward_keys+=['progress','stall','time_cost']
        self.scene_metadata.update(task_version='recovery_v2',reward_parts=self.reward_keys,
            curriculum_heights=RECOVERY_HEIGHTS,partial_start_probability=.3,
            reward_changes='progress-dependent tracking; no positive stationary tracking bonus; stalled episodes terminate after 5 s')
        (self.directory/'metadata.json').write_text(json.dumps(self.scene_metadata,indent=2)+'\n')
        # Mid-flight training starts use real flat landings, in both directions.
        centers=[]
        for prefix in ('up_landing_','down_landing_'):
            centers.append([(s['x0']+s['x1'])/2 for s in self.route.segments if s['name'].startswith(prefix)])
        self.landings=jp.array(centers)

    def reset(self,key,task_override=-1):
        state=super().reset(key,task_override)
        info=dict(state.info)
        key,ka,kb=jax.random.split(info['rng'],3)
        partial=(task_override<0)&(info['task']<2)&(jax.random.uniform(ka)<.3)
        # For reversed travel, descent lies on the nominal up side, and conversely.
        side=jp.where(info['task']==0,jp.where(info['direction']>0,0,1),jp.where(info['direction']>0,1,0))
        landing=self.landings[side,jax.random.randint(kb,(),0,2)]
        q=state.data.qpos.at[0].set(jp.where(partial,landing,state.data.qpos[0]))
        q=q.at[2].set(self.terrain(q[0],q[1])+.278)
        data=mjx_env.make_data(self.mj_model,qpos=q,qvel=jp.zeros(self.mj_model.nv),
            ctrl=self._default_pose,impl='warp',naconmax=32768,njmax=512)
        data=mjx.forward(self.mjx_model,data)
        distance=jp.abs(info['goal_x']-q[0])
        info.update(rng=key,partial=partial,best_distance=distance,stagnant_steps=jp.array(0,jp.int32))
        return state.replace(data=data,info=info,obs=self._get_obs(data,info))

    def step(self,state,action):
        following=super().step(state,action)
        info=dict(following.info);rewards=dict(following.metrics)
        distance=jp.abs(info['goal_x']-following.data.qpos[0])
        moving=(distance>.25)&(info['task']!=3)
        improvement=state.info['best_distance']-distance
        info['stagnant_steps']=jp.where(moving & (improvement<.025),state.info['stagnant_steps']+1,0)
        info['best_distance']=jp.where(improvement>=.025,distance,state.info['best_distance'])
        stalled=(info['stagnant_steps']>=250)&moving&~info['failure']&~info['success']
        # Only progress toward the actual goal earns positive locomotion reward.
        previous_distance=jp.abs(state.info['goal_x']-state.data.qpos[0])
        velocity=(previous_distance-distance)/self.dt
        rewards['progress']=2.0*jp.clip(velocity/jp.maximum(info['cruise'],.2),-1.,1.25)*moving
        local_velocity=self.get_local_linvel(following.data)
        error=jp.sum(jp.square(info['command'][:2]-local_velocity[:2]))
        rewards['tracking_lin_vel']=1.0*(jp.exp(-error/.05)-1.)
        rewards['tracking_ang_vel']-=self._config.reward_config.scales.tracking_ang_vel
        rewards['pose']-=self._config.reward_config.scales.pose
        rewards['gait']-=.2*(jp.abs(info['command'][0])>.05)
        rewards['stall']=-2.*stalled/self.dt
        rewards['time_cost']=-.15*moving
        rewards['stand']=jp.where(info['task']==3,rewards['stand']*.3,rewards['stand'])
        info['failure']=info['failure']|stalled
        info['reason']=jp.where(stalled,8,info['reason'])
        reward=sum(rewards.values())*self.dt
        reward=jp.where(jp.all(jp.isfinite(following.data.qpos)),reward,-1.)
        info['episode_return']=state.info['episode_return']+reward
        return following.replace(info=info,metrics=rewards,reward=reward,
            done=(following.done.astype(bool)|stalled).astype(jp.float32))


class BoundedRecoveryEnv(RecoveryEnv):
    """Keep rare long-air outliers from dominating the critic and policy updates."""
    def __init__(self,directory,stage=0,seed=20261028):
        super().__init__(directory,stage,seed)
        self.scene_metadata.update(task_version='recovery_v3',long_air_limit_s=2.0,
            long_air_penalty='capped at -0.6 per foot per simulated second; prolonged nonuse ends episode')
        (self.directory/'metadata.json').write_text(json.dumps(self.scene_metadata,indent=2)+'\n')

    def step(self,state,action):
        following=super().step(state,action)
        info=dict(following.info);rewards=dict(following.metrics)
        moving=(jp.abs(info['goal_x']-following.data.qpos[0])>.25)&(info['task']!=3)
        unused=jp.any(info['feet_air_time']>2.)&moving&~info['failure']&~info['success']
        rewards['long_air']=-.6*jp.sum(jp.square(jp.clip(info['feet_air_time']-.45,0.,1.)))
        rewards['failure']-=unused/self.dt
        info['failure']|=unused
        info['reason']=jp.where(unused,9,info['reason'])
        reward=jp.where(jp.all(jp.isfinite(following.data.qpos)),sum(rewards.values())*self.dt,-1.)
        info['episode_return']=state.info['episode_return']+reward
        return following.replace(info=info,metrics=rewards,reward=reward,
            done=(following.done.astype(bool)|unused).astype(jp.float32))
