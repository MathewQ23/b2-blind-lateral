"""Left-right mirror symmetry for the B2 payload task: an exactly equivariant actor-critic and two gait costs.

Mirroring reflects the world through the robot's sagittal plane (y -> -y). The scene, the payload and the
official B2 model are left-right symmetric (combined centre-of-mass offset about 1.4 mm), and every reward term
is mirror invariant. The trot clock pairs FR+RL and FL+RR, so a mirrored trot is the same trot half a cycle later.
"""
import json
import flax.linen as nn
import jax.numpy as jp
import numpy as np
from stairs.b2_posture import B2PostureEnv

# Actuator order FR, FL, RR, RL x (hip, thigh, calf). Mirroring swaps the legs and reverses hip abduction.
ACTION_PERM=np.array([3,4,5,0,1,2,9,10,11,6,7,8])
ACTION_SIGN=np.tile(np.array([-1.,1.,1.]),4)


def _observation_map():
    """Permutation and sign of the 82 observations (B2PayloadEnv._obs) under the mirror."""
    perm,sign=[],[]
    def block(p,s,offset):
        perm.extend(np.asarray(p)+offset);sign.extend(s)
    block([0,1,2],[1,-1,1],0)                       # base linear velocity, body frame
    block([0,1,2],[-1,1,-1],3)                      # gyro: angular velocity is a pseudovector
    block([0,1,2],[1,-1,1],6)                       # gravity in body frame
    for offset in (9,21,33):block(ACTION_PERM,ACTION_SIGN,offset)  # joint offsets, joint velocities, last action
    block([0,1,2],[1,-1,-1],45)                     # command: forward speed, lateral speed, yaw rate
    block([3*(i//3)+2-i%3 for i in range(24)],[1]*24,48)  # terrain scan, columns at y = -.35, 0, .35
    block([1,0,3,2],[1]*4,72)                       # last foot contact FR, FL, RR, RL
    block([0,1],[-1,-1],76)                         # trot clock: phase + pi
    block([0,1],[-1,1],78)                          # heading error: negated
    block([0,1],[1,1],80)                           # distance to goal, payload / 30
    return np.array(perm),np.array(sign,dtype=np.float32)


OBSERVATION_PERM,OBSERVATION_SIGN=_observation_map()
assert sorted(OBSERVATION_PERM)==list(range(82))


def mirror_observation(obs):
    return obs[...,OBSERVATION_PERM]*OBSERVATION_SIGN


def mirror_action(action):
    return action[...,ACTION_PERM]*ACTION_SIGN


def symmetric_norm(norm):
    """Observation statistics of a mirror-closed data set, so mirroring commutes with normalization."""
    mean,var,count=norm
    return .5*(mean+mirror_observation(mean)),.5*(var+var[...,OBSERVATION_PERM]),count


class SymmetricActorCritic(nn.Module):
    """Same layers and parameter names as stairs.train.ActorCritic, symmetrized:
    pi(s) = (f(s) + M f(M s)) / 2 and V(s) = (V(s) + V(M s)) / 2, with one exploration width per mirrored joint pair.
    Inputs must be normalized with symmetric statistics (symmetric_norm)."""
    actions: int = 12
    width: int = 128
    minimum_logstd: float = -3.
    maximum_logstd: float = -.5

    def setup(self):
        self.actor_0=nn.Dense(self.width);self.actor_1=nn.Dense(self.width)
        self.critic_0=nn.Dense(self.width);self.critic_1=nn.Dense(self.width)
        self.mean=nn.Dense(self.actions,kernel_init=nn.initializers.orthogonal(.01))
        self.value=nn.Dense(1)
        self.logstd=self.param('logstd',nn.initializers.constant(-.7),(self.actions,))

    def branch(self,obs):
        actor=nn.elu(self.actor_1(nn.elu(self.actor_0(obs))))
        critic=nn.elu(self.critic_1(nn.elu(self.critic_0(obs))))
        return self.mean(actor),self.value(critic)[...,0]

    def __call__(self,obs):
        mean,value=self.branch(obs)
        mirrored_mean,mirrored_value=self.branch(mirror_observation(obs))
        logstd=.5*(self.logstd+self.logstd[ACTION_PERM])
        return (.5*(mean+mirror_action(mirrored_mean)),jp.clip(logstd,self.minimum_logstd,self.maximum_logstd),
                .5*(value+mirrored_value))


class B2SymmetricEnv(B2PostureEnv):
    """b2_payload_v3 plus a hip-splay cost and a short-contact cost; physics and acceptance unchanged."""
    hip_splay_weight=1.
    short_contact_cost=.04
    short_contact_s=.1
    # Charged from the start, the cost teaches a random policy to keep its feet planted instead of learning to walk
    # (6M-sample pilot: progress -0.08 vs +1.54 without it). It is off on the first stage and reaches full weight at stage 4.
    short_contact_ramp_stages=4

    def __init__(self,directory,stage=0,seed=20261028):
        super().__init__(directory,stage,seed)
        self._hips=jp.array([0,3,6,9])
        ramp=min(1.,stage/self.short_contact_ramp_stages) if self.short_contact_ramp_stages else 1.
        self.short_contact_weight=self.short_contact_cost*ramp
        self.reward_keys+=['hip_splay','short_contact']
        self.scene_metadata.update(task_version='b2_payload_v4',reward_parts=self.reward_keys,
            reward_revision=('Mirror-equivariant actor-critic; hip abduction cost against splayed legs and a cost per '
                             'touchdown or lift-off ending a swing or stance shorter than 0.1 s, against tapping; '
                             'acceptance and physics unchanged'),
            symmetry=dict(observation_perm=OBSERVATION_PERM.tolist(),observation_sign=OBSERVATION_SIGN.tolist(),
                          action_perm=ACTION_PERM.tolist(),action_sign=ACTION_SIGN.tolist()),
            hip_splay_weight=self.hip_splay_weight,short_contact_cost=self.short_contact_weight,short_contact_s=self.short_contact_s,
            short_contact_full_cost=self.short_contact_cost,short_contact_ramp_stages=self.short_contact_ramp_stages)
        (self.directory/'metadata.json').write_text(json.dumps(self.scene_metadata,indent=2)+'\n')

    def reset(self,key,task_override=-1):
        state=super().reset(key,task_override)
        info=dict(state.info);info['feet_contact_time']=jp.zeros(4)
        return state.replace(info=info)

    def step(self,state,action):
        following=super().step(state,action)
        data=following.data;info=dict(following.info);rewards=dict(following.metrics)
        hips=data.qpos[self._qadr][self._hips]
        rewards['hip_splay']=-self.hip_splay_weight*jp.sum(jp.square(hips))
        # Tapping is a touchdown after a very short swing or a lift-off after a very short stance; ordinary steps cost nothing.
        before,now=state.info['last_contact'],info['last_contact']
        touchdown=now&~before&(state.info['feet_air_time']<self.short_contact_s)
        liftoff=~now&before&(state.info['feet_contact_time']<self.short_contact_s)
        short=jp.sum(touchdown|liftoff)*(state.info['episode_steps']>0)
        info['feet_contact_time']=(state.info['feet_contact_time']+self.dt)*now
        rewards['short_contact']=-self.short_contact_weight*short/self.dt
        valid=jp.all(jp.isfinite(data.qpos))&jp.all(jp.isfinite(data.qvel))
        reward=jp.where(valid,following.reward+(rewards['hip_splay']+rewards['short_contact'])*self.dt,-1.)
        info['episode_return']=state.info['episode_return']+reward
        return following.replace(info=info,metrics=rewards,reward=reward)
