#!/usr/bin/env python3
"""Independent task and gait acceptance for a shared Taishan policy."""
import os
os.environ.setdefault("JAX_PLATFORMS","cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import argparse
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jp
import numpy as np
from stairs.train import ActorCritic,normalize
from stairs.taishan_env import TaishanEnv,HEIGHTS,TASKS
from stairs.taishan_recovery import RecoveryEnv,BoundedRecoveryEnv,RECOVERY_HEIGHTS
from stairs.b2_env import B2PayloadEnv,B2SwingEnv
from stairs.b2_posture import B2PostureEnv
from stairs.b2_blind import B2BlindEnv
from stairs.b2_blind_ascent import B2BlindAscentEnv
from stairs.b2_blind_centered import B2BlindCenteredEnv
from stairs.b2_blind_odometry import B2BlindOdometryEnv
from stairs.b2_blind_lateral import B2BlindLateralEnv
from stairs.b2_symmetry import B2SymmetricEnv,SymmetricActorCritic
from stairs.b2_scene import STAGES as B2_STAGES


def main():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--episodes",type=int,default=20,help="Episodes per task")
    p.add_argument("--seed",type=int,default=900000)
    p.add_argument('--terrain-seed',type=int,help='Separate terrain seed for controlled diagnostics')
    p.add_argument('--action-sampling',choices=['mean','stochastic'],default='mean')
    p.add_argument('--diagnostic',action='store_true',help='Never mark this run as independent acceptance')
    p.add_argument("--stage",type=int,default=-1,help="-1 uses the checkpoint stage")
    p.add_argument('--task-version',choices=['auto','legacy','recovery_v2','recovery_v3','b2_payload_v1','b2_payload_v2','b2_payload_v3','b2_payload_v4','b2_blind_v1','b2_blind_ascent_v1','b2_blind_centered_v1','b2_blind_odometry_v1','b2_blind_lateral_v1'],default='auto')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with a.checkpoint.open("rb") as f:saved=pickle.load(f)
    version=saved.get('task_version','legacy') if a.task_version=='auto' else a.task_version
    B2_VERSIONS=('b2_payload_v1','b2_payload_v2','b2_payload_v3','b2_payload_v4','b2_blind_v1','b2_blind_ascent_v1','b2_blind_centered_v1','b2_blind_odometry_v1','b2_blind_lateral_v1')
    b2=version in B2_VERSIONS
    if (saved.get('task_version') in B2_VERSIONS)!=b2:
        raise ValueError('B2 and Go1 checkpoints use different observations and actuator mappings; cross-robot evaluation is invalid')
    heights=tuple(s[0] for s in B2_STAGES) if b2 else (RECOVERY_HEIGHTS if version!='legacy' else HEIGHTS)
    env_type={'legacy':TaishanEnv,'recovery_v2':RecoveryEnv,'recovery_v3':BoundedRecoveryEnv,'b2_payload_v1':B2PayloadEnv,'b2_payload_v2':B2SwingEnv,'b2_payload_v3':B2PostureEnv,'b2_payload_v4':B2SymmetricEnv,'b2_blind_v1':B2BlindEnv,'b2_blind_ascent_v1':B2BlindAscentEnv,'b2_blind_centered_v1':B2BlindCenteredEnv,'b2_blind_odometry_v1':B2BlindOdometryEnv,'b2_blind_lateral_v1':B2BlindLateralEnv}[version]
    stage=saved['stage'] if a.stage<0 else a.stage
    terrain_seed=a.seed if a.terrain_seed is None else a.terrain_seed
    independent=(not a.diagnostic and a.action_sampling=='mean' and terrain_seed!=saved.get('seed'))
    env=env_type(a.output/"scene",stage,terrain_seed)
    # The network class follows the checkpoint, not the evaluation task, so a symmetric policy is never evaluated unsymmetrized.
    net=(SymmetricActorCritic if saved.get('task_version')=='b2_payload_v4' else ActorCritic)(width=saved["width"],minimum_logstd=saved.get('minimum_logstd',-3.),
                   maximum_logstd=saved.get('maximum_logstd',-.5))
    params,norm=jax.device_put(saved["params"]),jax.device_put(saved["norm"])
    task_ids=[0,3] if version in ('b2_blind_ascent_v1','b2_blind_centered_v1','b2_blind_odometry_v1','b2_blind_lateral_v1') else [0,1,2,3]
    task_names=[TASKS[i] for i in task_ids]
    tasks=jp.repeat(jp.array(task_ids),a.episodes);n=len(tasks)
    keys=jp.stack([jax.random.PRNGKey(a.seed+i) for i in range(n)])
    state=jax.jit(jax.vmap(env.reset))(keys,tasks)
    @jax.jit
    def run(state):
        def tick(carry,_):
            state,finished,key=carry
            mean,logstd,_=net.apply(params,normalize(state.obs["state"],norm))
            key,sample=jax.random.split(key)
            if a.action_sampling=='stochastic':
                stand_scale=.04 if b2 else .2
                offset=jp.where(jp.linalg.norm(state.info['command'][:,:2],axis=-1)<.01,jp.log(stand_scale),0.)
                u=mean+jp.exp(logstd+offset[:,None])*jax.random.normal(sample,mean.shape)
            else:u=mean
            nextstate=jax.vmap(env.step)(state,jp.tanh(u))
            def retain(new,old):
                if new.ndim==0 or new.shape[0]!=n:return new
                return jp.where(finished.reshape(finished.shape+(1,)*(new.ndim-1)),old,new)
            nextstate=jax.tree.map(retain,nextstate,state)
            finished=finished|(nextstate.done>0)
            trace=dict(qpos=nextstate.data.qpos,qvel=nextstate.data.qvel,ctrl=nextstate.data.ctrl,
                foot_contact=nextstate.info["last_contact"],episode_step=nextstate.info["episode_steps"],
                done=nextstate.done,success=nextstate.info["success"],failure=nextstate.info["failure"])
            return (nextstate,finished,key),trace
        return jax.lax.scan(tick,(state,jp.zeros(n,dtype=bool),jax.random.PRNGKey(a.seed+1000000)),None,length=4500)
    (state,_,_),trace=jax.device_get(run(state))
    np.savez_compressed(a.output/"episodes.npz",**trace)
    rows=[]
    for i in range(n):
        info={k:np.asarray(v[i]) for k,v in state.info.items()}
        count=max(int(info["climb_frames"]),1)
        contact=info["contact_count"]/count;flight=float(info["airborne_count"]/count)
        gait=bool(contact.min()>=.15 and flight<=.1 and info["max_flight"]<=10) if info["task"]!=3 else True
        rows.append(dict(seed=a.seed+i,task=TASKS[int(info["task"])],success=bool(info["success"]),
            gait_passed=gait,accepted=bool(info["success"]) and gait,failure=bool(info["failure"]),
            timeout=bool(info["timeout"]),reason=int(info['reason']),goal_x=float(info['goal_x']),final_x=float(state.data.qpos[i,0]),
            contact_fraction=contact.tolist(),airborne_fraction=flight,
            max_airborne_s=float(info["max_flight"]*.02),duration_s=float(info["episode_steps"]*.02)))
    rates={name:float(np.mean([r["accepted"] for r in rows if r["task"]==name])) for name in task_names}
    summary=dict(kind="independent_evaluation" if independent else 'diagnostic_evaluation',checkpoint=str(a.checkpoint.resolve()),stage=stage,height=heights[stage],task_version=version,
        robot='B2' if b2 else 'Go1',payload_kg=getattr(env,'payload_kg',0.),
        terrain_seed=terrain_seed,new_terrain_seed=terrain_seed if independent else None,action_sampling=a.action_sampling,
        episodes_per_task=a.episodes,evaluated_tasks=task_names,accepted_rates=rates,
        passed=independent and a.episodes>=20 and min(rates.values())>=.8,
        target_passed=independent and stage==saved.get('target_stage',len(heights)-1) and a.episodes>=20 and min(rates.values())>=.8,results=rows)
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!="results"}),flush=True)


if __name__=="__main__":main()
