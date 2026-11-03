"""Unified ascent/descent PPO with four complete world histories."""
import os
os.environ.setdefault("JAX_PLATFORMS","cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
os.environ.setdefault("XLA_FLAGS","--xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("OMP_NUM_THREADS","4")
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import signal
import time

import jax
import jax.numpy as jp
import numpy as np
import optax
from flax.training.train_state import TrainState
from stairs.train import ActorCritic, normalize, logprob, build_functions, atomic_pickle
from stairs.taishan_env import TaishanEnv, HEIGHTS, TASKS
from stairs.taishan_recovery import RecoveryEnv,BoundedRecoveryEnv,RECOVERY_HEIGHTS
from stairs.b2_env import B2PayloadEnv,B2SwingEnv
from stairs.b2_posture import B2PostureEnv
from stairs.b2_blind import B2BlindEnv
from stairs.b2_blind_ascent import B2BlindAscentEnv
from stairs.b2_blind_centered import B2BlindCenteredEnv
from stairs.b2_blind_odometry import B2BlindOdometryEnv
from stairs.b2_blind_lateral import B2BlindLateralEnv
from stairs.b2_symmetry import B2SymmetricEnv,SymmetricActorCritic,mirror_observation,symmetric_norm
from stairs.b2_scene import STAGES as B2_STAGES
from stairs.selected_recording import SelectedRecorder, archive_end
from stairs.exploration import exploration_settings


def collector(env,net,horizon,num_envs,record_count,exploration_profile='standard'):
    overrides=jp.where(jp.arange(num_envs)<4,jp.arange(num_envs),-1)
    reset=jax.vmap(env.reset)
    def select(a,b,mask):
        if a.ndim==0 or a.shape[0]!=num_envs:return a
        return jp.where(mask.reshape(mask.shape+(1,)*(a.ndim-1)),b,a)

    @jax.jit
    def collect(state,key,params,norm):
        def tick(carry,_):
            state,key=carry
            key,sample,resetkey=jax.random.split(key,3)
            raw=state.obs["state"];x=normalize(raw,norm)
            mean,logstd,value=net.apply(params,x)
            offset,stand_mask,low_noise,noise_scale=exploration_settings(state.info['command'],jp.arange(num_envs),
                b2=isinstance(env,B2PayloadEnv),profile=exploration_profile)
            logstd=logstd+offset[:,None]
            u=mean+jp.exp(logstd)*jax.random.normal(sample,mean.shape)
            action=jp.tanh(u)
            following=jax.vmap(env.step)(state,action)
            _,_,nextvalue=net.apply(params,normalize(following.obs["state"],norm))
            done=following.done>0
            info=following.info;data=following.data
            take=lambda a:a[:record_count]
            record={k:take(v) for k,v in dict(
                qpos_before=state.data.qpos,qvel_before=state.data.qvel,
                qpos=data.qpos,qvel=data.qvel,qacc=data.qacc,ctrl=data.ctrl,
                actuator_force=data.actuator_force,qfrc_constraint=data.qfrc_constraint,
                sensordata=data.sensordata,action=action,raw_action=u,
                observation=raw,next_observation=following.obs["state"],
                policy_mean=mean,policy_logstd=logstd,value=value,log_probability=logprob(u,mean,logstd),
                command=state.info["command"],next_command=info["command"],reward=following.reward,
                reward_parts=jp.stack([following.metrics[k] for k in env.reward_keys],axis=-1),
                done=done,success=info["success"],failure=info["failure"],timeout=info["timeout"],
                reason=info["reason"],episode_id=state.info["episode_id"],episode_step=info["episode_steps"],
                sim_time=data.time,task=info["task"],goal_x=info["goal_x"],direction=info["direction"],
                foot_contact=info["last_contact"],foot_position=data.site_xpos[:,env._feet_site_id],
                foot_velocity=data.sensordata[:,env._foot_linvel_sensor_adr],
                feet_air_time=info["feet_air_time"],contact_count=info["contact_count"],
                climb_frames=info["climb_frames"],airborne_count=info["airborne_count"],max_flight=info["max_flight"],
                nefc=data._impl.nefc).items()}
            if isinstance(env,B2PayloadEnv):
                record['payload_kg']=take(info['payload_kg'])
                record['joint_target']=take(jp.clip(env._default_pose+action*env.action_scale,env._lowers,env._uppers))
                record['exploration_scale']=take(noise_scale)
                record['low_noise_group']=take(low_noise)
            summary={k:info[k] for k in ("task","success","failure","timeout","reason","episode_return",
                "contact_count","climb_frames","airborne_count","max_flight")}
            summary['partial']=info.get('partial',jp.zeros(num_envs,dtype=bool))
            summary['low_noise_group']=low_noise
            summary.update(done=done,max_constraints=jp.max(data._impl.nefc),
                finite=jp.all(jp.isfinite(data.qpos))&jp.all(jp.isfinite(data.qvel)),
                foot_contact=info["last_contact"],
                reward_parts_mean=jp.stack([jp.mean(following.metrics[k]) for k in env.reward_keys]))
            batch=dict(x=x,rawobs=raw,u=u,logp=logprob(u,mean,logstd),std_offset=offset,
                stand_mask=stand_mask,
                old_mean=mean,old_logstd=logstd,
                value=value,nextvalue=nextvalue,reward=following.reward,done=done,
                terminal=info["failure"]|info["success"],record=record,summary=summary)
            def autoreset(nextstate):
                fresh=reset(jax.random.split(resetkey,num_envs),overrides)
                restored=jax.tree.map(lambda a,b:select(a,b,done),nextstate,fresh)
                restored.info["episode_id"]=state.info["episode_id"]+done.astype(jp.int32)
                return restored
            restored=jax.lax.cond(jp.any(done),autoreset,lambda s:s,following)
            return (restored,key),batch
        return jax.lax.scan(tick,(state,key),None,length=horizon)
    return collect


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True,type=Path)
    p.add_argument("--num-envs",type=int,default=512)
    p.add_argument("--horizon",type=int,default=32)
    p.add_argument("--epochs",type=int,default=4)
    p.add_argument("--width",type=int,default=192)
    p.add_argument("--steps",type=int,default=100_000_000)
    p.add_argument("--seed",type=int,default=20261028)
    p.add_argument("--resume",action="store_true")
    p.add_argument('--resize-envs',action='store_true',help='Explicitly reset the physics pool when changing parallel world count on resume')
    p.add_argument("--checkpoint-every",type=int,default=2_000_000)
    p.add_argument('--task-version',choices=['legacy','recovery_v2','recovery_v3','b2_payload_v1','b2_payload_v2','b2_payload_v3','b2_payload_v4','b2_blind_v1','b2_blind_ascent_v1','b2_blind_centered_v1','b2_blind_odometry_v1','b2_blind_lateral_v1'],default='b2_blind_lateral_v1')
    p.add_argument('--warm-start',type=Path)
    p.add_argument('--warm-start-mode',choices=['actor','actor-critic'],default='actor')
    p.add_argument('--actor-lr',type=float,default=1e-4)
    p.add_argument('--start-stage',type=int,default=0)
    p.add_argument('--target-stage',type=int,default=14,help='Stop after this curriculum stage passes; default 14 is 22cm/20kg')
    p.add_argument('--max-stage-steps',type=int,default=12_000_000)
    p.add_argument('--optimizer-variant',choices=['pre_update_kl','post_update_kl'],default='post_update_kl')
    p.add_argument('--exploration-profile',choices=['standard','mixed','refine','gentle'],default='standard')
    a=p.parse_args()
    if a.num_envs<12 or (a.num_envs*a.horizon)%256:
        p.error("At least 12 environments and rollout size divisible by 256 required")
    recovery=a.task_version!='legacy'
    b2=a.task_version in ('b2_payload_v1','b2_payload_v2','b2_payload_v3','b2_payload_v4','b2_blind_v1','b2_blind_ascent_v1','b2_blind_centered_v1','b2_blind_odometry_v1','b2_blind_lateral_v1')
    symmetric=a.task_version=='b2_payload_v4'
    blind=a.task_version in ('b2_blind_v1','b2_blind_ascent_v1','b2_blind_centered_v1','b2_blind_odometry_v1','b2_blind_lateral_v1')
    ascent_only=a.task_version in ('b2_blind_ascent_v1','b2_blind_centered_v1','b2_blind_odometry_v1','b2_blind_lateral_v1')
    required_tasks=(0,3) if ascent_only else (0,1,2,3)
    if a.task_version=='b2_blind_v1' and a.warm_start:p.error('Blind training starts from scratch; use --resume only for an existing blind run')
    low_noise_required=a.exploration_profile in ('mixed','refine','gentle')
    if low_noise_required and not b2:p.error('Mixed exploration is currently supported for B2 tasks only')
    if a.actor_lr<=0:p.error('Actor learning rate must be positive')
    env_type={'legacy':TaishanEnv,'recovery_v2':RecoveryEnv,'recovery_v3':BoundedRecoveryEnv,'b2_payload_v1':B2PayloadEnv,'b2_payload_v2':B2SwingEnv,'b2_payload_v3':B2PostureEnv,'b2_payload_v4':B2SymmetricEnv,'b2_blind_v1':B2BlindEnv,'b2_blind_ascent_v1':B2BlindAscentEnv,'b2_blind_centered_v1':B2BlindCenteredEnv,'b2_blind_odometry_v1':B2BlindOdometryEnv,'b2_blind_lateral_v1':B2BlindLateralEnv}[a.task_version]
    heights=tuple(s[0] for s in B2_STAGES) if b2 else (RECOVERY_HEIGHTS if recovery else HEIGHTS)
    target_stage=len(heights)-1 if a.target_stage<0 else a.target_stage
    if not 0<=target_stage<len(heights):p.error('Invalid target stage')
    if a.start_stage>target_stage:p.error('Initial stage exceeds target stage')
    if not 0<=a.start_stage<len(heights):p.error('Invalid initial stage')
    if a.resume and a.warm_start:p.error('Warm start creates a new run; do not combine with resume')
    if a.resize_envs and not a.resume:p.error('--resize-envs requires --resume')
    root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    (root/"checkpoints").mkdir(exist_ok=True)
    project=Path(__file__).resolve().parents[1]
    dynamics_hashes={name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in
                    ('stairs/b2_env.py','stairs/b2_scene.py','configs/b2_payload_training.json','vendor/unitree_b2/b2.xml')} if b2 else {}
    if a.task_version in ('b2_payload_v3','b2_payload_v4','b2_blind_v1','b2_blind_ascent_v1','b2_blind_centered_v1','b2_blind_odometry_v1','b2_blind_lateral_v1'):dynamics_hashes['stairs/b2_posture.py']=hashlib.sha256((project/'stairs/b2_posture.py').read_bytes()).hexdigest()
    if blind:dynamics_hashes['stairs/b2_blind.py']=hashlib.sha256((project/'stairs/b2_blind.py').read_bytes()).hexdigest()
    if ascent_only:dynamics_hashes['stairs/b2_blind_ascent.py']=hashlib.sha256((project/'stairs/b2_blind_ascent.py').read_bytes()).hexdigest()
    if a.task_version=='b2_blind_centered_v1':dynamics_hashes['stairs/b2_blind_centered.py']=hashlib.sha256((project/'stairs/b2_blind_centered.py').read_bytes()).hexdigest()
    if a.task_version in ('b2_blind_odometry_v1','b2_blind_lateral_v1'):dynamics_hashes['stairs/b2_blind_odometry.py']=hashlib.sha256((project/'stairs/b2_blind_odometry.py').read_bytes()).hexdigest()
    if a.task_version=='b2_blind_lateral_v1':dynamics_hashes['stairs/b2_blind_lateral.py']=hashlib.sha256((project/'stairs/b2_blind_lateral.py').read_bytes()).hexdigest()
    if symmetric or blind:dynamics_hashes['stairs/b2_symmetry.py']=hashlib.sha256((project/'stairs/b2_symmetry.py').read_bytes()).hexdigest()
    if jax.default_backend()!="gpu":raise RuntimeError("CUDA GPU required")
    stored=None
    if a.resume:
        with (root/"checkpoints/latest.pkl").open("rb") as f:stored=pickle.load(f)
        for k in ("horizon","width","epochs","seed"):
            if stored["args"][k]!=getattr(a,k):raise ValueError(f"Resume cannot change {k}")
        if stored['args']['num_envs']!=a.num_envs and not a.resize_envs:
            raise ValueError('Changing world count requires explicit --resize-envs')
        if archive_end(root)!=stored["step"]:raise ValueError("Archive ahead of checkpoint; refusing a discontinuous resume")
        if stored.get('task_version','legacy')!=a.task_version:raise ValueError('Resume must retain task version')
        if stored.get('optimizer_variant','pre_update_kl')!=a.optimizer_variant:
            raise ValueError('Optimizer variant changed: use a documented warm start in a new run')
        if stored.get('exploration_profile','standard')!=a.exploration_profile:
            raise ValueError('Exploration profile changed: use a documented warm start in a new run')
        if stored.get('actor_lr',1e-4)!=a.actor_lr:
            raise ValueError('Actor learning rate changed: create a documented warm start in a new run')
        if b2 and stored.get('dynamics_hashes',dynamics_hashes)!=dynamics_hashes:
            raise ValueError('B2 dynamics/config changed: create a documented new run rather than overwrite historical scenes')
    elif (root/"config.json").exists():
        raise ValueError("Existing run: use --resume or a new directory")
    step=stored["step"] if stored else 0
    resizing=bool(stored and stored['args']['num_envs']!=a.num_envs)
    if resizing:
        backup=root/f"checkpoints/before_resize_{stored['args']['num_envs']}_to_{a.num_envs}_at_{step}.pkl"
        if not backup.exists():shutil.copy2(root/'checkpoints/latest.pkl',backup)
    changing_stage=bool(stored and stored.get('pending_stage') is not None)
    resetting_pool=changing_stage or resizing
    stage=stored['pending_stage'] if changing_stage else (stored["stage"] if stored else a.start_stage)
    update=stored["update"] if stored else 0
    stage_start=step if changing_stage else (stored["stage_start"] if stored else 0)
    pending_stage=None
    histories=[deque(maxlen=100) for _ in TASKS]
    low_histories=[deque(maxlen=100) for _ in TASKS]
    if stored and not changing_stage:
        histories=[deque(v,maxlen=100) for v in stored["histories"]]
        low_histories=[deque(v,maxlen=100) for v in stored.get('low_histories',[[] for _ in TASKS])]
    env=env_type(root/f"scenes/stage_{stage:02d}",stage,a.seed)
    net=(SymmetricActorCritic if symmetric else ActorCritic)(width=a.width,minimum_logstd=float(np.log(.10)) if recovery else -3.,
                    maximum_logstd=float(np.log(.35)) if recovery else -.5)
    key=jax.device_put(stored['key']) if resetting_pool else jax.random.PRNGKey(a.seed)
    overrides=jp.where(jp.arange(a.num_envs)<4,jp.arange(a.num_envs),-1)
    key,init=jax.random.split(key)
    state=jax.jit(jax.vmap(env.reset))(jax.random.split(init,a.num_envs),overrides)
    jax.block_until_ready(state.data.qpos)
    key,init=jax.random.split(key)
    params=net.init(init,state.obs["state"])
    if b2:params['params']['logstd']=jp.full(12,np.log(.2),dtype=jp.float32)
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-4 if recovery else 3e-4,eps=1e-5))
    if a.task_version=='recovery_v3' or b2:
        labels={'params':{name:jax.tree.map(lambda _:('critic' if name.startswith('critic_') or name=='value' else 'actor'),value)
                          for name,value in params['params'].items()}}
        tx=optax.multi_transform({
            'actor':optax.chain(optax.clip_by_global_norm(1.),optax.adam(a.actor_lr if b2 else 5e-5,eps=1e-5)),
            'critic':optax.chain(optax.clip_by_global_norm(1.),optax.adam(3e-4,eps=1e-5))},labels)
    ts=TrainState.create(apply_fn=net.apply,params=params,tx=tx)
    size=state.obs["state"].shape[-1]
    norm=(jp.zeros(size),jp.ones(size),jp.array(1e-4))
    if a.warm_start:
        with a.warm_start.open('rb') as f:source=pickle.load(f)
        if b2 and source.get('task_version') not in ('b2_payload_v1','b2_payload_v2','b2_payload_v3','b2_payload_v4','b2_blind_v1','b2_blind_ascent_v1','b2_blind_centered_v1','b2_blind_odometry_v1','b2_blind_lateral_v1'):raise ValueError('B2 cannot inherit a Go1 policy')
        if blind and source.get('task_version') not in ('b2_blind_v1','b2_blind_ascent_v1','b2_blind_centered_v1','b2_blind_odometry_v1','b2_blind_lateral_v1'):
            raise ValueError('Blind policies may inherit only an audited blind policy')
        if source['width']!=a.width or len(source['norm'][0])!=size:raise ValueError('Warm-start observation/network mismatch')
        params=dict(ts.params);layer=dict(params['params'])
        inherited_layers=('actor_0','actor_1','mean','critic_0','critic_1','value') if a.warm_start_mode=='actor-critic' else ('actor_0','actor_1','mean')
        for name in inherited_layers:
            layer[name]=jax.device_put(source['params']['params'][name])
        if a.warm_start_mode=='actor-critic':
            layer['logstd']=jax.device_put(source['params']['params']['logstd'])
        elif a.task_version=='recovery_v3':
            layer['logstd']=jp.clip(jax.device_put(source['params']['params']['logstd']),np.log(.1),np.log(.2))
        else:
            layer['logstd']=jp.full(12,np.log(.16),dtype=jp.float32)
        params['params']=layer;norm=jax.device_put(source['norm'])
        ts=ts.replace(params=params,opt_state=ts.tx.init(params))
        with (root/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(type='explicit_policy_warm_start',mode=a.warm_start_mode,source=str(a.warm_start.resolve()),
                source_sample=source['step'],source_stage=source['stage'],retained=('actor, critic, exploration std and observation normalization' if a.warm_start_mode=='actor-critic' else 'actor and observation normalization'),
                reset=('optimizer, physics pool and new run sample counter' if a.warm_start_mode=='actor-critic' else 'critic, optimizer, exploration, physics pool and new run sample counter'),
                exploration_std=np.exp(np.asarray(layer['logstd'])).tolist(),task_version=a.task_version))+'\n')
    if stored:
        norm=jax.device_put(stored['norm'])
        if not resetting_pool:
            state,key=(jax.device_put(stored[k]) for k in ('state','key'))
        elif resizing and not changing_stage:
            next_episode=int(np.max(np.asarray(stored['state'].info['episode_id'])))+1
            state.info['episode_id']=jp.full(a.num_envs,next_episode,dtype=jp.int32)
        ts=ts.replace(params=jax.device_put(stored["params"]),opt_state=jax.device_put(stored["opt_state"]),step=stored["optimizer_step"])
    collect=collector(env,net,a.horizon,a.num_envs,12,a.exploration_profile)
    optimization=dict(gamma=.995,entropy_cost=.003,stand_prior_cost=.01,project_logstd=True,kl_limit=.025) if recovery else {}
    if recovery and a.optimizer_variant=='post_update_kl':optimization['post_update_kl']=True
    _,optimize=build_functions(env,net,state,a.horizon,256,a.epochs,
        observation_mirror=mirror_observation if symmetric else None,**optimization)
    recorder=SelectedRecorder(root,stage,a.num_envs)
    if stored and not resetting_pool:recorder.restore(stored["recorder_state"])
    if resizing:
        with (root/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(type='explicit_world_pool_resize',step=step,
                old_worlds=stored['args']['num_envs'],new_worlds=a.num_envs,
                retained='actor, critic, optimizer state, normalization, cumulative samples and existing archives',
                reset='physics pool and in-flight episode buffers; episode IDs advanced at the boundary',
                backup=str(backup)))+'\n')
    if changing_stage:
        with (root/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(type='curriculum_pool_reset_after_process_restart',step=step,stage=stage,
                reason='Release compiled scene/physics allocations before loading the next stage'))+'\n')
    config=dict(vars(a));config["run"]=str(root)
    config['warm_start']=str(a.warm_start.resolve()) if a.warm_start else None
    config['learning_rates']={'actor':a.actor_lr if b2 else 5e-5,'critic':3e-4} if a.task_version=='recovery_v3' or b2 else {'all':1e-4 if recovery else 3e-4}
    if b2:
        config.update(robot='B2',payload_curriculum=list(B2_STAGES),target_payload_kg=float(B2_STAGES[target_stage][1]),
            target_height_m=float(heights[target_stage]),
            stand_exploration_scale=.04,
            walking_exploration_scales=({'gentle':[.0625,.125,.1875,.25], 'refine':[.125,.25,.375,.5], 'mixed':[.125,.25,.5,1.]}.get(a.exploration_profile,[1.])),
            low_noise_promotion_required=low_noise_required,
            dynamics_caveats=env.scene_metadata['caveats'],official_model_mass_kg=env.scene_metadata['official_model_mass_kg'],
            payload_observation_scaling='Last observation is known payload / 30, with fixed mean=0 and variance=1')
    if blind:
        config.update(payload_observation_scaling='No payload input', observation_history_frames=25,
            observation_frame_size=env.frame_size, excluded_policy_inputs=env.scene_metadata['excluded_policy_inputs'],
            initialization='Continued from an audited blind actor' if a.warm_start else 'From scratch; no terrain-aware actor warm start',
            trained_tasks=['ascent','stand'] if ascent_only else TASKS)
    if symmetric:
        config.update(symmetry='Exactly mirror-equivariant actor and mirror-invariant critic (stairs/b2_symmetry.py); '
            'observation statistics updated with mirrored observations and symmetrized after every update')
    config.update(recorded_world_ids=[0,1,2,3],extra_candidate_world_ids=list(range(4,12)),
        recording_hz=50,simulation_hz=250,task_names=TASKS,heights=list(heights),observation_size=size,
        policy=('one shared actor for ascent/stand' if ascent_only else 'one shared actor for ascent/descent/traverse/stand') + ('; explicit actor warm start' if a.warm_start else ''),
        optimization=optimization,minimum_std=.1 if recovery else float(np.exp(-3)),
        abnormal_limit="12 complete episodes per reason per stage, sampled from worlds 4..11",
        checkpoint_retention="initial policy, latest+previous full state, best-training policy, stage-completed policies, final policy",
        video_plan="all four fixed worlds, original 50 fps and 50x overview after training")
    (root/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    source_dir=root/"source";source_dir.mkdir(exist_ok=True)
    project=Path(__file__).resolve().parents[1]
    if stored:
        session=root/f'sessions/at_{step}_stage_{stage:02d}'
        session.mkdir(parents=True,exist_ok=True)
        for name in ('training_finished.json','training_exit_code.txt','evaluation_exit_code.txt','pipeline_finished.txt'):
            path=root/name
            if path.exists():path.replace(session/name)
        for name in ('stairs/train.py','stairs/train_taishan.py','scripts/run_b2_pipeline.sh'):
            target=session/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(project/name,target)
    if not stored:
        hashes={}
        for name in ("stairs/taishan.py","stairs/taishan_env.py","stairs/train_taishan.py","stairs/train.py",
                     "stairs/exploration.py",
                     "stairs/selected_recording.py","stairs/recording.py","stairs/taishan_recovery.py","scripts/render_selected_history.py",
                     "scripts/evaluate_taishan.py","scripts/run_taishan_pipeline.sh",
                     "stairs/b2_scene.py","stairs/b2_env.py","configs/b2_payload_training.json","configs/payload_target.json",
                     "stairs/b2_posture.py","stairs/b2_symmetry.py","stairs/b2_blind.py","stairs/b2_blind_ascent.py","stairs/b2_blind_centered.py","stairs/b2_blind_odometry.py","stairs/b2_blind_lateral.py","scripts/build_studio_stairs.py","scripts/prepare_blind_odometry.py",
                     "vendor/unitree_b2/b2.xml","vendor/unitree_b2/LICENSE","vendor/unitree_b2/source.json",
                     "scripts/run_b2_pipeline.sh","scripts/run_blind_studio_pipeline.sh","tests/test_blind_lateral.py","scripts/postprocess_taishan_run.sh","scripts/check_b2_payload.py"):
            path=project/name
            if path.exists():
                target=source_dir/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        (source_dir/"sha256.json").write_text(json.dumps(hashes,indent=2))
        shutil.copy2(project/"configs/requirements-lock.txt",source_dir/"requirements-lock.txt")
    stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    last_save=step;started=time.monotonic();best=stored.get("best",-1.) if stored else -1.
    reason="budget_reached"
    rates={name:0. for name in TASKS};rawrates=dict(rates);low_rates=dict(rates)
    def policy(name):
        atomic_pickle(root/f"checkpoints/{name}.pkl",dict(params=ts.params,norm=norm,width=a.width,step=step,stage=stage,seed=a.seed,
            exploration_profile=a.exploration_profile,
            actor_lr=a.actor_lr,
            task_version=a.task_version,target_stage=target_stage,
            minimum_logstd=net.minimum_logstd,maximum_logstd=net.maximum_logstd))
    def save():
        payload=dict(params=ts.params,opt_state=ts.opt_state,optimizer_step=ts.step,norm=norm,
            state=state,key=key,step=step,stage=stage,update=update,stage_start=stage_start,
            args={k:getattr(a,k) for k in ("num_envs","horizon","width","epochs","seed")},
            histories=[list(h) for h in histories],best=best,recorder_state=recorder.state_dict())
        payload['task_version']=a.task_version
        payload['optimizer_variant']=a.optimizer_variant
        payload['exploration_profile']=a.exploration_profile
        payload['actor_lr']=a.actor_lr
        payload['low_histories']=[list(h) for h in low_histories]
        payload['pending_stage']=pending_stage
        payload['dynamics_hashes']=dynamics_hashes
        latest=root/"checkpoints/latest.pkl"
        atomic_pickle(root/"checkpoints/pending.pkl",payload)
        if latest.exists():latest.replace(root/"checkpoints/previous.pkl")
        (root/"checkpoints/pending.pkl").replace(latest)
    if not stored:policy("initial")
    save()
    initial_path=root/f"scenes/stage_{stage:02d}/initial_state.npz"
    if resizing:initial_path=initial_path.with_name(f'pool_reset_at_{step}_worlds{a.num_envs}.npz')
    if not initial_path.exists():
        np.savez(initial_path,qpos=np.asarray(state.data.qpos[:4]),qvel=np.asarray(state.data.qvel[:4]))
    print("READY",json.dumps(dict(step=step,stage=stage,worlds=a.num_envs,obs=size,devices=[str(d) for d in jax.devices()])),flush=True)
    try:
        while step<a.steps and not stop[0]:
            if shutil.disk_usage(root).free<6*1024**3:
                reason="disk_reserve";break
            t=time.monotonic()
            (nextstate,nextkey),batch=collect(state,key,ts.params,norm)
            host=jax.device_get(dict(record=batch["record"],summary=batch["summary"]))
            entry=recorder.append(host["record"],step,update)
            state,key=nextstate,nextkey;step=entry["end_control_step"]
            summary=host["summary"]
            if not np.all(summary["finite"]) or np.max(summary["max_constraints"])>=512:
                raise RuntimeError("Invalid physics or constraint capacity reached")
            ts,norm,key,loss=optimize(ts,norm,batch,key)
            if b2 and not blind:
                # Known mass is already scaled to 0..1. Keep it well conditioned when leaving the unloaded stage.
                norm=(norm[0].at[-1].set(0.),norm[1].at[-1].set(1.),norm[2])
            if symmetric:norm=symmetric_norm(norm)
            loss=np.asarray(loss)
            if not np.isfinite(loss).all():raise FloatingPointError("Nonfinite PPO update")
            update+=1
            for t0,w0 in zip(*np.where(summary["done"])):
                if summary['partial'][t0,w0]:continue
                n=max(int(summary["climb_frames"][t0,w0]),1)
                foot=summary["contact_count"][t0,w0]/n
                flight=float(summary["airborne_count"][t0,w0]/n)
                gait=bool(foot.min()>=.15 and flight<=.10 and summary["max_flight"][t0,w0]<=10)
                task=int(summary["task"][t0,w0])
                episode=dict(success=bool(summary["success"][t0,w0]),
                    gait=gait if task!=3 else True,return_=float(summary["episode_return"][t0,w0]),
                    contact_fraction=foot.tolist(),airborne_fraction=flight)
                histories[task].append(episode)
                if summary['low_noise_group'][t0,w0]:low_histories[task].append(episode)
            rates={TASKS[i]:float(np.mean([r["success"] and r["gait"] for r in h])) if h else 0. for i,h in enumerate(histories)}
            rawrates={TASKS[i]:float(np.mean([r["success"] for r in h])) if h else 0. for i,h in enumerate(histories)}
            low_rates={TASKS[i]:float(np.mean([r['success'] and r['gait'] for r in h])) if h else 0. for i,h in enumerate(low_histories)}
            metric=dict(step=step,update=update,stage=stage,height=env.height,num_envs=a.num_envs,
                payload_kg=getattr(env,'payload_kg',0.),robot='B2' if b2 else 'Go1',
                sps=a.num_envs*a.horizon/(time.monotonic()-t),elapsed_s=time.monotonic()-started,
                success_with_gait=rates,task_success=rawrates,recent_episodes={TASKS[i]:len(h) for i,h in enumerate(histories)},
                low_noise_success=low_rates,low_noise_episodes={TASKS[i]:len(h) for i,h in enumerate(low_histories)},
                reward=float(np.asarray(batch["reward"]).mean()),policy_loss=float(loss[0]),value_loss=float(loss[1]),entropy=float(loss[2]),
                approximate_kl=float(loss[3]),exploration_std=np.exp(np.asarray(ts.params['params']['logstd'])).tolist(),
                actor_step_scale=float(loss[4]) if len(loss)>4 else 1.,
                max_constraints=int(np.max(summary["max_constraints"])),recorded_worlds=4,recorded_through=step,
                global_contact_fraction=summary["foot_contact"].mean(axis=(0,1)).tolist(),
                global_airborne_fraction=float((~summary["foot_contact"].any(-1)).mean()),
                reward_parts=dict(zip(env.reward_keys,summary["reward_parts_mean"].mean(0).tolist())),
                ended_episodes=int(summary["done"].sum()),failure_counts={str(i):int(((summary["reason"]==i)&summary["done"]).sum()) for i in (1,2,3,4,8,9)})
            with (root/"metrics.jsonl").open("a") as f:f.write(json.dumps(metric)+"\n")
            if update%10==0:
                print(json.dumps(metric),flush=True)
                (root/"status.json").write_text(json.dumps(metric,indent=2))
            motion_score=min(rates['ascent'],rates['descent'],rates['traverse']) if recovery else min(rates['ascent'],rates['descent'])
            if low_noise_required:motion_score=min(motion_score,low_rates['ascent'],low_rates['descent'],low_rates['traverse'])
            if ascent_only:
                motion_score=min(rates['ascent'],low_rates['ascent']) if low_noise_required else rates['ascent']
            score=stage+motion_score
            if score>best and (motion_score>=.25 or not recovery) and min(len(histories[i]) for i in required_tasks)>=50:
                best=score;policy("best_training")
            if step-last_save>=a.checkpoint_every:
                save();last_save=step
            promotion=(min(rates['ascent'],rates['descent'])>=.75 and rates['traverse']>=.65 and rates['stand']>=.8
                       and min(len(h) for h in histories)>=50) if recovery else (min(rates['ascent'],rates['descent'])>=.7)
            if low_noise_required:
                promotion=promotion and min(low_rates['ascent'],low_rates['descent'])>=.7 and low_rates['traverse']>=.6 and low_rates['stand']>=.8 and min(len(h) for h in low_histories)>=50
            if ascent_only:
                promotion=(rates['ascent']>=.8 and rates['stand']>=.8
                    and min(len(histories[i]) for i in required_tasks)>=50)
                if low_noise_required:
                    promotion=promotion and low_rates['ascent']>=.8 and low_rates['stand']>=.8
                    promotion=promotion and min(len(low_histories[i]) for i in required_tasks)>=50
            if stage==target_stage and step-stage_start>=2_000_000 and promotion:
                policy(f"stage_{stage:02d}_completed")
                reason='target_stage_mastered'
                break
            if stage<target_stage and step-stage_start>=2_000_000 and min(len(histories[i]) for i in required_tasks)>=50 and promotion:
                policy(f"stage_{stage:02d}_completed")
                if b2:
                    pending_stage=stage+1
                    reason='curriculum_restart'
                    with (root/'events.jsonl').open('a') as f:
                        f.write(json.dumps(dict(type='curriculum_pending_process_restart',step=step,
                            completed_stage=stage,next_stage=pending_stage,old_rates=rates))+'\n')
                    break
                stage+=1;stage_start=step
                with (root/"events.jsonl").open("a") as f:f.write(json.dumps(dict(type="curriculum_pool_reset",step=step,stage=stage,old_rates=rates))+"\n")
                env=env_type(root/f"scenes/stage_{stage:02d}",stage,a.seed)
                key,resetkey=jax.random.split(key)
                state=jax.jit(jax.vmap(env.reset))(jax.random.split(resetkey,a.num_envs),overrides)
                collect=collector(env,net,a.horizon,a.num_envs,12,a.exploration_profile)
                _,optimize=build_functions(env,net,state,a.horizon,256,a.epochs,
                    observation_mirror=mirror_observation if symmetric else None,**optimization)
                recorder=SelectedRecorder(root,stage,a.num_envs)
                histories=[deque(maxlen=100) for _ in TASKS]
                low_histories=[deque(maxlen=100) for _ in TASKS]
                np.savez(root/f"scenes/stage_{stage:02d}/initial_state.npz",qpos=np.asarray(state.data.qpos[:4]),qvel=np.asarray(state.data.qvel[:4]))
                save();last_save=step
            if recovery and step-stage_start>=a.max_stage_steps:
                reason='stage_stall' if motion_score<.15 else 'stage_budget_exhausted'
                break
        if stop[0]:reason="signal_stop"
    except BaseException:
        reason="error";raise
    finally:
        save();policy("final")
        result=dict(state=reason,step=step,stage=stage,height=env.height,recorded_through=archive_end(root),
                    payload_kg=getattr(env,'payload_kg',0.),robot='B2' if b2 else 'Go1',
                    task_success=rawrates if update else {},success_with_gait=rates if update else {},time=time.time())
        result['low_noise_success']=low_rates
        (root/"training_finished.json").write_text(json.dumps(result,indent=2))
        print("SAVED",json.dumps(result),flush=True)


if __name__=="__main__":main()
