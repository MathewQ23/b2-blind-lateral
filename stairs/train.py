"""JAX/Flax PPO, GPU MJX simulation, lossless recording of every sampled step."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")

import argparse
import collections
import json
import math
from pathlib import Path
import pickle
import shutil
import signal
import time

import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
from stairs.environment import StairsEnv
from stairs.recording import Recorder


class ActorCritic(nn.Module):
    actions: int = 12
    width: int = 128
    minimum_logstd: float = -3.
    maximum_logstd: float = -.5

    @nn.compact
    def __call__(self, obs):
        actor, critic = obs, obs
        for i in range(2):
            actor = nn.elu(nn.Dense(self.width, name=f"actor_{i}")(actor))
            critic = nn.elu(nn.Dense(self.width, name=f"critic_{i}")(critic))
        mean = nn.Dense(self.actions, kernel_init=nn.initializers.orthogonal(.01), name="mean")(actor)
        value = nn.Dense(1, name="value")(critic)[..., 0]
        logstd = self.param("logstd", nn.initializers.constant(-.7), (self.actions,))
        return mean, jnp.clip(logstd, self.minimum_logstd, self.maximum_logstd), value


def logprob(u, mean, logstd):
    return -.5*jnp.sum(jnp.square((u-mean)*jnp.exp(-logstd))
                       +2*logstd+math.log(2*math.pi), axis=-1)


def normalize(obs, norm):
    return jnp.clip((obs-norm[0])/jnp.sqrt(norm[1]+1e-5), -10, 10)


def update_norm(norm, obs):
    x = obs.reshape((-1, obs.shape[-1]))
    mean, var, count = norm
    n = x.shape[0]
    delta = jnp.mean(x,0)-mean
    total = count+n
    newvar = (var*count+jnp.var(x,0)*n+jnp.square(delta)*count*n/total)/total
    return mean+delta*n/total, newvar, total


def atomic_pickle(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(jax.device_get(payload), stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def build_functions(env, net, template, horizon, minibatch, epochs,
                    gamma=.99, entropy_cost=.0005, stand_prior_cost=.05,
                    project_logstd=False, kl_limit=None, post_update_kl=False,
                    observation_mirror=None):
    if post_update_kl and kl_limit is None:
        raise ValueError('Post-update KL protection requires a KL limit')
    step_batch = jax.vmap(env.step)

    @jax.jit
    def collect(state, key, params, norm):
        def one(carry, _):
            state, key = carry
            key, sample = jax.random.split(key)
            rawobs = state.obs["state"]
            x = normalize(rawobs,norm)
            mean, logstd, value = net.apply(params,x)
            # Use less exploration for a zero-speed command. The policy's mean
            # action is still learned; no scripted stopping action is applied.
            stand=jnp.linalg.norm(state.info["command"],axis=-1)<.01
            std_offset=jnp.where(stand,jnp.log(.1),0.)
            logstd=logstd+std_offset[:,None]
            u = mean + jnp.exp(logstd)*jax.random.normal(sample,mean.shape)
            action = jnp.tanh(u)
            nextstate = step_batch(state, action)
            _,_,nextvalue = net.apply(params, normalize(nextstate.obs["state"],norm))
            done = nextstate.done > 0
            terminal = nextstate.info["failure"] | nextstate.info["success"]
            record = dict(qpos=nextstate.data.qpos, qvel=nextstate.data.qvel,
                ctrl=nextstate.data.ctrl, action=action, command=state.info["command"],
                reward=nextstate.reward, done=done, success=nextstate.info["success"],
                failure=nextstate.info["failure"], episode_id=state.info["episode_id"],
                nefc=nextstate.data._impl.nefc,
                episode_step=nextstate.info["episode_steps"], sim_time=nextstate.data.time,
                reward_parts=jnp.stack([nextstate.metrics[k] for k in env.reward_keys],axis=-1))
            batch = dict(x=x, rawobs=rawobs, u=u, logp=logprob(u,mean,logstd),
                std_offset=std_offset,
                value=value, nextvalue=nextvalue, reward=nextstate.reward,
                done=done, terminal=terminal, record=record,
                episode_return=nextstate.info["episode_return"])
            # Record terminal poses BEFORE auto-reset. Each world has explicit episode IDs.
            def reset_leaf(current, initial):
                # Warp has shared/empty buffers without a leading world axis.
                if current.ndim == 0 or current.shape[0] != done.shape[0]:
                    return current
                mask = done.reshape(done.shape+(1,)*(current.ndim-1))
                return jnp.where(mask, initial, current)
            restored = jax.tree.map(reset_leaf, nextstate, template)
            restored.info["rng"] = nextstate.info["rng"]
            restored.info["episode_id"] = state.info["episode_id"] + done.astype(jnp.int32)
            return (restored,key), batch
        return jax.lax.scan(one,(state,key),None,length=horizon)

    @jax.jit
    def optimize(trainstate, norm, batch, key):
        def advantage(carry, row):
            reward, value, nextvalue, done, terminal = row
            delta = reward + gamma*(1-terminal)*nextvalue - value
            gae = delta + gamma*.95*(1-done)*carry
            return gae, gae
        _, adv = jax.lax.scan(advantage, jnp.zeros(batch["reward"].shape[1]),
            (batch["reward"],batch["value"],batch["nextvalue"],
             batch["done"].astype(jnp.float32),batch["terminal"].astype(jnp.float32)), reverse=True)
        target = adv+batch["value"]
        adv = (adv-jnp.mean(adv))/(jnp.std(adv)+1e-8)
        flat = {k: v.reshape((-1,)+v.shape[2:]) for k,v in
                dict(x=batch["x"],u=batch["u"],oldlogp=batch["logp"],
                     std_offset=batch["std_offset"],
                     oldvalue=batch["value"],target=target,adv=adv).items()}
        flat['stand_mask']=batch.get('stand_mask',batch['std_offset']<0).reshape(-1)
        if post_update_kl:
            flat['old_mean']=batch['old_mean'].reshape((-1,batch['old_mean'].shape[-1]))
            flat['old_logstd']=batch['old_logstd'].reshape((-1,batch['old_logstd'].shape[-1]))
        size = flat["adv"].shape[0]
        def loss(params, b):
            mean, logstd, value = net.apply(params,b["x"])
            logstd=logstd+b["std_offset"][:,None]
            ratio = jnp.exp(logprob(b["u"],mean,logstd)-b["oldlogp"])
            policy = -jnp.mean(jnp.minimum(ratio*b["adv"],jnp.clip(ratio,.8,1.2)*b["adv"]))
            clipped = b["oldvalue"]+jnp.clip(value-b["oldvalue"],-.2,.2)
            value_loss = .5*jnp.mean(jnp.maximum(jnp.square(value-b["target"]),jnp.square(clipped-b["target"])))
            entropy = jnp.mean(jnp.sum(logstd+.5*math.log(2*math.pi*math.e),axis=-1))
            stand_prior=stand_prior_cost*jnp.mean(b['stand_mask']*jnp.mean(jnp.square(mean),axis=-1))
            kl=jnp.mean(ratio-1-jnp.log(ratio+1e-8))
            return policy+.5*value_loss-entropy_cost*entropy+stand_prior, jnp.array([policy,value_loss,entropy,kl])
        def epoch(carry,_):
            ts,key=carry
            key,permkey=jax.random.split(key)
            perm=jax.random.permutation(permkey,size).reshape((-1,minibatch))
            def update(ts,idx):
                b=jax.tree.map(lambda x:x[idx],flat)
                (_,metrics),grads=jax.value_and_grad(loss,has_aux=True)(ts.params,b)
                following=ts.apply_gradients(grads=grads)
                if project_logstd:
                    # Project the parameter itself, so clipping cannot leave a dead gradient outside the bounds.
                    params=dict(following.params)
                    layer=dict(params["params"])
                    layer["logstd"]=jnp.clip(layer["logstd"],net.minimum_logstd,net.maximum_logstd)
                    params["params"]=layer
                    following=following.replace(params=params)
                if post_update_kl:
                    # Check the proposed actor update, not only the pre-update likelihood.
                    # Backtrack actor parameters while retaining the independent critic update.
                    old_layers=ts.params['params'];new_layers=following.params['params']
                    def blended(scale):
                        layers={}
                        for name,new in new_layers.items():
                            if name.startswith('actor_') or name in ('mean','logstd'):
                                layers[name]=jax.tree.map(lambda a,b:a+scale*(b-a),old_layers[name],new)
                            else:
                                layers[name]=new
                        return {**following.params,'params':layers}
                    def analytic_kl(params):
                        mean,logstd,_=net.apply(params,b['x'])
                        logstd=logstd+b['std_offset'][:,None]
                        value=logstd-b['old_logstd']+.5*(
                            jnp.exp(2*(b['old_logstd']-logstd))+
                            jnp.square(b['old_mean']-mean)*jnp.exp(-2*logstd)-1.)
                        return jnp.maximum(jnp.mean(jnp.sum(value,axis=-1)),0.)
                    proposed_kl=analytic_kl(following.params)
                    def needs_reduction(carry):
                        _,kl,count=carry
                        return ((~jnp.isfinite(kl))|(kl>kl_limit))&(count<8)
                    def reduce_step(carry):
                        scale,_,count=carry;scale=scale*.5
                        return scale,analytic_kl(blended(scale)),count+1
                    scale,final_kl,_=jax.lax.while_loop(needs_reduction,reduce_step,
                        (jnp.array(1.),proposed_kl,jnp.array(0)))
                    scale=jnp.where(jnp.isfinite(final_kl)&(final_kl<=kl_limit),scale,0.)
                    params=blended(scale)
                    following=following.replace(params=params)
                    metrics=jnp.concatenate((metrics[:3],jnp.array([analytic_kl(params),scale])))
                elif kl_limit is not None:
                    following=jax.lax.cond(metrics[3]<kl_limit,lambda _:following,lambda _:ts,None)
                return following,metrics
            ts,metrics=jax.lax.scan(update,ts,perm)
            return (ts,key),jnp.mean(metrics,0)
        (ts,key),metrics=jax.lax.scan(epoch,(trainstate,key),None,length=epochs)
        rawobs=batch["rawobs"]
        if observation_mirror is not None:
            # Mirrored samples keep the statistics symmetric, so the mirror commutes with normalization.
            rawobs=jnp.concatenate((rawobs,observation_mirror(rawobs)))
        return ts,update_norm(norm,rawobs),key,jnp.mean(metrics,0)
    return collect,optimize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--num-envs",type=int,default=64)
    parser.add_argument("--horizon",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=4)
    parser.add_argument("--width",type=int,default=128)
    parser.add_argument("--steps",type=int,default=50_000_000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--resize-envs",action="store_true",
        help="Explicitly rebuild the world pool on resume, preserving policy/optimizer and all history")
    parser.add_argument("--initial-height",type=float,default=0.)
    parser.add_argument("--impl",choices=["jax","warp"],default="warp")
    args=parser.parse_args()
    root=Path(args.run).resolve()
    root.mkdir(parents=True,exist_ok=True)
    (root/"checkpoints").mkdir(exist_ok=True)
    if (root/"metrics.jsonl").exists() and not args.resume:
        raise SystemExit("Run already exists; choose a new run or explicitly --resume")
    if jax.default_backend()!="gpu":
        raise RuntimeError("CUDA GPU is required")
    print("DEVICES",jax.devices(),flush=True)
    metadata=dict(vars(args),devices=[str(d) for d in jax.devices()],
        recording="all environments, every control step, pre-reset terminal states",
        entropy_cost=0.0005,maximum_logstd=-0.5,
        stop_exploration_scale=0.1,zero_command_neutral_pose_prior=0.05,
        controls_hz=50,stair_count=6,target_height=.10)
    key=jax.random.PRNGKey(args.seed)
    height=args.initial_height
    step=update=stage=0
    stored=None
    if args.resume:
        with (root/"checkpoints/latest.pkl").open("rb") as f: stored=pickle.load(f)
        if stored["width"] != args.width:
            raise ValueError("Resume must retain the checkpoint's network width")
        if stored["num_envs"] != args.num_envs and not args.resize_envs:
            raise ValueError("World-count changes require explicit --resize-envs")
        height,step,update,stage=(stored[k] for k in ("height","step","update","stage"))
    resizing=bool(stored and stored["num_envs"]!=args.num_envs)
    if resizing:
        key=jax.device_put(stored["key"])
        backup=root/f"checkpoints/before_resize_{stored['num_envs']}_to_{args.num_envs}_at_{step}.pkl"
        if not backup.exists():
            shutil.copy2(root/"checkpoints/latest.pkl",backup)
    net=ActorCritic(width=args.width)
    env=StairsEnv(root/f"scenes/stage_{stage:02d}",height,impl=args.impl)
    key,initkey=jax.random.split(key)
    template=jax.jit(jax.vmap(env.reset))(jax.random.split(initkey,args.num_envs))
    jax.block_until_ready(template.data.qpos)
    state=template
    fresh_template=template
    obs=state.obs["state"]
    key,initkey=jax.random.split(key)
    params=net.init(initkey,obs)
    ts=TrainState.create(apply_fn=net.apply,params=params,
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(3e-4,eps=1e-5)))
    norm=(jnp.zeros(obs.shape[-1]),jnp.ones(obs.shape[-1]),jnp.array(1e-4))
    stage_start=step
    if stored:
        ts=ts.replace(params=jax.device_put(stored["params"]),
            opt_state=jax.device_put(stored["opt_state"]),step=stored["optimizer_step"])
        norm=jax.device_put(stored["norm"])
        if resizing:
            # A different world count needs fresh physics buffers. No old samples
            # are discarded; the reset boundary is explicit in the event log.
            next_episode=int(np.max(stored["state"].info["episode_id"]))+1
            template.info["episode_id"]=jnp.full(args.num_envs,next_episode,jnp.int32)
            state=template
        else:
            key,state,template=(jax.device_put(stored[k]) for k in ("key","state","template"))
            # Versioned task additions may introduce counters without changing
            # the policy's observation shape or the simulator's physical state.
            for name,value in fresh_template.info.items():
                if name not in state.info:
                    state.info[name]=value
                if name not in template.info:
                    template.info[name]=value
        stage_start=stored["stage_start"]
    minibatch=min(256,args.num_envs*args.horizon)
    collect,optimize=build_functions(env,net,template,args.horizon,minibatch,args.epochs)
    recorder=Recorder(root/f"recordings/stage_{stage:02d}",args.num_envs,env.dt,stage)
    initial_path=root/f"scenes/stage_{stage:02d}/initial_state.npz"
    if resizing:
        initial_path=initial_path.with_name(f"pool_reset_at_{step}_worlds{args.num_envs}.npz")
    np.savez(initial_path,
             qpos=np.asarray(template.data.qpos),qvel=np.asarray(template.data.qvel))
    metadata.update(resume_sample_step=step,resized_from=stored["num_envs"] if resizing else None)
    (root/"sessions").mkdir(exist_ok=True)
    (root/f"sessions/at_{step}_worlds{args.num_envs}.json").write_text(json.dumps(metadata,indent=2))
    (root/"config.json").write_text(json.dumps(metadata,indent=2))
    stop=[False]
    def request_stop(*_): stop[0]=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    history=collections.deque(maxlen=100)
    started=time.monotonic()
    last_save=step
    def save(name="latest"):
        atomic_pickle(root/f"checkpoints/{name}.pkl",dict(
            params=ts.params,opt_state=ts.opt_state,optimizer_step=ts.step,
            norm=norm,key=key,state=state,template=template,step=step,update=update,
            stage=stage,height=height,stage_start=stage_start,
            width=args.width,num_envs=args.num_envs,horizon=args.horizon,seed=args.seed))
    save("latest")
    if resizing:
        event=dict(type="world_pool_resize",sample_step=step,old_worlds=stored["num_envs"],
            new_worlds=args.num_envs,time=time.time(),initial_state=str(initial_path),
            retained="policy/value parameters, optimizer state, normalization, cumulative samples and all archives",
            reset="physics world pool; explicitly recorded episode boundary")
        with (root/"events.jsonl").open("a") as f:f.write(json.dumps(event)+"\n")
        print("RESIZED",json.dumps(event),flush=True)
    if step==0 and not args.resume:
        save("initial")
    print(f"READY obs={obs.shape} height={height} start_step={step}",flush=True)
    try:
        while step<args.steps and not stop[0]:
            if shutil.disk_usage(root).free<3*1024**3:
                print("STOP: 3GB disk reserve reached; full history preserved",flush=True)
                break
            t=time.monotonic()
            (state,key),batch=collect(state,key,ts.params,norm)
            host=jax.device_get(batch)
            entry=recorder.append(host["record"],step,update)
            step=entry["end_control_step"]
            if np.max(host["record"]["nefc"]) >= env._config.njmax:
                raise RuntimeError("Constraint capacity reached; history preserved, fix physics capacity before continuing")
            done=host["done"].astype(bool)
            for success in host["record"]["success"][done]: history.append(bool(success))
            ts,norm,key,losses=optimize(ts,norm,batch,key)
            losses=np.asarray(losses)
            if not np.isfinite(losses).all(): raise FloatingPointError(f"Nonfinite PPO loss: {losses}")
            update+=1
            success_rate=float(np.mean(history)) if history else 0.
            metric=dict(step=step,update=update,stage=stage,step_height=height,
                num_envs=args.num_envs,horizon=args.horizon,
                elapsed_s=time.monotonic()-started,sps=args.num_envs*args.horizon/(time.monotonic()-t),
                reward=float(np.mean(host["reward"])),success_rate=success_rate,
                finished_episodes=int(done.sum()),mean_x=float(np.mean(host["record"]["qpos"][...,0])),
                policy_loss=float(losses[0]),value_loss=float(losses[1]),entropy=float(losses[2]),
                recording_bytes=entry["bytes"],recorded_through=step)
            with (root/"metrics.jsonl").open("a") as f: f.write(json.dumps(metric)+"\n")
            if update%10==0:
                print(json.dumps(metric),flush=True)
                (root/"status.json").write_text(json.dumps(metric,indent=2))
            if step-last_save>=250_000:
                save(); last_save=step
                atomic_pickle(root/f"checkpoints/policy_{step:010d}.pkl",
                    dict(params=ts.params,norm=norm,width=args.width,step=step,height=height,stage=stage))
            if step-stage_start>=500_000 and len(history)>=50 and success_rate>=.7 and height<.10:
                save(f"stage_{stage:02d}_completed")
                stage+=1
                height=round(min(height+.02,.10),3)
                print(f"CURRICULUM stage={stage} height={height}",flush=True)
                env=StairsEnv(root/f"scenes/stage_{stage:02d}",height,impl=args.impl)
                key,resetkey=jax.random.split(key)
                template=jax.jit(jax.vmap(env.reset))(jax.random.split(resetkey,args.num_envs))
                jax.block_until_ready(template.data.qpos)
                state=template
                collect,optimize=build_functions(env,net,template,args.horizon,minibatch,args.epochs)
                recorder=Recorder(root/f"recordings/stage_{stage:02d}",args.num_envs,env.dt,stage)
                np.savez(root/f"scenes/stage_{stage:02d}/initial_state.npz",
                         qpos=np.asarray(template.data.qpos),qvel=np.asarray(template.data.qvel))
                stage_start=step
                history.clear()
                save()
    finally:
        save()
        print(f"SAVED step={step} stage={stage} height={height} run={root}",flush=True)


if __name__=="__main__": main()
