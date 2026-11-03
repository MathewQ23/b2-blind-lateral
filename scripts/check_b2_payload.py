#!/usr/bin/env python3
"""Native/GPU B2 full-payload standing and observation/PD mapping smoke test."""
import os
os.environ.setdefault('JAX_PLATFORMS','cuda')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jp
import numpy as np
from stairs.b2_env import B2PayloadEnv
from stairs.b2_scene import STAGES

root=Path(__file__).resolve().parents[2]/'data/go1_stairs/b2_validation'
env=B2PayloadEnv(root/'gpu_full_payload',len(STAGES)-1)
tasks=jp.arange(4)
state=jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(831),4),tasks)
@jax.jit
def run(state):
    def tick(s,_):
        s=jax.vmap(env.step)(s,jp.zeros((4,12)))
        return s,(s.data.qpos,s.data.qvel,s.info['last_contact'],s.data.ctrl,s.data._impl.nefc,s.reward)
    return jax.lax.scan(tick,state,None,length=150)
state,trace=jax.device_get(run(state))
qpos,qvel,contacts,torque,nefc,reward=trace
ground=np.asarray(env.terrain(state.data.qpos[:,0]))
report=dict(robot='B2',payload_kg=env.payload_kg,model_mass_kg=env.scene_metadata['official_model_mass_kg'],
    total_mass_kg=float(env.mj_model.body_mass.sum()),observation_shape=list(state.obs['state'].shape),
    all_fields_finite=bool(np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(reward).all()),
    final_foot_contact=contacts[-1].tolist(),body_clearance_m=(qpos[-1,:,2]-ground).tolist(),
    max_torque_Nm=float(np.abs(torque).max()),max_constraints=int(nefc.max()),
    standing_task_success=bool(state.info['success'][3]),simulated_seconds_per_world=3.)
report['passed']=bool(report['all_fields_finite'] and np.all(contacts[-1]) and min(report['body_clearance_m'])>.4
                      and report['standing_task_success'] and np.all(np.abs(torque)<=np.asarray(env._torque_limits)+1e-4))
(root/'gpu_full_payload.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if not report['passed']:raise SystemExit(1)
