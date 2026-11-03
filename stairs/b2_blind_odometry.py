"""Blind ascent with stance-foot velocity estimated only from onboard signals.

The estimate assumes a contacting foot is stationary. It is deliberately imperfect
under slipping/impact; simulator position, base velocity and terrain are never read.
"""
import copy
import json

import jax.numpy as jp
import numpy as np

from stairs.b2_blind_ascent import B2BlindAscentEnv
from stairs.b2_scene import FEET


def rotate_x(v, angle):
    c, s = jp.cos(angle), jp.sin(angle)
    return jp.stack((v[..., 0], c * v[..., 1] - s * v[..., 2],
                     s * v[..., 1] + c * v[..., 2]), axis=-1)


def rotate_y(v, angle):
    c, s = jp.cos(angle), jp.sin(angle)
    return jp.stack((c * v[..., 0] + s * v[..., 2], v[..., 1],
                     -s * v[..., 0] + c * v[..., 2]), axis=-1)


def foot_kinematics(joint_position, joint_velocity, offsets):
    """Foot positions and Jacobian*qdot in the base frame, FR/FL/RR/RL order."""
    q, dq = joint_position.reshape(4, 3), joint_velocity.reshape(4, 3)
    hip, thigh_offset, calf_offset, toe_offset = offsets
    thigh = hip + rotate_x(thigh_offset, q[:, 0])
    calf = thigh + rotate_x(rotate_y(calf_offset, q[:, 1]), q[:, 0])
    foot = calf + rotate_x(rotate_y(toe_offset, q[:, 1] + q[:, 2]), q[:, 0])
    hip_axis = jp.broadcast_to(jp.array([1., 0., 0.]), (4, 3))
    pitch_axis = rotate_x(jp.broadcast_to(jp.array([0., 1., 0.]), (4, 3)), q[:, 0])
    velocity = (jp.cross(hip_axis, foot - hip) * dq[:, 0, None]
                + jp.cross(pitch_axis, foot - thigh) * dq[:, 1, None]
                + jp.cross(pitch_axis, foot - calf) * dq[:, 2, None])
    return foot, velocity


def stance_velocity(position, relative_velocity, gyro, contact, previous):
    candidates = -relative_velocity - jp.cross(gyro, position)
    count = jp.sum(contact)
    estimate = jp.sum(candidates * contact[:, None], axis=0) / jp.maximum(count, 1)
    # A contact bit alone cannot distinguish slip from stance. Bound impacts and
    # smooth the estimate; retain the previous estimate when all feet are airborne.
    filtered = .75 * previous + .25 * jp.clip(estimate, -2., 2.)
    return jp.where(count > 0, filtered, previous)


def model_leg_offsets(model):
    """Read fixed link geometry once; reject models the analytic FK cannot express."""
    offsets = []
    for kind in ('hip', 'thigh', 'calf'):
        body_ids = [model.body(f'{foot}_{kind}').id for foot in FEET]
        np.testing.assert_allclose(model.body_quat[body_ids], np.tile([1., 0., 0., 0.], (4, 1)), atol=1e-7)
        axis = [1., 0., 0.] if kind == 'hip' else [0., 1., 0.]
        for foot in FEET:
            joint = model.joint(f'{foot}_{kind}_joint').id
            np.testing.assert_allclose(model.jnt_pos[joint], 0., atol=1e-7)
            np.testing.assert_allclose(model.jnt_axis[joint], axis, atol=1e-7)
            if abs(model.qpos0[model.jnt_qposadr[joint]]) > 1e-7:
                raise ValueError('Analytic FK expects zero joint reference positions')
        offsets.append(jp.asarray(model.body_pos[body_ids]))
    offsets.append(jp.asarray(model.site_pos[[model.site(foot).id for foot in FEET]]))
    return tuple(offsets)


class B2BlindOdometryEnv(B2BlindAscentEnv):
    frame_size = 52
    frame_fields = dict(B2BlindAscentEnv.frame_fields, estimated_body_velocity=[49, 52])

    def __init__(self, directory, stage=0, seed=20261030):
        super().__init__(directory, stage, seed)
        self._leg_offsets = model_leg_offsets(self.mj_model)
        self.scene_metadata.update(
            task_version='b2_blind_odometry_v1', observation_size=25 * 52,
            observation='25 frames of the original 49 onboard signals plus stance-foot estimated body velocity',
            observation_frame_fields=self.frame_fields,
            velocity_estimator='Analytic joint FK/Jacobian, noisy joint encoders/IMU gyro and binary foot contact; stationary stance-foot approximation, 0.25 low-pass coefficient, no-contact hold, +/-2 m/s clamp',
            estimator_limitations='Stance-foot velocity is biased by slip, rolling contact and impact',
            policy_architecture='Ordinary MLP over 25 proprioceptive frames; no symmetry projection')
        (self.directory / 'metadata.json').write_text(json.dumps(self.scene_metadata, indent=2) + '\n')

    def _obs(self, data, info):
        # Parent retains its original 25 x 49 noisy sensor history. The estimator
        # consumes that exact sensor frame, never mjx world positions/velocities.
        super()._obs(data, info)
        sensors = info['proprio_history'][-1]
        position, velocity = foot_kinematics(sensors[6:18] + self._default_pose,
                                            sensors[18:30], self._leg_offsets)
        previous = info.get('estimated_body_velocity', jp.zeros(3))
        estimate = stance_velocity(position, velocity, sensors[:3], sensors[42:46], previous)
        info['estimated_body_velocity'] = estimate
        if 'velocity_history' in info:
            history = jp.concatenate((info['velocity_history'][1:], estimate[None]), axis=0)
        else:
            history = jp.repeat(estimate[None], self.history_frames, axis=0)
        info['velocity_history'] = history
        return {'state': jp.concatenate((info['proprio_history'], history), axis=-1).reshape(-1)}


def expand_blind_checkpoint(source):
    """Zero-pad new feature weights, preserving the previous actor/critic function."""
    if source.get('task_version') not in ('b2_blind_v1', 'b2_blind_ascent_v1'):
        raise ValueError('Only original, audited 49-channel blind policies may be expanded')
    if len(source['norm'][0]) != 25 * 49:
        raise ValueError('Expected a 25 x 49 observation history')
    result = copy.deepcopy(source)
    for name in ('actor_0', 'critic_0'):
        kernel = np.asarray(source['params']['params'][name]['kernel'])
        expanded = np.zeros((25, 52, kernel.shape[-1]), dtype=kernel.dtype)
        expanded[:, :49] = kernel.reshape(25, 49, -1)
        result['params']['params'][name]['kernel'] = expanded.reshape(25 * 52, -1)
    mean, variance, count = source['norm']
    expanded_mean = np.zeros((25, 52), dtype=np.asarray(mean).dtype)
    expanded_variance = np.full((25, 52), .25, dtype=np.asarray(variance).dtype)
    expanded_mean[:, :49] = np.asarray(mean).reshape(25, 49)
    expanded_variance[:, :49] = np.asarray(variance).reshape(25, 49)
    result['norm'] = (expanded_mean.reshape(-1), expanded_variance.reshape(-1), count)
    result['task_version'] = 'b2_blind_odometry_v1'
    result['observation_migration'] = dict(source_version=source['task_version'],
        source_step=source['step'], source_stage=source['stage'],
        method='Preserve all old weights/statistics, append 3 zero-weight features per frame, new feature variance .25')
    return result
