"""Remember lateral travel using existing onboard estimates and command signals."""
import copy
import json

import jax.numpy as jp
import numpy as np

from stairs.b2_blind_odometry import B2BlindOdometryEnv


def route_lateral_velocity(body_velocity, gravity, yaw_rate_command):
    """Project estimated body velocity into the commanded route's lateral axis.

    The existing heading servo sends yaw_rate=clip(1.5*heading_error, +/-.6).
    Its command already tells the actor this bounded heading error. No simulator
    heading or position is read here. Large heading errors saturate this estimate.
    """
    gravity = gravity / jp.maximum(jp.linalg.norm(gravity), 1e-5)
    pitch = jp.arcsin(jp.clip(gravity[0], -1., 1.))
    roll = jp.arctan2(-gravity[1], -gravity[2])
    yaw = -yaw_rate_command / 1.5
    sy, cy = jp.sin(yaw), jp.cos(yaw)
    sp, cp = jp.sin(pitch), jp.cos(pitch)
    sr, cr = jp.sin(roll), jp.cos(roll)
    lateral_axis = jp.array([sy * cp, sy * sp * sr + cy * cr,
                             sy * sp * cr - cy * sr])
    return jp.dot(lateral_axis, body_velocity)


def integrate_lateral(previous, body_velocity, gravity, yaw_rate_command, dt=.02):
    # This is travel relative to the initial pose, not absolute path position.
    # The origin has the same unknown +/-4 cm start offset as the existing task.
    return jp.clip(previous + dt * route_lateral_velocity(body_velocity, gravity, yaw_rate_command), -2., 2.)


class B2BlindLateralEnv(B2BlindOdometryEnv):
    frame_size = 53
    frame_fields = dict(B2BlindOdometryEnv.frame_fields, estimated_lateral_travel=[52, 53])

    def __init__(self, directory, stage=0, seed=20261030):
        super().__init__(directory, stage, seed)
        self.scene_metadata.update(
            task_version='b2_blind_lateral_v1', observation_size=25 * 53,
            observation='Original 52 onboard/command features plus accumulated estimated lateral travel in every history frame',
            observation_frame_fields=self.frame_fields,
            lateral_memory='Integrate stance-foot estimated body velocity projected using noisy gravity and the existing heading-servo command; zero origin on reset, +/-2 m clamp',
            lateral_memory_limitations='Dead reckoning drifts under slip and sensor bias; unknown initial position offset; heading estimate saturates with the existing command',
            navigation_revision='Original navigation command and reward unchanged; no world-position centering controller')
        (self.directory / 'metadata.json').write_text(json.dumps(self.scene_metadata, indent=2) + '\n')

    def _obs(self, data, info):
        original = super()._obs(data, info)['state'].reshape(self.history_frames, 52)
        sensors = info['proprio_history'][-1]
        previous = info.get('estimated_lateral_travel', jp.array(0.))
        displacement = integrate_lateral(previous, info['estimated_body_velocity'],
                                          sensors[3:6], sensors[48], self.dt)
        info['estimated_lateral_travel'] = displacement
        if 'lateral_history' in info:
            history = jp.concatenate((info['lateral_history'][1:], displacement[None, None]), axis=0)
        else:
            history = jp.full((self.history_frames, 1), displacement)
        info['lateral_history'] = history
        return {'state': jp.concatenate((original, history), axis=-1).reshape(-1)}


def expand_odometry_checkpoint(source):
    """Append zero-weight memory features; preserve the original policy function."""
    if source.get('task_version') != 'b2_blind_odometry_v1' or len(source['norm'][0]) != 25 * 52:
        raise ValueError('Expected an audited 25 x 52 blind-odometry policy')
    result = copy.deepcopy(source)
    for name in ('actor_0', 'critic_0'):
        kernel = np.asarray(source['params']['params'][name]['kernel'])
        expanded = np.zeros((25, 53, kernel.shape[-1]), dtype=kernel.dtype)
        expanded[:, :52] = kernel.reshape(25, 52, -1)
        result['params']['params'][name]['kernel'] = expanded.reshape(25 * 53, -1)
    mean, variance, count = source['norm']
    expanded_mean = np.zeros((25, 53), dtype=np.asarray(mean).dtype)
    expanded_variance = np.full((25, 53), .25, dtype=np.asarray(variance).dtype)
    expanded_mean[:, :52] = np.asarray(mean).reshape(25, 52)
    expanded_variance[:, :52] = np.asarray(variance).reshape(25, 52)
    result['norm'] = (expanded_mean.reshape(-1), expanded_variance.reshape(-1), count)
    result['task_version'] = 'b2_blind_lateral_v1'
    result['observation_migration'] = dict(source_version=source['task_version'],
        source_step=source['step'], source_stage=source['stage'],
        method='Preserve old weights/statistics; append one zero-weight estimated-travel feature per frame, variance .25')
    return result
