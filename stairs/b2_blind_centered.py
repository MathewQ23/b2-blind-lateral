"""Blind foothold control with explicit route-centering commands from the navigator."""
import json
import jax.numpy as jp
from stairs.b2_blind_ascent import B2BlindAscentEnv


def centered_command(command, nominal_heading, direction, lateral_position, yaw):
    moving = jp.abs(command[0]) > .01
    correction = jp.clip(-.35 * direction * lateral_position, -.3, .3) * moving
    desired_heading = nominal_heading + correction
    error = jp.arctan2(jp.sin(desired_heading - yaw), jp.cos(desired_heading - yaw))
    return command.at[2].set(jp.clip(error * 1.5, -.6, .6)), error


class B2BlindCenteredEnv(B2BlindAscentEnv):
    def __init__(self, directory, stage=0, seed=20261030):
        super().__init__(directory, stage, seed)
        self.scene_metadata.update(
            task_version='b2_blind_centered_v1',
            navigation='Simulator route planner uses lateral offset and heading to issue yaw-rate commands; policy receives commands, not terrain heights',
            navigation_caveat='Blind foothold control with external route guidance, not proprioceptive-only autonomous navigation',
            centering_gain=.35, maximum_heading_correction_rad=.3,
            reward_revision='Original physical/gait acceptance; heading reward follows the route-centering navigation target')
        (self.directory / 'metadata.json').write_text(json.dumps(self.scene_metadata, indent=2) + '\n')

    def _command(self, data, info):
        rotation = data.xmat[self._base_id]
        yaw = jp.arctan2(rotation[1, 0], rotation[0, 0])
        return centered_command(info['command'], info['heading'], info['direction'], data.qpos[1], yaw)

    def _obs(self, data, info):
        info['command'], _ = self._command(data, info)
        return super()._obs(data, info)

    def step(self, state, action):
        following = super().step(state, action)
        data, info = following.data, dict(following.info)
        _, error = self._command(data, info)
        metrics = dict(following.metrics)
        desired_reward = -.4 * error ** 2
        delta = (desired_reward - metrics['heading']) * self.dt
        metrics['heading'] = desired_reward
        valid = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        reward = jp.where(valid, following.reward + delta, -1.)
        info['episode_return'] = state.info['episode_return'] + reward
        return following.replace(info=info, metrics=metrics, reward=reward)
