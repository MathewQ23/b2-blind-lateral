"""B2 policy observations restricted to a 0.5 s history of body/joint/contact sensors."""
import json
import jax
import jax.numpy as jp
import numpy as np
from stairs.b2_symmetry import B2SymmetricEnv


class B2BlindEnv(B2SymmetricEnv):
    history_frames = 25
    frame_size = 49
    frame_fields = dict(gyro=[0, 3], gravity=[3, 6], joint_position=[6, 18],
                        joint_velocity=[18, 30], previous_action=[30, 42],
                        foot_contact=[42, 46], command=[46, 49])

    def __init__(self, directory, stage=0, seed=20261030):
        super().__init__(directory, stage, seed)
        self.scene_metadata.update(
            task_version='b2_blind_v1', observation_size=self.history_frames * self.frame_size,
            observation='25 frames of gyro, gravity, joint positions/velocities, last action, binary foot contact and command',
            observation_frame_fields=self.frame_fields, history_dt=self.dt, history_frames=self.history_frames,
            excluded_policy_inputs=['terrain scan', 'terrain height', 'payload mass', 'world position',
                                    'goal distance', 'gait phase', 'simulator base linear velocity'],
            reward_privilege='Terrain may be used to compute reward/termination, never as a policy input')
        # This is a look-only copy. Existing code checks physical properties before it is installed.
        from scripts.build_studio_stairs import build
        styled = self.directory / 'studio.xml'
        build(self.xml_path, styled)
        styled.replace(self.xml_path)
        (self.directory / 'metadata.json').write_text(json.dumps(self.scene_metadata, indent=2) + '\n')

    def _obs(self, data, info):
        key, noise_key = jax.random.split(info['rng'])
        info['rng'] = key
        rotation = data.xmat[self._base_id]
        frame = jp.concatenate((self.get_gyro(data), rotation.T @ jp.array([0., 0., -1.]),
                                data.qpos[self._qadr] - self._default_pose,
                                data.qvel[self._vadr], info['last_act'],
                                info['last_contact'].astype(jp.float32), info['command']))
        noise = jp.concatenate((jp.full(3, .05), jp.full(3, .0125), jp.full(12, .0075),
                                jp.full(12, .375), jp.zeros(19)))
        frame += jax.random.uniform(noise_key, frame.shape, minval=-1., maxval=1.) * noise
        if 'proprio_history' in info:
            history = jp.concatenate((info['proprio_history'][1:], frame[None]), axis=0)
        else:
            history = jp.repeat(frame[None], self.history_frames, axis=0)
        info['proprio_history'] = history
        return {'state': history.reshape(-1)}
