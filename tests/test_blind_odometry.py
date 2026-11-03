"""Check onboard-only estimation, physical kinematics and lossless policy expansion."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from stairs.b2_blind_odometry import (B2BlindOdometryEnv, expand_blind_checkpoint,
    foot_kinematics, model_leg_offsets, stance_velocity)
from stairs.b2_scene import build_b2_scene, FEET
from stairs.train import ActorCritic, normalize


class KinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        scene, meta, _ = build_b2_scene(Path(cls.directory.name), stage=11)
        cls.model = mujoco.MjModel.from_xml_path(str(scene))
        cls.order = meta['actuator_joint_order']
        cls.qadr = np.array([cls.model.jnt_qposadr[cls.model.joint(n).id] for n in cls.order])
        cls.vadr = np.array([cls.model.jnt_dofadr[cls.model.joint(n).id] for n in cls.order])
        cls.offsets = model_leg_offsets(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_fk_and_joint_jacobian_match_native_mujoco(self):
        rng = np.random.default_rng(12)
        data = mujoco.MjData(self.model)
        calc = jax.jit(foot_kinematics)
        for _ in range(12):
            q = np.tile([0., .8, -1.6], 4) + rng.uniform(-.3, .3, 12)
            dq = rng.normal(0., 3., 12)
            data.qpos[:3] = rng.normal(0., 5., 3)
            quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
            data.qpos[3:7] = quat
            data.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, data)
            rotation = data.xmat[self.model.body('base_link').id].reshape(3, 3)
            base = data.xpos[self.model.body('base_link').id]
            position, velocity = calc(jp.asarray(q), jp.asarray(dq), self.offsets)
            for i, foot in enumerate(FEET):
                site = self.model.site(foot).id
                jac = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, data, jac, None, site)
                np.testing.assert_allclose(position[i], rotation.T @ (data.site_xpos[site] - base), atol=2e-6)
                np.testing.assert_allclose(velocity[i], rotation.T @ jac[:, self.vadr] @ dq, atol=2e-6)

    def test_stance_constraint_recovers_velocity_and_ignores_swing_feet(self):
        position = jp.array([[.3, -.2, -.5], [.3, .2, -.5], [-.3, -.2, -.5], [-.3, .2, -.5]])
        body = jp.array([.3, -.1, .05]); gyro = jp.array([.2, .1, -.3])
        relative = -body - jp.cross(gyro, position)
        relative = relative.at[1:].set(100.)
        estimate = stance_velocity(position, relative, gyro, jp.array([1., 0., 0., 0.]), body)
        np.testing.assert_allclose(estimate, body, atol=1e-6)
        airborne = stance_velocity(position, relative, gyro, jp.zeros(4), body)
        np.testing.assert_array_equal(airborne, body)

    def make_env(self):
        env = object.__new__(B2BlindOdometryEnv)
        env._base_id = 0
        env._qadr = jp.arange(7, 19); env._vadr = jp.arange(6, 18)
        env._default_pose = jp.tile(jp.array([0., .8, -1.6]), 4)
        env._leg_offsets = self.offsets
        env.get_gyro = lambda d: d.gyro
        env.terrain = lambda *args: self.fail('Actor must not read terrain')
        return env

    def info(self):
        return dict(rng=jax.random.PRNGKey(7), last_act=jp.zeros(12),
            last_contact=jp.ones(4, dtype=bool), command=jp.array([.3, 0., 0.]),
            payload_kg=jp.array(0.), goal_x=jp.array(10.), phase=jp.array(0.))

    def test_observation_cannot_read_simulator_base_velocity_or_terrain(self):
        env = self.make_env()
        data = SimpleNamespace(xmat=jp.eye(3)[None], qpos=jp.zeros(19),
            qvel=jp.zeros(18), gyro=jp.zeros(3))
        first = env._obs(data, self.info())['state']
        changed = SimpleNamespace(**vars(data))
        changed.qpos = changed.qpos.at[:3].set(jp.array([100., -50., 20.]))
        changed.qvel = changed.qvel.at[:6].set(jp.array([3., 2., 1., -2., 5., 8.]))
        info = self.info(); info.update(payload_kg=jp.array(20.), goal_x=jp.array(-400.), phase=jp.array(2.))
        second = env._obs(changed, info)['state']
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (25 * 52,))

    def test_history_preserves_sensor_and_estimator_frames(self):
        env = self.make_env(); info = self.info()
        data = SimpleNamespace(xmat=jp.eye(3)[None], qpos=jp.zeros(19), qvel=jp.zeros(18), gyro=jp.zeros(3))
        old = np.asarray(env._obs(data, info)['state']).reshape(25, 52)
        data.qvel = data.qvel.at[7].set(2.)
        new = np.asarray(env._obs(data, info)['state']).reshape(25, 52)
        np.testing.assert_array_equal(new[:-1], old[1:])
        self.assertGreater(np.linalg.norm(new[-1, 49:] - old[-1, 49:]), .01)


class MigrationTest(unittest.TestCase):
    def test_expansion_preserves_actor_critic_for_arbitrary_new_features(self):
        net = ActorCritic(width=8)
        rng = np.random.default_rng(8)
        old = rng.normal(size=(10, 25 * 49)).astype(np.float32)
        params = jax.device_get(net.init(jax.random.PRNGKey(9), jp.asarray(old)))
        norm = (rng.normal(size=25 * 49).astype(np.float32),
                rng.uniform(.01, 2., 25 * 49).astype(np.float32), np.float32(1e7))
        source = dict(params=params, norm=norm, step=12, stage=11, task_version='b2_blind_ascent_v1')
        expanded = expand_blind_checkpoint(source)
        new = np.concatenate((old.reshape(10, 25, 49), rng.normal(size=(10, 25, 3))), axis=-1)
        expected = net.apply(params, normalize(old, norm))
        actual = net.apply(expanded['params'], normalize(new.reshape(10, -1), expanded['norm']))
        for before, after in zip(expected, actual):
            np.testing.assert_allclose(before, after, rtol=1e-5, atol=2e-6)
        self.assertEqual(source['params']['params']['actor_0']['kernel'].shape[0], 1225)

    def test_height_aware_checkpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            expand_blind_checkpoint(dict(task_version='b2_payload_v4'))


if __name__ == '__main__':
    unittest.main()
