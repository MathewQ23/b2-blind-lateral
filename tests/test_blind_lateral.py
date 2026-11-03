"""Check route projection, long-lived memory, sensor isolation and policy migration."""
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jp
import numpy as np

from stairs.b2_blind_lateral import (B2BlindLateralEnv, expand_odometry_checkpoint,
    integrate_lateral, route_lateral_velocity)
from stairs.train import ActorCritic, normalize


class LateralMemoryTest(unittest.TestCase):
    def test_velocity_projection_corrects_body_tilt_and_heading(self):
        for roll, pitch, yaw in [(0., 0., 0.), (.2, .3, -.2), (-.3, -.2, .3)]:
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            rotation = rz @ ry @ rx
            route_velocity = np.array([.35, -.12, .08])
            body = rotation.T @ route_velocity
            gravity = rotation.T @ np.array([0., 0., -1.])
            estimate = route_lateral_velocity(jp.asarray(body), jp.asarray(gravity), -1.5 * yaw)
            self.assertAlmostEqual(float(estimate), -.12, places=6)

    def test_memory_spans_more_than_the_half_second_history_and_is_bounded(self):
        def tick(position, _):
            following = integrate_lateral(position, jp.array([.3, .1, 0.]), jp.array([0., 0., -1.]), 0.)
            return following, following
        final, _ = jax.lax.scan(tick, jp.array(0.), None, length=500)
        self.assertAlmostEqual(float(final), 1., delta=1e-5)
        self.assertEqual(float(integrate_lateral(2., jp.array([0., 1., 0.]), jp.array([0., 0., -1.]), 0.)), 2.)

    def make_env(self):
        env = object.__new__(B2BlindLateralEnv)
        env._base_id = 0; env._qadr = jp.arange(7, 19); env._vadr = jp.arange(6, 18)
        env._default_pose = jp.tile(jp.array([0., .8, -1.6]), 4)
        env._leg_offsets = (jp.array([[.3285, -.072, 0.], [.3285, .072, 0.], [-.3285, -.072, 0.], [-.3285, .072, 0.]]),
            jp.array([[0., -.11973, 0.], [0., .11973, 0.], [0., -.11973, 0.], [0., .11973, 0.]]),
            jp.tile(jp.array([0., 0., -.35]), (4, 1)), jp.tile(jp.array([0., 0., -.35]), (4, 1)))
        env.get_gyro = lambda d: d.gyro
        env.terrain = lambda *args: self.fail('Actor must not read terrain')
        return env

    def info(self):
        return dict(rng=jax.random.PRNGKey(7), last_act=jp.zeros(12), last_contact=jp.ones(4, dtype=bool),
                    command=jp.array([.3, 0., 0.]), payload_kg=jp.array(0.), goal_x=jp.array(10.), phase=jp.array(0.))

    def test_world_position_and_real_base_velocity_cannot_change_observation(self):
        env = self.make_env()
        data = SimpleNamespace(xmat=jp.eye(3)[None], qpos=jp.zeros(19), qvel=jp.zeros(18), gyro=jp.zeros(3))
        first = env._obs(data, self.info())['state']
        changed = SimpleNamespace(**vars(data))
        changed.qpos = changed.qpos.at[:3].set(jp.array([100., -50., 20.]))
        changed.qvel = changed.qvel.at[:6].set(jp.array([3., 2., 1., -2., 5., 8.]))
        info = self.info(); info.update(payload_kg=jp.array(20.), goal_x=jp.array(-400.), phase=jp.array(2.))
        np.testing.assert_array_equal(first, env._obs(changed, info)['state'])
        self.assertEqual(first.shape, (25 * 53,))

    def test_observation_keeps_memory_history(self):
        env = self.make_env(); info = self.info()
        data = SimpleNamespace(xmat=jp.eye(3)[None], qpos=jp.zeros(19), qvel=jp.zeros(18), gyro=jp.zeros(3))
        old = np.asarray(env._obs(data, info)['state']).reshape(25, 53)
        info['estimated_lateral_travel'] = jp.array(.5)
        new = np.asarray(env._obs(data, info)['state']).reshape(25, 53)
        np.testing.assert_array_equal(new[:-1], old[1:])
        self.assertAlmostEqual(float(new[-1, 52]), .5, delta=.01)


class LateralMigrationTest(unittest.TestCase):
    def test_actor_and_critic_outputs_survive_new_memory_channels(self):
        net = ActorCritic(width=8); rng = np.random.default_rng(11)
        old = rng.normal(size=(10, 1300)).astype(np.float32)
        params = jax.device_get(net.init(jax.random.PRNGKey(4), jp.asarray(old)))
        norm = (rng.normal(size=1300).astype(np.float32), rng.uniform(.01, 2., 1300).astype(np.float32), np.float32(1e7))
        source = dict(params=params, norm=norm, step=12, stage=11, task_version='b2_blind_odometry_v1')
        expanded = expand_odometry_checkpoint(source)
        new = np.concatenate((old.reshape(10, 25, 52), rng.normal(size=(10, 25, 1))), axis=-1).astype(np.float32)
        expected = net.apply(params, normalize(old, norm))
        actual = net.apply(expanded['params'], normalize(new.reshape(10, 1325), expanded['norm']))
        for a, b in zip(expected, actual):np.testing.assert_allclose(a, b, atol=2e-6, rtol=1e-5)
        self.assertEqual(source['params']['params']['actor_0']['kernel'].shape[0], 1300)

    def test_non_odometry_source_is_rejected(self):
        with self.assertRaises(ValueError):expand_odometry_checkpoint(dict(task_version='b2_payload_v4'))


if __name__ == '__main__':
    unittest.main()
