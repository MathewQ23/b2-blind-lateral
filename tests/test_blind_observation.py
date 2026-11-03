"""Audit the actor input boundary without compiling the GPU physics engine."""
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jp
import numpy as np
from stairs.b2_blind import B2BlindEnv


class BlindObservationTest(unittest.TestCase):
    def setUp(self):
        self.env = object.__new__(B2BlindEnv)
        self.env._base_id = 0
        self.env._qadr = jp.arange(7, 19)
        self.env._vadr = jp.arange(6, 18)
        self.env._default_pose = jp.zeros(12)
        self.env.get_gyro = lambda d: d.gyro
        self.env.terrain = lambda *a, **k: self.fail('Terrain must never be read by the actor observation')
        self.data = SimpleNamespace(xmat=jp.eye(3)[None], qpos=jp.zeros(19), qvel=jp.zeros(18), gyro=jp.zeros(3))

    def info(self):
        return dict(rng=jax.random.PRNGKey(7), last_act=jp.zeros(12), last_contact=jp.zeros(4, dtype=bool),
                    command=jp.array([.3, 0., 0.]), payload_kg=jp.array(0.), goal_x=jp.array(10.), phase=jp.array(0.))

    def test_forbidden_environment_information_does_not_change_observation(self):
        first = self.env._obs(self.data, self.info())['state']
        other = SimpleNamespace(**vars(self.data))
        other.qpos = other.qpos.at[:3].set(jp.array([100., -50., 20.]))
        other.qvel = other.qvel.at[:3].set(jp.array([3., 2., 1.]))
        info = self.info()
        info.update(payload_kg=jp.array(20.), goal_x=jp.array(-400.), phase=jp.array(2.))
        second = self.env._obs(other, info)['state']
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (25 * 49,))

    def test_history_uses_previous_sensor_frames(self):
        info = self.info()
        old = np.array(self.env._obs(self.data, info)['state']).reshape(25, 49)
        self.data.qpos = self.data.qpos.at[7].set(.4)
        new = np.array(self.env._obs(self.data, info)['state']).reshape(25, 49)
        np.testing.assert_array_equal(new[:-1], old[1:])
        self.assertGreater(new[-1, 6] - old[-1, 6], .38)


if __name__ == '__main__':
    unittest.main()
