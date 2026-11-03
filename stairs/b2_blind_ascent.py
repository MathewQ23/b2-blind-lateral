"""Ascent/stand curriculum using exactly the existing blind policy sensor boundary."""
import json
import jax
import jax.numpy as jp
from stairs.b2_blind import B2BlindEnv


def ascent_task(key, task_override=-1):
    random_task = jp.where(jax.random.uniform(jax.random.fold_in(key, 301)) < .1, 3, 0)
    explicit = jp.where(task_override == 3, 3, 0)
    return jp.where(task_override >= 0, explicit, random_task).astype(jp.int32)


class B2BlindAscentEnv(B2BlindEnv):
    def __init__(self, directory, stage=0, seed=20261030):
        super().__init__(directory, stage, seed)
        self.scene_metadata.pop('symmetry', None)
        self.scene_metadata.update(
            task_version='b2_blind_ascent_v1', trained_tasks=['ascent', 'stand'],
            policy_architecture='Ordinary MLP over 25 proprioceptive frames; no symmetry projection',
            reward_revision='Existing B2 stability, gait and clearance rewards; ascent/stand task sampling only',
            reset_policy='Full ascent start or stand; no partial stair starts')
        (self.directory / 'metadata.json').write_text(json.dumps(self.scene_metadata, indent=2) + '\n')

    def reset(self, key, task_override=-1):
        return super().reset(key, ascent_task(key, task_override))
