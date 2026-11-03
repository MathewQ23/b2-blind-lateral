#!/usr/bin/env python3
"""Expand an audited blind checkpoint without changing its actor/critic outputs."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import numpy as np
from stairs.b2_blind_odometry import expand_blind_checkpoint
from stairs.train import ActorCritic, atomic_pickle, normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path; do not overwrite a migration artifact')
    with args.source.open('rb') as stream:
        source = pickle.load(stream)
    expanded = expand_blind_checkpoint(source)
    # Exercise all old feature rows plus arbitrary estimated velocities. Zero new
    # weights must preserve the action means, logstd and value before retraining.
    rng = np.random.default_rng(20261031)
    mean, variance, _ = source['norm']
    old = (mean + rng.normal(size=(64, 1225)) * np.sqrt(variance + 1e-5)).astype(np.float32)
    new = np.concatenate((old.reshape(64, 25, 49), rng.normal(size=(64, 25, 3))), axis=-1).astype(np.float32)
    net = ActorCritic(width=source['width'], minimum_logstd=source.get('minimum_logstd', -3.),
                      maximum_logstd=source.get('maximum_logstd', -.5))
    before = net.apply(source['params'], normalize(old, source['norm']))
    after = net.apply(expanded['params'], normalize(new.reshape(64, 1300), expanded['norm']))
    differences = [float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(before, after)]
    for a, b in zip(before, after):
        np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    expanded['observation_migration'].update(source_path=str(args.source.resolve()), source_sha256=source_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_pickle(args.output, expanded)
    report = dict(expanded['observation_migration'], output=str(args.output.resolve()),
                  max_absolute_errors=dict(zip(('action_mean', 'logstd', 'value'), differences)),
                  source_observations=1225, expanded_observations=1300, passed=True)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
