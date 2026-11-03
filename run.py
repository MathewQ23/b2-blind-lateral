"""Load-check, evaluate, or render the retained B2 blind ascent policy."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'evaluate', 'render'))
    parser.add_argument('--checkpoint', type=Path, default=root/'weights/policy.pkl')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--evaluation', type=Path, default=root/'outputs/evaluation')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1040031)
    parser.add_argument('--episode', type=int, default=0, help='Ascent episode index for rendering')
    args = parser.parse_args()
    if args.command == 'check':
        os.environ['JAX_PLATFORMS'] = 'cpu'
        import jax
        import jax.numpy as jp
        from stairs.train import ActorCritic, normalize
        with args.checkpoint.open('rb') as stream:
            saved = pickle.load(stream)
        net = ActorCritic(width=saved['width'], minimum_logstd=saved['minimum_logstd'],
                          maximum_logstd=saved['maximum_logstd'])
        observation = jp.zeros((1, len(saved['norm'][0])))
        mean, _, _ = net.apply(saved['params'], normalize(observation, saved['norm']))
        action = jp.tanh(mean)
        print(json.dumps(dict(task=saved['task_version'], stage=saved['stage'], training_steps=saved['step'],
                              observations=observation.shape[-1], actions=action.shape[-1],
                              finite=bool(jp.isfinite(action).all()), device=str(jax.devices()[0])), indent=2))
    elif args.command == 'evaluate':
        output = args.output or args.evaluation
        subprocess.run([sys.executable, str(root/'scripts/evaluate_taishan.py'), str(args.checkpoint),
                        '--stage', '14', '--episodes', str(args.episodes), '--seed', str(args.seed),
                        '--output', str(output)], check=True, cwd=root)
    else:
        output = args.output or root/'outputs/ascent.mp4'
        subprocess.run([sys.executable, str(root/'scripts/render_taishan_evaluation.py'), str(args.evaluation),
                        '--task', 'ascent', '--episode', str(args.episode), '--output', str(output)],
                       check=True, cwd=root)


if __name__ == '__main__':
    main()
