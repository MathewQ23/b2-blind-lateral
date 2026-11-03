"""Explicit training exploration regimes, separate from zero-command pose priors."""
import jax.numpy as jp


def exploration_settings(command,world_ids,*,b2=False,profile='standard'):
    stand=jp.linalg.norm(command[...,:2],axis=-1)<.01
    if profile=='mixed':
        walking=jp.array([.125,.25,.5,1.])[world_ids%4]
        low_noise=walking<=.25
    elif profile=='refine':
        # Continue an already walking blind policy with bounded perturbations.
        walking=jp.array([.125,.25,.375,.5])[world_ids%4]
        low_noise=walking<=.25
    elif profile=='gentle':
        walking=jp.array([.0625,.125,.1875,.25])[world_ids%4]
        low_noise=walking<=.125
    elif profile=='standard':
        walking=jp.ones_like(command[...,0])
        low_noise=jp.zeros_like(stand)
    else:raise ValueError(f'Unknown exploration profile {profile}')
    scale=jp.where(stand,.04 if b2 else .2,walking)
    return jp.log(scale),stand,low_noise,scale
