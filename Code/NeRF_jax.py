import sys
import random
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
from sys import platform
import time
import jax
import jax.numpy as jnp
import flax.nn as nn
from flax.nn import initializers
from flax import optim
from flax.training import checkpoints
import matplotlib.pyplot as plt
import os
from load_llff import *

slash = '/' if (platform == "darwin" or platform == "linux") else '\\'
BASE_DIR = sys.path[0] + "/.."
DATA_DIR = BASE_DIR + '/Dataset'
CODE_DIR = BASE_DIR + "/Code"
LLFF_DATA = DATA_DIR+"/nerf_llff_data"


@jax.jit
def posenc(x):     # γ(.)
    rets = jnp.array(x)
    for i in range(L_embed):
        for fn in [jnp.sin, jnp.cos]:
            val = fn(2. ** i * x)
            rets = jnp.append(arr=rets, values=val, axis=1)
    return rets

# Embed Parameters
L_embed = 8
embed_fn = posenc

@jax.jit
def get_focal(bds, dt=.75):
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz
    return focal

@jax.jit
def poses_35_to_44(poses):
    # Transform an (n,3,5) pose array to (n,4,4)
    real_poses = jnp.array(poses[:, :3, :4])
    b = jnp.zeros((poses.shape[0], 1, 4))
    jax.ops.index_update(b, jax.ops.index[:,0,3], 1.0)
    result = jnp.append(arr=real_poses, values=b, axis=1)
    return result


class NeRF_Model(nn.Module):
    def apply(self, x):
        the_input = x
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        x = nn.Dense(x, features=256)
        x = jnp.append(arr=x, values=the_input, axis=1)  # Dense_4: (257, 256)
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        x = nn.Dense(x, features=4, kernel_init = initializers.normal(1),
                     bias_init=initializers.ones)
        # output: (r,g,b, \upsigma)
        return x


def create_model(key):
    if key is None:
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
    # The first in the tuple has to be 1; the second is the input size.
    # In the sine model case, it is 1 as well.
    _, initial_params = NeRF_Model.init_by_shape(key, [((1, 3+3*2*L_embed), jnp.float32)])
    model = nn.Model(NeRF_Model, initial_params)
    return model


def mean_squared_loss(logits, labels, scalar=1.0):
    # The shapes of both are (100, 1)
    return jnp.mean(jnp.sum((labels - logits)**2)) * scalar


def create_adam_optimizer(model, learning_rate, beta1=0.9, beta2=0.999, weight_decay=0):
    optimizer_def = optim.Adam(learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                               eps=1e-09, weight_decay=weight_decay)
    optimizer = optimizer_def.create(model)
    return optimizer


def get_rays(H, W, focal, c2w):
    i, j = jnp.meshgrid(jnp.arange(0, W, dtype=jnp.float32),
                        jnp.arange(0, H, dtype=jnp.float32), sparse=False, indexing='xy')
    dirs = jnp.stack(arrays=[(i-W*.5)/focal, -(j-H*.5)/focal, -jnp.ones_like(i)], axis=-1)
    old_shape = dirs.shape
    new_shape = jnp.array([dirs.shape[0], dirs.shape[1], 1, dirs.shape[2]])
    rays_d = jnp.sum(jnp.reshape(dirs, new_shape) * c2w[:3, :3], -1)
    rays_o = jnp.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o, rays_d


batchify_size=1024*16
def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False,
                key=jax.random.PRNGKey(0)):
    def batchify(fn):
        return jax.remat(lambda inputs: jnp.concatenate([fn(inputs[i:i + batchify_size])
                                         for i in range(0, inputs.shape[0], batchify_size)], 0))
    z_vals = jnp.linspace(near, far, N_samples)
    if rand:
        key, subkey = jax.random.split(key)
        z_vals += jax.random.uniform(subkey, list(rays_o.shape[:-1]) + [N_samples], dtype=jnp.float32) \
                  * (far - near) / N_samples
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    pts_flat = jnp.reshape(pts, [-1, 3])
    pts_flat = embed_fn(pts_flat)     # pts_flat is an array of shape (H*W*N*3, 51)
    # --- This is where I wish it to batchify, but it fails to! ---
    raw = batchify(network_fn)(pts_flat)
    jax.profiler.save_device_memory_profile("batchify-line.prof")
    raw = jnp.reshape(raw, list(pts.shape[:-1]) + [4])

    # Compute opacities and colors
    sigma_a = nn.relu(raw[..., 3])    # (H, W, N_samples)
    rgb = nn.sigmoid(raw[..., :3])    # (H, W, N_samples, 3)

    # Do volume rendering (P6 equation (3))
    dists = jnp.concatenate((z_vals[..., 1:] - z_vals[..., :-1],
                       jnp.broadcast_to([1e10], z_vals[..., :1].shape)), -1)   # (H, W, N_samples)
    alpha = 1. - jnp.exp(-sigma_a * dists)
    weights = alpha * jnp.cumprod(1. - alpha + 1e-10, axis=-1, dtype=jnp.float32)
    # Compute cumulative product along axis
    rgb_map   = jnp.sum(weights[..., None] * rgb, -2)
    depth_map = jnp.sum(weights * z_vals, -1)
    acc_map   = jnp.sum(weights, -1)
    return rgb_map, depth_map, acc_map


''' Function trains a jax nerf optimizer from its initial state '''
def train_nerf(optimizer, images, poses, focal, near, far, batchify_size=1024*32,
               N_samples=32, N_iters=2000, i_plot=100, monitor=30, test_index=0):
    H,W = (images.shape)[1:3]
    testimg = images[test_index]; testpose = poses[test_index]
    rand_key = jax.random.PRNGKey(0)
    for i in range(N_iters + 1):
        img_i = random.randint(0, images.shape[0])
        target = images[img_i]
        pose = poses[img_i]
        rays_o, rays_d = get_rays(H, W, focal, pose)
        # Render the image and calculate the loss
        def loss_fn(model):
            rgb, depth, acc = render_rays(model, rays_o, rays_d, near=near, far=far,
                                N_samples=N_samples, rand=True)
            loss = jnp.mean(jnp.square(rgb - target))
            return loss, rgb
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grad = grad_fn(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        print(f"Step {i} done; loss is {loss}")
    return optimizer


if __name__ == "__main__":
    L_embed = 10     # Input augmentation
    embed_fn = posenc

    # --- Load the fortress scene in 1\factor^2 resolution ---
    imagedir = LLFF_DATA+"/fortress"
    print(f"basedir is: {imagedir}")
    images, raw_poses, bds, render_poses, i_test = load_llff_data(imagedir, factor=64,
            recenter=True, bd_factor=.75, spherify=False, path_zflat=False)
    poses = poses_35_to_44(raw_poses)
    focal = get_focal(bds)
    print(f"images shape {images.shape}; poses shape {poses.shape}; focal is {focal}")

    N_iters = 3000
    model = create_model(key=None)
    optimizer = create_adam_optimizer(model, learning_rate=5e-4, beta1=0.9, beta2=0.999)
    train_nerf(optimizer, images, poses, focal, near=2., far=6.,
               N_samples=64, N_iters=2000, i_plot=100, monitor=30, test_index=0)
