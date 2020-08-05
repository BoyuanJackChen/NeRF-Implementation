import sys
import random
import jax
from sys import platform
import time
from load_llff import *
import os, sys, time
import imageio
import matplotlib.pyplot as plt
from jax import grad, jit, lax, random, partial, vmap
from jax import numpy as jnp
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, FanInConcat, FanOut, Identity, Relu
from jax.nn import relu, sigmoid
from tqdm.notebook import tqdm
from typing import Any, Optional, List, Tuple
import random as orandom

slash = '/' if (platform == "darwin" or platform == "linux") else '\\'
BASE_DIR = sys.path[0] + "/.."
DATA_DIR = BASE_DIR + '/../Dataset'
CODE_DIR = BASE_DIR + "/Code"
LLFF_DATA = DATA_DIR+"/nerf_llff_data"

@jax.jit
def poses_35_to_44(poses):
    # Transform an (n,3,5) pose array to (n,4,4)
    real_poses = jnp.array(poses[:, :3, :4])
    b = jnp.zeros((poses.shape[0], 1, 4))
    jax.ops.index_update(b, jax.ops.index[:,0,3], 1.0)
    result = jnp.append(arr=real_poses, values=b, axis=1)
    return result

@jax.jit
def get_focal(bds, dt=.75):
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz
    return focal


def build_model(D: int = 8, W: int = 256) -> Any:
    dense_block = lambda block_brep=1, W=W: [Dense(W), Relu] * 1    # * block_rep
    sub_net = stax.serial(*dense_block(5))
    model = stax.serial(
        FanOut(2),
        stax.parallel(sub_net, Identity),
        FanInConcat(-1),
        *dense_block(3),
        Dense(4),
    )
    return model


def embed_fn(x: jnp.ndarray, L_embed: int) -> jnp.ndarray:
    """Positional encoder embedding."""
    rets = [x]
    for i in range(L_embed):
        for fn in [jnp.sin, jnp.cos]:
            rets.append(fn(2.0 ** i * x))
    return jnp.concatenate(rets, -1)
    # rets = vmap(lambda idx: 2.0 ** idx * x)(jnp.arange(L_embed))
    # res = jnp.concatenate([x[None, ...], jnp.sin(rets), jnp.cos(rets)], 0)
    # return jnp.reshape(jnp.swapaxes(res, 0, 1), [-1, 3 + 3 * 2 * L_embed])


@partial(jit, static_argnums=(0, 1, 2))
def get_rays(H: int, W: int, focal: float, c2w: jnp.ndarray) -> jnp.ndarray:
    """Generate ray matrices."""
    i, j = jnp.meshgrid(jnp.arange(W), jnp.arange(H), indexing="xy")
    dirs = jnp.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -jnp.ones_like(i)], -1
    )
    rays_d = jnp.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)
    rays_o = jnp.broadcast_to(c2w[:3, -1], rays_d.shape)
    return jnp.stack([rays_o, rays_d])


# --- change batch_size here ---
L_embed = 10
batch_size = 126*95*2
def render_rays(
    net_fn: Any,
    rays: jnp.ndarray,
    near: float = 2.0,
    far: float = 6.0,
    N_samples: int = 64,
    L_embed: int = 10,
    batch_size: int = batch_size,
    rng: Optional[Any] = None,
    rand: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    print("in render_rays")
    rays_o, rays_d = rays
    # Compute 3D query points
    z_vals = jnp.linspace(near, far, N_samples)
    if rand:
        z_vals += ( random.uniform(rng, list(rays_o.shape[:-1]) + [N_samples])
            * (far - near)
            / N_samples )
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Run network
    pts_flat = jnp.reshape(pts, [-1, 3])
    pts_flat = embed_fn(pts_flat, L_embed)
    raw = lax.map(net_fn, jnp.reshape(pts_flat, [-1, batch_size, pts_flat.shape[-1]]))
    # jax.profiler.save_device_memory_profile("myagues_raw.prof")
    raw = jnp.reshape(raw, list(pts.shape[:-1]) + [4])

    # Compute opacities and colors
    sigma_a = relu(raw[..., 3])
    rgb = sigmoid(raw[..., :3])

    # Do volume rendering
    dists = jnp.concatenate(
        [ z_vals[..., 1:] - z_vals[..., :-1],
            jnp.broadcast_to([1e10], z_vals[..., :1].shape),
        ], -1,
    )
    alpha = 1.0 - jnp.exp(-sigma_a * dists)
    alpha_ = jnp.minimum(1.0, 1.0 - alpha + 1e-10)
    trans = jnp.concatenate([jnp.ones_like(alpha_[..., :1]), alpha_[..., :-1]], -1)
    weights = alpha * jnp.cumprod(trans, -1)

    rgb_map = jnp.sum(weights[..., None] * rgb, -2)
    depth_map = jnp.sum(weights * z_vals, -1)
    acc_map = jnp.sum(weights, -1)
    return rgb_map, depth_map, acc_map


def loss_fun(
    params: jnp.ndarray,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    rng: Optional[Any] = None,
    rand: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute loss function for optimizer and return generated image."""
    rays, target = batch
    model_fn_ = partial(model_fn, params)
    rgb, _, _ = render_rays(model_fn_, rays, near=2., far=6., N_samples=64,
                            L_embed=L_embed, rng=rng, rand=rand)
    return jnp.mean(jnp.square(rgb - target)), rgb


@jit
def update(i: int, opt_state: Any, rng: Any) -> Any:
    img_rng, fn_rng = random.split(random.fold_in(rng, i))
    img_idx = random.randint(img_rng, (1,), minval=0, maxval=len(sorted_list)-1)
    batch = (train_rays[img_idx][0], images[img_idx][...,:3]/255.)
    params = get_params(opt_state)
    print("entering loss")
    grads, _ = grad(loss_fun, has_aux=True)(params, batch, fn_rng, True)
    return opt_update(i, grads, opt_state)


@jit
def evaluate(params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluation step w/ PSNR metric."""
    loss, rgb = loss_fun(params, (test_rays, testimg))
    psnr = -10.0 * jnp.log(loss) / jnp.log(10.0)
    return rgb, psnr


if __name__ == "__main__":
    # --- Load the fortress scene in 1\factor^2 resolution ---
    factor = 32
    imagedir = LLFF_DATA+"/fortress"
    print(f"basedir is: {imagedir}")
    images, raw_poses, bds, render_poses, i_test = load_llff_data(imagedir, factor=64,
            recenter=True, bd_factor=.75, spherify=False, path_zflat=False)
    images = jnp.array(images)
    # raw_poses, bds, render_poses, i_test = load_llff_data_noimage(imagedir, factor=factor,
    #         recenter=True, bd_factor=.75, spherify=False, path_zflat=False)
    poses = jnp.array(poses_35_to_44(raw_poses))
    focal = get_focal(bds)
    # print(f"images shape {images.shape}; poses shape {poses.shape}; focal is {focal}")

    L_embed = 10
    key = random.PRNGKey(0)
    init_fn, model_fn = build_model()
    _, model_params = init_fn(key, input_shape=(3 + 3 * 2 * L_embed,))
    opt_init, opt_update, get_params = optimizers.adam(step_size=5e-4, b1=0.9, b2=0.999, eps=1e-08)
    opt_state = opt_init(model_params)
    testimg, testpose = images[0], poses[0]

    # basedir = LLFF_DATA + "/fortress"
    # imagedir = basedir + "/images"
    # if factor is not None:
    #     imagedir += "_" + str(factor)
    # print(f"basedir is: {imagedir}")
    # sorted_list = os.listdir(imagedir)
    # sorted_list.sort()
    # testimg = imageio.imread(imagedir + "/" + sorted_list[0])[..., :3] / 255.
    # testimg = np.asarray(testimg)
    # testpose = poses[0]

    H, W, _ = testimg.shape
    train_rays = lax.map(lambda pose: get_rays(H, W, focal, pose), poses)
    test_rays = get_rays(H, W, focal, testpose)

    N_iters = 1000
    psnrs: List[float] = []
    iternums: List[int] = []
    i_plot = 20
    for i in range(N_iters + 1):
        t = time.perf_counter()
        opt_state = update(i, opt_state, key)

        if i % i_plot == 0:
            print(f"Iterations: {i:4d}\t{time.perf_counter() - t:2.5f} sec/iter", end="")
            rgb, psnr = evaluate(get_params(opt_state))
            print(f"\tPSNR: {psnr:.5f}")
            psnrs.append(psnr)
            iternums.append(i)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.imshow(rgb)
            ax1.axis("off")
            ax2.plot(iternums, psnrs)
            plt.show()

    final_model_fn = partial(model_fn, get_params(opt_state))