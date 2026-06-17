import math
import numpy as np
import time
import sklearn.metrics


def sample_random_paths(
        pos_x,
        pos_y,
        orientation_z,
        *,
        n_steps,
        n_skip,
        dt,
        min_v,
        max_v,
        min_w,
        max_w,
        n_paths,
        rng,
        last_best_vels=None,
):
    """
    Sample random piecewise-constant control sequences and roll out unicycle dynamics.

    This mirrors the sampling-based ("MPPI-style") rollout used by FunctionalCPMPC so
    that every controller searches over the same continuous (v, w) action space rather
    than a fixed 3x3 grid of (v, w) levels. Controls are blocked over `n_skip` steps;
    if `last_best_vels` is provided, its shifted continuation seeds the first candidate
    (warm start).

    Returns
    -------
    paths : np.ndarray, shape (n_paths, n_steps + 1, 2)
    vels  : np.ndarray, shape (n_paths, n_steps, 2)
    """
    n_steps = int(n_steps)
    n_skip = int(n_skip)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    # Number of control epochs after blocking
    n_epochs = int(math.ceil(n_steps / max(1, n_skip)))

    # Sample epoch-wise controls
    v_epoch = rng.uniform(min_v, max_v, size=(n_paths, n_epochs)).astype(np.float32)
    w_epoch = rng.uniform(min_w, max_w, size=(n_paths, n_epochs)).astype(np.float32)

    # Warm start (first candidate): reuse the previous best plan shifted by one step
    if last_best_vels is not None and last_best_vels.shape[0] >= 2:
        v_warm = np.append(last_best_vels[1:, 0], rng.uniform(min_v, max_v))
        w_warm = np.append(last_best_vels[1:, 1], rng.uniform(min_w, max_w))
        v_epoch[0, :] = v_warm[:n_epochs]
        w_epoch[0, :] = w_warm[:n_epochs]

    # Expand to per-step controls and truncate to horizon length
    v = np.repeat(v_epoch, n_skip, axis=1)[:, :n_steps]  # (P, n_steps)
    w = np.repeat(w_epoch, n_skip, axis=1)[:, :n_steps]  # (P, n_steps)

    # Roll out positions
    paths = np.zeros((n_paths, n_steps + 1, 2), dtype=np.float32)
    paths[:, 0, 0] = float(pos_x)
    paths[:, 0, 1] = float(pos_y)

    th = np.full((n_paths,), float(orientation_z), dtype=np.float32)
    dt = float(dt)

    for t in range(n_steps):
        paths[:, t + 1, 0] = paths[:, t, 0] + dt * v[:, t] * np.cos(th)
        paths[:, t + 1, 1] = paths[:, t, 1] + dt * v[:, t] * np.sin(th)
        th = th + dt * w[:, t]

    vels = np.stack([v, w], axis=-1).astype(np.float32)  # (P, n_steps, 2)
    return paths, vels


def sample_random_paths_3d(
        robot_xyz,
        robot_yaw,
        *,
        n_steps,
        n_skip,
        dt,
        v_lim,
        w_lim,
        vz_lim,
        n_paths,
        rng,
        last_best_vels=None,
):
    """
    3D analogue of :func:`sample_random_paths` for the (v_xy, yaw_rate, vz) action space.

    Samples random piecewise-constant control sequences and rolls out the same
    unicycle-with-altitude dynamics used by the legacy grid generator so the egocentric
    3D baseline searches over a continuous, sampling-based ("MPPI-style") action set
    rather than a fixed (v, w, vz) meshgrid.

    Returns
    -------
    paths : np.ndarray, shape (n_paths, n_steps + 1, 3)
    vels  : np.ndarray, shape (n_paths, n_steps, 3)  with last dim (v_xy, yaw_rate, vz)
    """
    n_steps = int(n_steps)
    n_skip = int(n_skip)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    n_epochs = int(math.ceil(n_steps / max(1, n_skip)))
    vmin, vmax = float(v_lim[0]), float(v_lim[1])
    wmin, wmax = float(w_lim[0]), float(w_lim[1])
    vzmin, vzmax = float(vz_lim[0]), float(vz_lim[1])

    v_epoch = rng.uniform(vmin, vmax, size=(n_paths, n_epochs)).astype(np.float32)
    w_epoch = rng.uniform(wmin, wmax, size=(n_paths, n_epochs)).astype(np.float32)
    vz_epoch = rng.uniform(vzmin, vzmax, size=(n_paths, n_epochs)).astype(np.float32)

    # Warm start (first candidate): reuse the previous best plan shifted by one step
    if last_best_vels is not None and last_best_vels.shape[0] >= 2:
        v_warm = np.append(last_best_vels[1:, 0], rng.uniform(vmin, vmax))
        w_warm = np.append(last_best_vels[1:, 1], rng.uniform(wmin, wmax))
        vz_warm = np.append(last_best_vels[1:, 2], rng.uniform(vzmin, vzmax))
        v_epoch[0, :] = v_warm[:n_epochs]
        w_epoch[0, :] = w_warm[:n_epochs]
        vz_epoch[0, :] = vz_warm[:n_epochs]

    v = np.repeat(v_epoch, n_skip, axis=1)[:, :n_steps]    # (P, n_steps)
    w = np.repeat(w_epoch, n_skip, axis=1)[:, :n_steps]
    vz = np.repeat(vz_epoch, n_skip, axis=1)[:, :n_steps]

    paths = np.zeros((n_paths, n_steps + 1, 3), dtype=np.float32)
    paths[:, 0, 0] = float(robot_xyz[0])
    paths[:, 0, 1] = float(robot_xyz[1])
    paths[:, 0, 2] = float(robot_xyz[2])

    yaw = np.full((n_paths,), float(robot_yaw), dtype=np.float32)
    dt = float(dt)

    for t in range(n_steps):
        yaw = yaw + dt * w[:, t]
        paths[:, t + 1, 0] = paths[:, t, 0] + dt * v[:, t] * np.cos(yaw)
        paths[:, t + 1, 1] = paths[:, t, 1] + dt * v[:, t] * np.sin(yaw)
        paths[:, t + 1, 2] = paths[:, t, 2] + dt * vz[:, t]

    vels = np.stack([v, w, vz], axis=-1).astype(np.float32)  # (P, n_steps, 3)
    return paths, vels


def compute_quantiles(x, axis, levels):

    levels_clipped = np.clip(levels, 0., 1.)
    x_sorted = np.sort(x, axis)
    sample_size = x.shape[axis]
    idxs = (sample_size - 1) * levels_clipped
    idxs_low = np.floor(idxs).astype(int)
    idxs_high = np.ceil(idxs).astype(int)

    idxs_low = np.expand_dims(idxs_low, axis)
    idxs_high = np.expand_dims(idxs_high, axis)

    values_low = np.squeeze(np.take_along_axis(x_sorted, idxs_low, axis), axis)
    values_high = np.squeeze(np.take_along_axis(x_sorted, idxs_high, axis), axis)

    fractions = idxs - np.floor(idxs)

    quantiles = values_low + fractions * (values_high - values_low)

    quantiles = np.where(levels <= 0., 0., quantiles)
    quantiles = np.where(levels >= 1., np.inf, quantiles)

    return quantiles


def compute_pairwise_distances(x, y):
    """
    x: numpy array of shape (m_0, ..., m_{k-1}, feature dim.) where k >= 0
    y: numpy array of shape (n_0, ..., n_{l-1}, feature dim.) where l >= 0
    return: numpy array of shape (m_0, ..., m_{k-1}, n_0, ..., n_{l-1}) containing the pairwise distances between x & y
    """
    feature_dim = x.shape[-1]
    assert feature_dim == y.shape[-1]

    batch_shape_x = x.shape[:-1]
    batch_shape_y = y.shape[:-1]
    # flattened so that # dim = 2
    X = np.reshape(x, newshape=(-1, feature_dim))
    Y = np.reshape(y, newshape=(-1, feature_dim))
    # matrix of shape (m_0 x ... x m_{k-1}, n_0 x ... x n_{l-1})
    D = sklearn.metrics.pairwise_distances(X, Y, metric='euclidean')
    batch_shape = batch_shape_x + batch_shape_y
    D = np.reshape(D, newshape=batch_shape)
    return D



def compute_pairwise_distances_along_axis(x, y, axis):
    """
    x: numpy array of shape (m, feature dim.) where k >= 0 & m = (m_0, ..., m_{k-1}): multi-index
    y: numpy array of shape (n, feature dim.) where l >= 0 & n = (n_0, ..., n_{l-1}): multi-index
    axis : tuple of two integers i & j representing the axis of x & axis of y along which the computation is performed, respectively
           Note that i != k, j != l, and m_i must be equal to n_j.
    return: numpy array of shape (dim, m_{-i}, n_{-j}) containing the pairwise distances between x & y, where dim: dimension at the given axis
    """
    ndim_x = x.ndim
    ndim_y = y.ndim

    assert ndim_x >= 2 and ndim_y >= 2

    axis_x, axis_y = axis

    axis_dim = x.shape[axis_x]
    assert axis_dim == y.shape[axis_y]

    # axis != last axis
    assert axis_x % ndim_x != ndim_x - 1
    assert axis_y % ndim_y != ndim_y - 1

    # same feature dim
    feature_dim = x.shape[-1]
    assert feature_dim == y.shape[-1]

    batch_shape_x = x.shape[:-1]
    batch_shape_y = y.shape[:-1]

    n_batch_dim_x = len(batch_shape_x)
    n_batch_dim_y = len(batch_shape_y)

    # permute the axes
    X = np.moveaxis(x, axis_x, 0)
    Y = np.moveaxis(y, axis_y, 0)

    # flatten the arrays
    X = np.reshape(X, newshape=(axis_dim, -1, 1, feature_dim))
    Y = np.reshape(Y, newshape=(axis_dim, 1, -1, feature_dim))

    # new shape except the specified axis
    batch_shape_x = tuple(batch_shape_x[i] for i in range(n_batch_dim_x) if i != axis_x)
    batch_shape_y = tuple(batch_shape_y[i] for i in range(n_batch_dim_y) if i != axis_y)

    Ds = []
    for i in range(axis_dim):
        D = compute_pairwise_distances(X[i], Y[i])
        Ds.append(D)

    D_final = np.array(Ds)

    final_shape = (axis_dim,) + batch_shape_x + batch_shape_y
    D_final = np.reshape(D_final, newshape=final_shape)
    return D_final


def test_quantile_computation():
    x = np.random.rand(2, 1000, 2)
    levels = np.random.rand(2, 2)
    levels[0, 0] = -0.2
    levels[1, 1] = 1.4
    print('levels:', levels)
    quantiles = compute_quantiles(x, axis=1, levels=levels)
    print('quantiles:', quantiles)


def test_pairwise_distance_computation():
    batch_shape_x = (729,)
    batch_shape_y = (100, 20)

    n_batch_dim_x = len(batch_shape_x)
    n_batch_dim_y = len(batch_shape_y)
    feature_dim = 2

    shape_x = batch_shape_x + (feature_dim,)
    shape_y = batch_shape_y + (feature_dim,)

    x = np.random.rand(*shape_x)
    y = np.random.rand(*shape_y)

    # scikit-learn implementation
    begin = time.time()
    D1 = compute_pairwise_distances(x, y)
    print('scikit-learn: time={:.6f}sec'.format(time.time() - begin))

    # alternative computation using based on broadcasting
    begin = time.time()
    extra_axes = tuple(range(n_batch_dim_x, n_batch_dim_x+n_batch_dim_y))
    x2 = np.expand_dims(x, axis=extra_axes)

    D2 = np.sum((x2 - y) ** 2, axis=-1) ** .5
    print('broadcasting: time={:.6f}sec'.format(time.time()-begin))
    print('max. diff. between scikit-learn & broadcasting:', np.max(np.abs(D1 - D2)))


def test_pairwise_distance_computation_along_axis():
    """
    The loop version is faster in general; may benefit from the optimized scikit-learn implementation
    """
    batch_shape_x = (729, 12)
    batch_shape_y = (100, 20, 12)

    feature_dim = 2

    shape_x = batch_shape_x + (feature_dim,)
    shape_y = batch_shape_y + (feature_dim,)

    x = np.random.rand(*shape_x)
    y = np.random.rand(*shape_y)
    axis = (1, 2)

    begin = time.time()
    D1 = compute_pairwise_distances_along_axis(x, y, axis=axis)
    print('loop: time={:.6f}sec'.format(time.time() - begin))


if __name__ == "__main__":
    test_quantile_computation()