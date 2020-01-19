from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import scipy.stats


def _scatter_and_plot(ax, data, predictions):
    xs, ys = data
    ax.scatter(xs.ravel(), ys.ravel(), s=0.5, c='black')
    ax.set_ylim([-2, 3])
    ax.set_xlim([-3, 3])
    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("y", fontsize=15)

    xs_test, ys_test = predictions
    fit = ax.plot(xs_test.ravel(), ys_test[0].ravel())[0]
    return fit


def animate_adaptive_layer(data, predictions, layer_stats, max_size, output_path=None):
    fig, axs = plt.subplots(
        1, 3,
        figsize=(20, 5),
        gridspec_kw={'width_ratios': [1, 1, 2]}
    )
    fig.subplots_adjust(wspace=0.3)

    fit = _scatter_and_plot(axs[0], data, predictions)

    domain_tn = np.linspace(1, max_size, 500)
    tn = axs[1].plot(domain_tn, np.zeros_like(domain_tn))[0]
    axs[1].set_ylim([-0.1, 4.0])
    axs[1].set_xlabel("Number of units", fontsize=15)
    axs[1].set_ylabel("Density", fontsize=15)

    locs, scales = layer_stats
    im = np.asarray([
        scipy.stats.truncnorm.pdf(
            domain_tn,
            a=(1. - loc) / scale,
            b=(max_size - loc) / scale,
            loc=loc,
            scale=scale
        ) for (loc, scale) in
        zip(locs, scales)
    ]).T
    img = axs[2].imshow(
        im,
        cmap='jet',
        aspect='auto',
        extent=[0, im.shape[1], max_size, 0],
        origin='upper'
    )
    axs[2].set_ylabel("Number of units", fontsize=15)
    axs[2].set_xlabel("Step", fontsize=15)

    xs_test, ys_test = predictions

    def update(i):
        fit.set_ydata(ys_test[i])
        density_tn = scipy.stats.truncnorm.pdf(
            domain_tn,
            a=(1. - locs[i]) / scales[i],
            b=(max_size - locs[i]) / scales[i],
            loc=locs[i],
            scale=scales[i])
        tn.set_ydata(density_tn)
        if hasattr(axs[2], "lines") and axs[2].lines:
            del axs[2].lines[-1]
        axs[2].axvline(x=i, color='white')
    #     plt.savefig(f"{output_path}/frames/{i:05d}.png", bbox_inches='tight', dpi=121)
        return fit, tn, img

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(ys_test),
        interval=50,
        blit=True,
        repeat_delay=1000
    )

    ani.save(output_path)
    return ani


def animate_skipped_layers(data, predictions, layer_stats, max_size, output_path=None):
    fig, axs = plt.subplots(
        1, 2,
        figsize=(6, 5),
        gridspec_kw={'width_ratios': [5, 1]}
    )
    fig.subplots_adjust(wspace=0.5)

    fit = _scatter_and_plot(axs[0], data, predictions)

    im = axs[1].imshow(
        layer_stats[0].reshape(layer_stats.shape[1], 1),
        cmap='binary',
        aspect='auto',
        origin='lower',
        vmin=0.,
        vmax=1.
    )
    axs[1].set_ylabel("Layer id", fontsize=15)
    axs[1].set_xlabel("Skip probability", fontsize=15)
    axs[1].set_yticks(np.arange(layer_stats.shape[1]))
    axs[1].yaxis.set_label_position("right")
    axs[1].set_xticks([])
    axs[1].xaxis.labelpad = 21
    axs[1].yaxis.tick_right()

    values = [0., 1.0]
    colors = [im.cmap(im.norm(v)) for v in values]
    # create a patch (proxy artist) for every color
    patches = [Patch(color=colors[0], label="Use"),
               Patch(color=colors[1], label="Skip")]
    # put those patched as legend-handles into the legend
    leg = axs[1].legend(
        handles=patches,
        loc='upper center',
        bbox_to_anchor=(3.18, 0.2),
        ncol=1,
        prop={'size': 15},
        facecolor="lightblue"
    )

    xs_test, ys_test = predictions

    def update(i):
        fit.set_ydata(ys_test[i])
        im.set_data(layer_stats[i].reshape(layer_stats.shape[1], 1))
        # plt.savefig(f"{output_path}/frames/{i:05d}.png", bbox_inches='tight', dpi=121)
        return fit, im, leg

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(ys_test),
        interval=50,
        blit=True,
        repeat_delay=1000
    )

    ani.save(output_path)
    return ani


def generate_1d_regression(
        n_points=2000,
        domain=(-2.0, 2.0),
        noise_std=0.1,
        seed=None):

    def f(x):
        x = np.atleast_1d(x)
        y = np.sin(6 * x) + 0.4 * x**2 - 0.1 * x**3 - x * np.cos(9 * np.sqrt(np.exp(x)))
        # y = 0.3 * x + (0.15 * x + 0.3) * np.sin(np.pi * x)
        return y

    rng = np.random.RandomState(seed)
    xs = rng.uniform(*domain, size=n_points).astype(np.float32).reshape(-1, 1)
    ys = (f(xs) + rng.normal(0.0, noise_std, size=(n_points, 1))).astype(np.float32)
    return xs, ys


def batch_generator(xs, ys, batch_size, shuffle=False, seed=None):
    """Batch generator of a labeled dataset.

    Notes:
        If len(xs) % batch_size > 0 the batches will not be of equal length
        and may even be larger than batch_size.
    """
    if len(xs) != len(ys):
        raise ValueError("Length of xs and ys is not the same.")

    rng = np.random.RandomState(seed)
    indices = np.arange(len(xs))
    if shuffle:
        rng.shuffle(indices)

    n_batches = len(xs) // batch_size
    batch_indices = np.array_split(indices, n_batches)
    for ids in batch_indices:
        yield xs[ids], ys[ids]
