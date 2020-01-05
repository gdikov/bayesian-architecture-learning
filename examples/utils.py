import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import scipy.stats as sp_stats


def animate_layer(data, predictions, layer_stats, max_size, output_path=None):
    fig, axs = plt.subplots(
        1, 3,
        figsize=(20, 5),
        gridspec_kw={'width_ratios': [1, 1, 2]}
    )
    fig.subplots_adjust(wspace=0.3)

    xs, ys = data
    axs[0].scatter(xs.ravel(), ys.ravel(), s=0.5, c='black')
    axs[0].set_ylim([-2, 3])
    axs[0].set_xlim([-3, 3])
    axs[0].set_xlabel("x", fontsize=15)
    axs[0].set_ylabel("y", fontsize=15)

    xs_test, ys_test = predictions
    fit = axs[0].plot(xs_test.ravel(), ys_test[0].ravel())[0]
    domain_tn = np.arange(max(max_size, 200))
    tn = axs[1].plot(domain_tn, np.zeros_like(domain_tn))[0]
    axs[1].set_ylim([-0.1, 4.0])
    axs[1].set_xlabel("Number of units", fontsize=15)
    axs[1].set_ylabel("Density", fontsize=15)
    # axs[1].yaxis.set_label_position("right")
    # axs[1].yaxis.tick_right()

    locs, scales = layer_stats
    im = np.asarray([
        sp_stats.truncnorm.pdf(
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
    # axs[2].set_xticks([])
    # axs[2].set_yticks(np.arange(1., n_units_hidden + 1))
    # axs[1].yaxis.set_label_position("right")
    axs[2].set_ylabel("Number of units", fontsize=15)
    axs[2].set_xlabel("Step", fontsize=15)

    def update(i):
        fit.set_ydata(ys_test[i])
        density_tn = sp_stats.truncnorm.pdf(
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

    ani.save(f"{output_path}/animation.mp4")
    return ani
