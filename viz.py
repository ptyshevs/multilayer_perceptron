import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore', np.RankWarning)


def smooth_y(x, y, smooth_level):
    return np.poly1d(np.polyfit(x, y, smooth_level))(x)

def plot_history(h, smooth_level=0.7):
    """
    Plot loss and various metrics values as a function of epoch
    """
    x_ax = h['epoch']
    plots = []

    for k in h.keys:
        if k == 'epoch' or k.startswith('val'):
            continue
        if 'val_' + k in h:
            plots.append((k, 'val_' + k))
        else:
            plots.append((k,))

    smooth_level = smooth_level / 2 * 50
            
    n_plots = len(plots)
    f,ax  = plt.subplots(nrows=n_plots, figsize=(6 * n_plots, 5 * n_plots), squeeze=False)

    alpha = 1 if not smooth_level else .3
    for i, plot in enumerate(plots):
        label = plot[0]
        r = ax[i, 0].plot(x_ax, h[label], c='C0', label=label, alpha=alpha)
        if smooth_level:
            r[0].set_label('')
            r = ax[i, 0].plot(x_ax, smooth_y(x_ax, h[label], smooth_level), label=label)
        if len(plot) > 1:
            val_label = plot[1]
            r = ax[i, 0].plot(x_ax, h[val_label], c='orange', label=val_label, alpha=alpha)
            if smooth_level:
                r[0].set_label('')
                ax[i, 0].plot(x_ax, smooth_y(x_ax, h[val_label], smooth_level), label=val_label, c='orange')
        ax[i, 0].set_xlabel('epoch')
        ax[i, 0].set_ylabel(label)
        ax[i, 0].set_title(label)
        if len(plot) > 1:
            ax[i, 0].legend()
    plt.show()