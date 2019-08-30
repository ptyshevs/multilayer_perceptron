import numpy as np
import matplotlib.pyplot as plt

def plot_history(h):
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

    n_plots = len(plots)
    f,ax  = plt.subplots(nrows=n_plots, figsize=(6 * n_plots, 5 * n_plots), squeeze=False)

    for i, plot in enumerate(plots):
        label = plot[0]
        ax[i, 0].plot(x_ax, h[label], label=label)
        if len(plot) > 1:
            val_label = plot[1]
            ax[i, 0].plot(x_ax, h[val_label], label=val_label)
        ax[i, 0].set_xlabel('epoch')
        ax[i, 0].set_ylabel(label)
        ax[i, 0].set_title(label)
        if len(plot) > 1:
            ax[i, 0].legend()
    plt.show()