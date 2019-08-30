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

    f,ax  = plt.subplots(nrows=len(plots), figsize=(10, 10))

    for i, plot in enumerate(plots):
        label = plot[0]
        ax[i].plot(x_ax, h[label], label=label)
        if len(plot) > 1:
            val_label = plot[1]
            ax[i].plot(x_ax, h[val_label], label=val_label)
        ax[i].set_xlabel('epoch')
        ax[i].set_ylabel(label)
        ax[i].set_title(label)
        if len(plot) > 1:
            ax[i].legend()
    plt.show()