import numpy as np
import pickle
import gzip


def percentiles(values, bin_values, bin_edges, percentile):
    # Seems complicated for vector defined as [inf, inf, .., inf]
    percentiles_binned = np.squeeze(
        np.full((len(bin_edges) - 1, len(values.shape)), np.inf)
    )
    err_percentiles_binned = np.squeeze(
        np.full((len(bin_edges) - 1, len(values.shape)), np.inf)
    )
    for i, (bin_l, bin_h) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        try:
            print(i)
            print(bin_l)
            print(bin_h)
            distribution = values[(bin_values > bin_l) & (bin_values < bin_h)]
            percentiles_binned[i] = np.percentile(distribution, percentile)
            print(percentiles_binned[i])
            err_percentiles_binned[i] = percentiles_binned[i] / np.sqrt(
                len(distribution)
            )
        except IndexError:
            pass
    return percentiles_binned.T, err_percentiles_binned.T


def plot_hist(ax, data, edges, norm=False, yerr=False, hist_kwargs=None, error_kw=None):
    """Utility function to plot histogram"""

    hist_kwargs = hist_kwargs or {}
    error_kw = error_kw or {}

    weights = np.ones_like(data)
    if norm is True:
        weights = weights / float(np.sum(data))
    if yerr is True:
        yerr = np.sqrt(data) * weights
    else:
        yerr = np.zeros(len(data))

    centers = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1:] - edges[:-1]
    ax.bar(
        centers,
        data * weights,
        width=width,
        yerr=yerr,
        error_kw=error_kw,
        **hist_kwargs
    )

    return ax


def save_obj(obj, name):
    """Save object in binary"""
    with gzip.open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """Load object in binary"""
    with gzip.open(name, "rb") as f:
        return pickle.load(f)
