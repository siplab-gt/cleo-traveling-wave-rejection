# %5
import brian2.only as b2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


def plot_spiking_and_stim_separate(data, axs=None):
    spikes_to_keep = np.isin(data["i_spk"], data["indexes_half_radius"])
    t_spikes_to_keep = data["t_spk_ms"][spikes_to_keep]

    spike_counts_in_radius, bin_edges = np.histogram(
        t_spikes_to_keep / b2.ms, bins=data["stim_t_extended"]
    )

    if axs is None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    else:
        ax1, ax2, ax3 = axs

    stim_t = data["stim_t"]

    ax1.plot(stim_t[:-1], spike_counts_in_radius)
    ax1.set(ylabel="spikes per ms", title="Spikes within .5 mm of Probe")
    ax2.plot(stim_t[:-1], data["spike_vals"])
    ax2.set(ylabel="spikes per ms", title="Spikes Detected by Probe")
    ax3.plot(stim_t[:-1], data["stim_vals"])
    ax3.set(
        ylabel=r"$Irr_0$ (mm/mW$^2$)", title="Optogenetic Stimulus", xlabel="time (ms)"
    )


def plot_spiking_and_stim_together(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # t_spk_detected = data["t_spk_ms_detected"]

    # t_spk_binned, bin_edges = np.histogram(t_spk_detected, bins=data["stim_t_extended"])
    stim_t = data["stim_t"]
    stim_vals = data["stim_vals"]

    # line_radius = ax.plot(stim_t[:-1], t_spk_binned, "k-")
    # line_detect = ax.plot(stim_t[:-1], data["spike_vals"], "k-", color=".5")
    ax.plot(stim_t, data["spike_vals"], "k-")
    ax.set(
        ylabel="spikes/sample", title="Spikes detected at electrode", xlabel="time [ms]"
    )
    # ax_one.legend([line_radius,line_detect],['Spikes within .5 mm of Probe','Spikes Detected by Probe'])
    ylim = ax.get_ylim()
    for i in range(len(stim_vals) - 1):
        if stim_vals[i] > 0:
            ax.plot([stim_t[i], stim_t[i + 1]], [ylim[1], ylim[1]], c="#72b5f2", ms=4)


def plot_smooth_spikes_new(data, sampletime, smooth_std, ax=None):
    i_spk, t_spk_ms = data["i_spk"], data["t_spk_ms"]

    dt = np.min(np.diff(np.unique(data["t_spk_ms"])))
    T = int(np.max(np.unique(data["t_spk_ms"])) / dt) + 1
    spikes = np.zeros((data["numofexcneur"], T))
    t_spk_index = np.round(t_spk_ms / dt).astype(int)
    spikes[data["i_spk"], t_spk_index] = 1

    x_spk_mm = data["exc_x_mm"][i_spk]
    y_spk_mm = data["exc_y_mm"][i_spk]

    smoothed_spikes_all = gaussian_filter1d(spikes, smooth_std / dt, axis=1)
    # since the firing distribution is right-skewed, plot log instead
    smoothed_spikes_all = np.log(smoothed_spikes_all + 1e-3)

    sampletime_index = int(sampletime / dt)
    smoothed_spikes_samp = smoothed_spikes_all[:, sampletime_index]

    sidelength = int(np.sqrt(data["numofexcneur"]))
    smoothed_spikes_img = smoothed_spikes_samp.reshape((sidelength, sidelength))

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # ax.hist(np.log(smoothed_spikes_samp + 1e-3), color="k", alpha=0.5)
    cmap = sns.light_palette("#8000B4", as_cmap=True)
    im = ax.imshow(
        smoothed_spikes_img,
        extent=[-2.5, 2.5, -2.5, 2.5],
        interpolation="nearest",
        origin="lower",
        cmap=cmap,
        vmin=smoothed_spikes_all.min(),
        vmax=smoothed_spikes_all.max(),
    )
    # ax.scatter(x_spk_mm, y_spk_mm, s=2, c=firing_rate_smoothed, cmap="Greys")
    ax.set(
        xlabel="x [mm]",
        title=f"{sampletime} ms",
    )
    return im


def plot_all(results_dir):
    data = np.load(f"{results_dir}/data.npz")

    fig, ax = plt.subplots(figsize=(5.5, 1.25), layout="constrained")
    plot_spiking_and_stim_together(data, ax)
    fig.savefig(f"{results_dir}/spiking_and_stim.svg")

    t_samps = [0, 4, 8, 12]
    fig, axs = plt.subplots(1, len(t_samps), figsize=(5.5, 2), layout="constrained")
    for ax, t_samp in zip(axs, t_samps):
        smooth_std = 0.8
        im = plot_smooth_spikes_new(data, t_samp, smooth_std, ax)
    cbar = fig.colorbar(im, label="log(smoothed spikes)", aspect=20, ticks=[])
    axs[0].set(ylabel="y [mm]")
    fig.savefig(f"{results_dir}/spiking_over_time_smoothed.svg")
    fig.savefig(f"{results_dir}/spiking_over_time_smoothed.pdf")
