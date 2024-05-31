# %%
import cleo
import matplotlib.pyplot as plt
import numpy as np

cleo.utilities.style_plots_for_paper()
# mpl.rc_file_defaults()

# %%
data_opto_on = np.load("results/opto_on_delay0ms/data.npz")
data_opto_off = np.load("results/opto_off/data.npz")
data_delay = np.load("results/opto_on_delay3ms/data.npz")

light_473nm = "#72b5f2"
light_473nm_dark = "#265a82"

# %%
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(5.75, 1.35), layout="constrained")
# line_radius=ax_one.plot(stim_t[0:(len(stim_t)-1)],spike_counts_in_radius,'k-')
ax.plot(
    data_opto_off["stim_t"],
    data_opto_off["spike_vals"],
    "-",
    color=".5",
    label="without opto",
)
ax.plot(
    data_opto_on["stim_t"],
    data_opto_on["spike_vals"],
    c=light_473nm_dark,
    label="with opto",
)

ylim = ax.get_ylim()

stim_on = (np.array(data_opto_on["stim_vals"]) > 0) * ylim[1]
# stim_on = np.copy(data_opto_on["spike_vals"]).astype(float)
stim_on[stim_on == 0] = np.nan
# stim_on[np.array(data_opto_on["stim_vals"]) == 0] = np.nan
opto_line = ax.step(
    data_opto_on["stim_t"], stim_on, where="post", color=light_473nm, label="light on"
)
# opto_line = ax.plot(
#     data_opto_on["stim_t"], stim_on, color=light_473nm, label="light on"
# )

# opto_line[0].set()

ax.axhline(2, color="k", linestyle="--", label="stim threshold", lw=1)

ax.set(ylabel="spikes/sample", xlabel="time [ms]", title="Spikes detected at electrode")
ax.legend(loc="best")
plt.rcParams["svg.fonttype"] = "none"
plt.savefig("results/spiking_and_stim_opto_compare.svg")

# %%
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


def plot_smooth_spikes_new(smoothed_spikes_all, dt, sampletime, ax=None, vlim=None):
    sampletime_index = int(sampletime / dt)
    smoothed_spikes_samp = smoothed_spikes_all[:, sampletime_index]

    sidelength = int(np.sqrt(smoothed_spikes_all.shape[0]))
    smoothed_spikes_img = smoothed_spikes_samp.reshape((sidelength, sidelength))

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # ax.hist(np.log(smoothed_spikes_samp + 1e-3), color="k", alpha=0.5)
    if not vlim:
        vlim = (smoothed_spikes_all.min(), smoothed_spikes_all.max())
    cmap = sns.light_palette("#8000B4", as_cmap=True)
    # cmap = "Greys"
    im = ax.imshow(
        smoothed_spikes_img,
        extent=[-2.5, 2.5, -2.5, 2.5],
        interpolation="nearest",
        origin="lower",
        cmap=cmap,
        vmin=vlim[0],
        vmax=vlim[1],
    )
    return im


smooth_std = 0.8


def spikes_from_data(data):
    i_spk, t_spk_ms = data["i_spk"], data["t_spk_ms"]

    dt = np.min(np.diff(np.unique(t_spk_ms)))
    T = int(np.max(np.unique(t_spk_ms)) / dt) + 1
    spikes = np.zeros((data["numofexcneur"], T))
    t_spk_index = np.round(t_spk_ms / dt).astype(int)
    spikes[data["i_spk"], t_spk_index] = 1

    smoothed_spikes_all = gaussian_filter1d(spikes, smooth_std / dt, axis=1)
    # since the firing distribution is right-skewed, plot log instead
    # smoothed_spikes_all = np.log(smoothed_spikes_all + 1e-3)
    return smoothed_spikes_all


ss_opto_on = spikes_from_data(data_opto_on)
ss_opto_off = spikes_from_data(data_opto_off)
ss_opto_delay = spikes_from_data(data_delay)

t_samps = [0, 4, 8, 12]
t_samps = [0, 3, 6, 9, 12]

fig, axs = plt.subplots(
    5,
    len(t_samps),
    height_ratios=[1, 1, 0.2, 1, 0.2],
    figsize=(5.75, 6),
    sharex=True,
    sharey=True,
    layout="compressed",
)
cax = fig.add_axes([1.02, 0.2, 0.02, 0.6])

# get same vlim for all plots
vlim = (
    min(ss_opto_on.min(), ss_opto_off.min()),
    max(ss_opto_on.max(), ss_opto_off.max()),
)

dt = np.min(np.diff(np.unique(data_opto_off["t_spk_ms"])))
for ss, row, label in [
    (ss_opto_off, 0, "Control off"),
    (ss_opto_on, 1, "Control on"),
    (ss_opto_delay, 3, "Control on\n+3 ms delay"),
]:
    for ax, t_samp in zip(axs[row], t_samps):
        im = plot_smooth_spikes_new(ss, dt, t_samp, ax, vlim)
        sns.despine(ax=ax, bottom=True, left=True)
        ax.set(xticks=[], yticks=[])
    axs[row, 0].text(
        -0.1, 1, label, ha="right", va="top", transform=axs[row, 0].transAxes
    )

for ax, t_samp in zip(axs[0], t_samps):
    ax.set(title=f"{t_samp} ms")

cbar = fig.colorbar(im, cax, label="smoothed spikes", aspect=20, ticks=[])


### clear out thin rows, add subfigures for stim
### from https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subfigures.html
gridspec = axs[2, 0].get_subplotspec().get_gridspec()

for a in np.concatenate([axs[2], axs[4]]):
    a.remove()

for data, row in [(data_opto_on, 2), (data_delay, 4)]:
    # make the subfigure in the empty gridspec slots:
    subfig = fig.add_subfigure(gridspec[row, :])
    ax_stim = subfig.add_axes([0, 0.6, 1, 0.3])
    sns.despine(ax=ax_stim, left=True)
    ax_stim.set(
        xlim=(0, 15),
        ylim=(0, 1),
        xticks=t_samps + [15],
        xticklabels=["0 ms", "", "", "", "", "15 ms"],
        yticks=[],
    )
    ax_stim.tick_params(direction="in")
    fiber_vals = data["fiber_vals"].copy()
    fiber_t = data["fiber_t"].copy()
    # fiber_vals[fiber_vals == 0] = np.nan
    starts = fiber_t[np.where(np.diff(fiber_vals) > 0)[0]]
    ends = fiber_t[np.where(np.diff(fiber_vals) < 0)[0]]
    # ax_stim.step(data["fiber_t"], data["fiber_vals"], where="post", color=light_473nm)
    if len(starts) > len(ends):
        ends = np.concatenate([ends, [fiber_t[-1]]])
    for start, end in zip(starts, ends):
        poly = ax_stim.axvspan(
            start,
            end,
            color=light_473nm,
            label="stimulation triggered when ≥3 spikes detected",
        )

ax_stim.legend(
    handles=[poly], loc="upper center", bbox_to_anchor=(0.5, 0), frameon=False
)

fig.savefig(f"results/spiking_comparison.svg")
# fig.savefig(f"results/spiking_comparison.pdf")

# %%
