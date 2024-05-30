# %%
import cleo
import matplotlib.pyplot as plt
import numpy as np

cleo.utilities.style_plots_for_paper()

# %%
data_opto_on = np.load("results/opto_on/data.npz")
data_opto_off = np.load("results/opto_off/data.npz")

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

t_samps = [0, 4, 8, 12]

fig, axs = plt.subplots(
    2, len(t_samps), figsize=(5.75, 4), sharex=True, sharey=True, layout="compressed"
)
cax = fig.add_axes([1.02, 0.2, 0.02, 0.6])

# get same vlim for all plots
vlim = (
    min(ss_opto_on.min(), ss_opto_off.min()),
    max(ss_opto_on.max(), ss_opto_off.max()),
)

for ax, t_samp in zip(axs[0], t_samps):
    dt = np.min(np.diff(np.unique(data_opto_on["t_spk_ms"])))
    im = plot_smooth_spikes_new(ss_opto_off, dt, t_samp, ax, vlim)
    ax.set(title=f"{t_samp} ms")
for ax, t_samp in zip(axs[1], t_samps):
    dt = np.min(np.diff(np.unique(data_opto_off["t_spk_ms"])))
    im = plot_smooth_spikes_new(ss_opto_on, dt, t_samp, ax, vlim)
    ax.set(xlabel="x [mm]", title="")

cbar = fig.colorbar(im, cax, label="smoothed spikes", aspect=20, ticks=[])
axs[0, 0].set(ylabel="y [mm]")
axs[1, 0].set(ylabel="y [mm]")
# fig.text(0.5, 0.9, "Without optogenetic stimulation", ha="center", va="top")
# fig.text(0.5, 0.5, "With optogenetic stimulation", ha="center")
row_title_loc = (-0.3, 1)
axs[0, 0].text(
    *row_title_loc, "Opto off", ha="right", va="top", transform=axs[0, 0].transAxes
)
axs[1, 0].text(
    *row_title_loc, "Opto on", ha="right", va="top", transform=axs[1, 0].transAxes
)
fig.savefig(f"results/spiking_comparison.svg")
fig.savefig(f"results/spiking_comparison.pdf")

# %%
