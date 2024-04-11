# %5
import brian2.only as b2
import matplotlib.pyplot as plt
import numpy as np

# %%


def plot_spiking_and_stim(
    excitespikes,
    stim_t,
    stim_vals,
    stim_t_extended,
    indexes_half_radius,
    spike_vals,
    results_dir,
):
    spikes_to_keep = np.isin(excitespikes.i, indexes_half_radius)
    t_spikes_to_keep = excitespikes.t[spikes_to_keep]

    spike_counts_in_radius, bin_edges = np.histogram(
        t_spikes_to_keep / b2.ms, bins=stim_t_extended
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    # ax1.plot(stim_t[0:(len(stim_t)-1)],spike_counts)
    ax1.plot(stim_t[0 : (len(stim_t) - 1)], spike_counts_in_radius)
    ax1.set(ylabel="spikes per ms", title="Spikes within .5 mm of Probe")
    ax2.plot(stim_t[0 : (len(stim_t) - 1)], spike_vals)
    ax2.set(ylabel="spikes per ms", title="Spikes Detected by Probe")
    ax3.plot(stim_t[0 : (len(stim_t) - 1)], stim_vals)
    ax3.set(
        ylabel=r"$Irr_0$ (mm/mW$^2$)", title="Optogenetic Stimulus", xlabel="time (ms)"
    )
    plt.savefig(f"{results_dir}/spiking_and_stim.png")
    plt.close()

    fig, (ax_one) = plt.subplots(1, 1, sharex=True)
    line_radius = ax_one.plot(
        stim_t[0 : (len(stim_t) - 1)], spike_counts_in_radius, "k-"
    )
    line_detect = ax_one.plot(
        stim_t[0 : (len(stim_t) - 1)], spike_vals, "k-", color=".5"
    )
    ax_one.set(ylabel="Spikes per ms", title="Spiking Activity")
    # ax_one.legend([line_radius,line_detect],['Spikes within .5 mm of Probe','Spikes Detected by Probe'])
    ylim = ax_one.get_ylim()
    for i in range(len(stim_vals) - 1):
        if stim_vals[i] > 0:
            ax_one.plot([stim_t[i], stim_t[i + 1]], [ylim[1], ylim[1]], "b-", ms=4)

    plt.savefig(f"{results_dir}/spiking_and_stim_one_plot.png")
    plt.savefig(f"{results_dir}/spiking_and_stim_one_plot.pdf")
    plt.close()


# In[ ]:


# light.max_Irr0_mW_per_mm2_viz = 5
# ani = vv.generate_Animation(plotargs, slowdown_factor=1000, figsize=[16,12])


# In[ ]:


# from matplotlib import animation as animation

# ani.save('100x100_neurons_15ms_animation.gif')
# plt.close()

# In[ ]:


def visualize_activity_at_time(state, num_of_neurons, sampletime, ax=None):
    sidelength = (np.sqrt(num_of_neurons)).astype(int)
    timeslice = np.zeros([sidelength, sidelength])
    for i in range(sidelength):
        for j in range(sidelength):
            neuronactivity = state.v[i * sidelength + j]
            timeslice[i, j] = neuronactivity[sampletime]
    if not ax:
        fig, ax = plt.subplots()
    im = ax.imshow(timeslice, cmap=plt.get_cmap("jet"))
    ax.set(
        ylabel="y position (mm)",
        title="Action Potential at {} ms".format(sampletime),
        xlabel="X position (mm)",
    )


# In[ ]:
def visualize_spiking_per_ms(spikes, num_of_neurons, sampletime, ax=None):
    sidelength = (np.sqrt(num_of_neurons)).astype(int)
    spiketimes = spikes.t
    neuronspikes = spikes.i
    x_coordinates = []
    y_coordinates = []
    for i in range(len(spiketimes)):
        if spiketimes[i] >= sampletime * b2.ms and spiketimes[i] < (
            (sampletime + 1) * b2.ms
        ):
            x_coordinates.append(
                0.05 * ((np.floor(neuronspikes[i] / sidelength)) - sidelength / 2)
            )
            y_coordinates.append(
                0.05 * ((neuronspikes[i] % sidelength) - sidelength / 2)
            )
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x_coordinates, y_coordinates, "ks", ms=1.6)
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set(
        ylabel="y position (mm)",
        title="Spiking Activity at {} ms".format(sampletime),
        xlabel="X position (mm)",
    )


#
# Identify spiking by index
# find x y value of index and distance from probe
#
def visualize_spiking_per_ms_smoothed(
    spikes, num_of_neurons, sampletime, smooth_std, ax=None
):
    sidelength = (np.sqrt(num_of_neurons)).astype(int)
    spiketimes = spikes.t
    neuronspikes = spikes.i
    # x_coordinates=[]
    # y_coordinates=[]
    x_coordinates = np.zeros(len(spiketimes))
    y_coordinates = np.zeros(len(spiketimes))
    if ax is None:
        fig, ax = plt.subplots()
        figsize = (1.3, 1.3)
    firing_rate_smoothed = np.zeros(len(spiketimes))
    for i in range(len(spiketimes)):
        if spiketimes[i] >= (sampletime - 1.25) * b2.ms and spiketimes[i] < (
            (sampletime + 1.25) * b2.ms
        ):
            x_coordinates[i] = 0.05 * (
                (np.floor(neuronspikes[i] / sidelength)) - sidelength / 2
            )
            y_coordinates[i] = 0.05 * ((neuronspikes[i] % sidelength) - sidelength / 2)
            index_times = []
            counter = 0
            for j in range(len(spiketimes)):
                if neuronspikes[j] == neuronspikes[i]:
                    index_times.append(j)
                    counter = counter + 1
                    if i == j:
                        index_index = counter
            firing_rate_smoothed[i] = np.sum(
                (
                    2.718
                    ** (
                        -0.5
                        * (
                            (
                                np.array(spiketimes[i])
                                - np.array(spiketimes[index_times])
                            )
                            / smooth_std
                        )
                        ** 2
                    )
                )
                / (smooth_std * 2.506)
                / 1000
            )
            # ax.plot(x_coordinate,y_coordinate,'ks',ms=8*firing_rate_smoothed)#was 1.6 with 10 ms standard deviation smoothing; 3.2 with 20 ms
            # ax.plot(x_coordinate,y_coordinate, 's', ms=2, c=plt.cm.Greys(8*firing_rate_smoothed))
    plt.scatter(x_coordinates, y_coordinates, s=2, c=firing_rate_smoothed, cmap="Greys")
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set(
        ylabel="y position (mm)",
        title="Spiking Activity at {} ms".format(sampletime),
        xlabel="X position (mm)",
    )


#
# Identify spiking by index
# find x y value of index and distance from probe
#
def plot_smooth_spikes_new(spikes, num_of_neurons, sampletime, smooth_std, ax=None):
    sidelength = (np.sqrt(num_of_neurons)).astype(int)
    spiketimes = spikes.t
    neuronspikes = spikes.i
    # x_coordinates=[]
    # y_coordinates=[]
    x_coordinates = np.zeros(len(spiketimes))
    y_coordinates = np.zeros(len(spiketimes))

    firing_rate_smoothed = np.zeros(len(spiketimes))
    for i in range(len(spiketimes)):
        if spiketimes[i] >= (sampletime - 1.25) * b2.ms and spiketimes[i] < (
            (sampletime + 1.25) * b2.ms
        ):
            x_coordinates[i] = 0.05 * (
                (np.floor(neuronspikes[i] / sidelength)) - sidelength / 2
            )
            y_coordinates[i] = 0.05 * ((neuronspikes[i] % sidelength) - sidelength / 2)
            index_times = []
            counter = 0
            for j in range(len(spiketimes)):
                if neuronspikes[j] == neuronspikes[i]:
                    index_times.append(j)
                    counter = counter + 1
                    if i == j:
                        index_index = counter
            firing_rate_smoothed[i] = np.sum(
                (
                    2.718
                    ** (
                        -0.5
                        * (
                            (
                                np.array(spiketimes[i])
                                - np.array(spiketimes[index_times])
                            )
                            / smooth_std
                        )
                        ** 2
                    )
                )
                / (smooth_std * 2.506)
                / 1000
            )
            # ax.plot(x_coordinate,y_coordinate,'ks',ms=8*firing_rate_smoothed)#was 1.6 with 10 ms standard deviation smoothing; 3.2 with 20 ms
            # ax.plot(x_coordinate,y_coordinate, 's', ms=2, c=plt.cm.Greys(8*firing_rate_smoothed))
    plt.scatter(x_coordinates, y_coordinates, s=2, c=firing_rate_smoothed, cmap="Greys")
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set(
        ylabel="y position (mm)",
        title="Spiking Activity at {} ms".format(sampletime),
        xlabel="X position (mm)",
    )


def plot_all(
    excitestate,
    excitespikes,
    numofexcneur,
    stim_t,
    stim_vals,
    stim_t_extended,
    indexes_half_radius,
    spike_vals,
    results_dir,
):
    plot_spiking_and_stim(
        excitespikes,
        stim_t,
        stim_vals,
        stim_t_extended,
        indexes_half_radius,
        spike_vals,
        results_dir,
    )

    for t_samp in [2, 5, 8]:
        fig, ax = plt.subplots()
        visualize_activity_at_time(excitestate, numofexcneur, t_samp, ax)
        fig.savefig(f"{results_dir}/100by100_activity_at_{t_samp}_ms.png")

    for t_samp in [0, 4, 8, 10, 12]:
        fig, ax = plt.subplots()
        visualize_spiking_per_ms(excitespikes, numofexcneur, t_samp, ax)
        fig.savefig(f"{results_dir}/100by100_spiking_at_{t_samp}_ms.png")

    for t_sample in [0, 4, 8, 12]:
        fig, ax = plt.subplots()
        visualize_spiking_per_ms_smoothed(
            excitespikes, numofexcneur, t_sample, smooth_std=0.2, ax=ax
        )
        fig.savefig(f"{results_dir}/100by100_spiking_at_{t_sample}_ms_smoother.png")
