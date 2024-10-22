#!/usr/bin/env python
# coding: utf-8

# In[1]:

# %%

import random as rnd
import time
from pathlib import Path

import brian2.only as b2
import cleo
import matplotlib.pyplot as plt
from brian2 import np

b2.prefs.codegen.target = "numpy"
cleo.utilities.style_plots_for_paper()

# %%
t_start = time.time()

# %%
# Lines 89-27 define default values. 9-16 are used in the generation of synapses
opto_on = True
delay_ms = 0
exp_name = "opto_on" if opto_on else "opto_off"
if opto_on:
    exp_name += f"_delay{delay_ms}ms"
results_dir = Path(f"results/{exp_name}")
if not results_dir.exists():
    results_dir.mkdir(parents=True)
generate_video = False


p0 = 0.1
unit_len = 50 * b2.um
sigma = 5 * unit_len
sigma_strong = 7 * unit_len
p1 = 1
# 14-15 are the time constants for synaptic currents
tau_ampa = 2 * b2.ms
tau_gaba = 5 * b2.ms
# these two determine the number of neurons
N_exc = 10000
N_inh = 2500
N_exc_strong = 1000
N_syn_strong = 500
g_weak = 0.1
g_strong = 2
if realistic_values := True:
    g_exc = 1 / (100 * b2.Mohm)
    g_inh = 1 / (200 * b2.Mohm)
    tau_M = 15 * b2.ms
    C_exc = tau_M * g_exc
    C_inh = tau_M * g_inh
else:
    g_exc = g_inh = b2.siemens
    C_exc = C_inh = b2.farad

# %%
# define the equations and variables for the excitatory and inhibitory neurons
exceqs = """dv/dt = (I_summed - v*g_exc)/C_exc : volt
I_summed = I_stim + I_exc_strong + I_exc_weak + I_inh + I_opto : amp
I_opto : amp
I_stim = stimulus_exc(t,i) * amp : amp
I_exc_weak : amp
I_exc_strong : amp
I_inh : amp
x = .05*((floor(i/100.0)) - 50)*mmeter : meter
y = .05*((i%100.0) - 50)*mmeter : meter
z: meter
sampled_neurons: boolean
thresh: volt"""

inheqs = """dv/dt = (I_stim + I_exc - v*g_inh)/C_inh :volt
# I_stim = stimulus_inh(t,i)*amp : amp
I_stim : amp
I_exc: amp
x = .1*((floor(i/50)) - 25) * mm : meter
y = .1*((i % 50) - 25) * mm : meter
z = .5 * mm : meter
thresh: volt"""


# NOTE: I_stim is currently set to 100 across all neurons, however this is not true for the final model, and needs to be fixed (2ms input)
ng_exc = b2.NeuronGroup(
    N_exc,
    exceqs,
    threshold="v > thresh",
    reset="v = -5*volt",
    namespace={"g_exc": 1 / (100 * b2.Mohm)},
)
ng_inh = b2.NeuronGroup(N_inh, inheqs, threshold="v > thresh", reset="v = 0*volt")

random_sampled_neurons = rnd.sample(range(0, N_exc - 1), 1000)
sampled_neurons_bool = np.zeros((N_exc), dtype=bool)
sampled_neurons_bool[random_sampled_neurons] = True
ng_exc.z = np.random.uniform(0.45, 0.55, N_exc) * b2.mm
# Initial conditions for neurons
ng_exc.v = np.random.uniform(-0.5, 1, N_exc) * b2.volt
ng_exc.thresh = np.random.uniform(0.5, 2, N_exc) * b2.volt
ng_exc.sampled_neurons = sampled_neurons_bool

ng_inh.v = np.random.uniform(-0.5, 0.5, N_inh) * b2.volt
ng_inh.thresh = np.random.uniform(0, 1, N_inh) * b2.volt

# %%
# configure initial stimulus
stim_radius = 0.5 * b2.mm
stim_arr_exc = np.zeros((2, N_exc))  # T x N
dist_from_center = np.sqrt(ng_exc.x_**2 + ng_exc.y_**2) * b2.meter
i2stim = dist_from_center < stim_radius
print(f"Number of excitatory neurons stimulated: {np.sum(i2stim)}")

stim_level = 675
stim_arr_exc[0, i2stim] = stim_level
print(f"stimulating at {stim_level} amps")

stimulus_exc = b2.TimedArray(stim_arr_exc, dt=2 * b2.ms)


# stim_arr_inh = np.zeros((2, N_inh))  # T x N
# dist_from_center = np.sqrt(inhneurons.x_**2 + inhneurons.y_**2) * b2.meter
# i2stim = dist_from_center < stim_radius
# print(f"Number of inhibitory neurons stimulated: {np.sum(i2stim)}")

# stim_arr_inh[0, i2stim] = 500

# stimulus_inh = b2.TimedArray(stim_arr_inh, dt=2 * b2.ms)


# %%
# synapses
syn_model = """
    ds/dt = -s/tau : amp (clock-driven)
    I_NAME_post= w / N_incoming * s : amp (summed)
"""
on_pre = "s += 0.5*amp"
# original paper used Manhattan distance
# p="p0*exp(-abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post) / sigma**2)",
# Euclidean distance instead:
connect_prob = "p0 * exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2) / sigma**2)"

syn_e2e_weak = b2.Synapses(
    ng_exc,
    ng_exc,
    syn_model.replace("NAME", "exc_weak"),
    on_pre=on_pre,
    namespace={"w": 40000, "tau": tau_ampa},
)
syn_e2e_weak.connect(
    condition="i!=j",
    p=connect_prob,
)
print(f"Number of weak synapses: {len(syn_e2e_weak)}")

# %%
# NOTE: why do they hardcode in a 0.1 vs. 2 factor for weak vs strong? do they have the same base weight?
# 90-96 defines the strong excitatory synapses, note that the total number needs to be capped at 500 to replicate the paper
i_strong_neurons = np.random.choice(range(N_exc), N_exc_strong, replace=False)
x_strong = ng_exc.x[i_strong_neurons]
y_strong = ng_exc.y[i_strong_neurons]
p_strong_connect = p1 * np.exp(
    (
        (10 * unit_len) ** 2
        - (x_strong - x_strong[:, None]) ** 2
        - (y_strong - y_strong[:, None]) ** 2
    )
    / sigma_strong**2
)
np.fill_diagonal(p_strong_connect, 0)
assert p_strong_connect.shape == (N_exc_strong, N_exc_strong)
strong_connections = np.random.rand(N_exc_strong, N_exc_strong) < p_strong_connect
i_syn_strong, j_syn_strong = np.where(strong_connections)
j_syn_strong
idx_strong_to_keep = np.random.choice(
    range(len(i_syn_strong)), N_syn_strong, replace=False
)
assert len(idx_strong_to_keep) == N_syn_strong
i_syn_strong = i_syn_strong[idx_strong_to_keep]
j_syn_strong = j_syn_strong[idx_strong_to_keep]
assert len(i_syn_strong) == N_syn_strong
# map back from index among strong neurons to index among all neurons
i_syn_strong_orig = i_strong_neurons[i_syn_strong]
j_syn_strong_orig = i_strong_neurons[j_syn_strong]
assert len(i_syn_strong_orig) == len(j_syn_strong_orig) == N_syn_strong


# %%
syn_e2e_strong = b2.Synapses(
    ng_exc,
    ng_exc,
    syn_model.replace("NAME", "exc_strong"),
    on_pre=on_pre,
    namespace={"w": 80000, "tau": tau_ampa},
)
syn_e2e_strong.connect(i=i_syn_strong_orig, j=j_syn_strong_orig)
print(f"Number of strong synapses (should be 500): {len(syn_e2e_strong)}")
# sexcstrong.w = 1;

# %%

syn_i2e = b2.Synapses(
    ng_inh,
    ng_exc,
    syn_model.replace("NAME", "inh"),
    on_pre=on_pre,
    namespace={"w": -10000000, "tau": tau_gaba},
)
syn_i2e.connect(
    condition="i!=j",
    p=connect_prob,
)

syn_e2i = b2.Synapses(
    ng_exc,
    ng_inh,
    syn_model.replace("NAME", "exc"),
    on_pre=on_pre,
    namespace={"w": 10000000, "tau": tau_ampa},
)
syn_e2i.connect(
    condition="i!=j",
    p=connect_prob,
)

# exc_st_mon = b2.StateMonitor(ng_exc, ["v", "I_stim", "Isynext"], record=True)
exc_st_mon = b2.StateMonitor(ng_exc, ["v"], record=True)
exc_spike_mon = b2.SpikeMonitor(ng_exc)

# exciteneurons = SpikeGeneratorGroup(N_exc, indices, times)

net = b2.Network(
    ng_exc,
    ng_inh,
    syn_e2e_weak,
    syn_e2e_strong,
    syn_e2i,
    syn_i2e,
    exc_st_mon,
    exc_spike_mon,
)


# In[3]:
simple_opsin = cleo.opto.ProportionalCurrentOpsin(
    name="simple_opsin",
    # handpicked gain to make firing rate roughly comparable to EIF
    I_per_Irr=-2400000 * b2.amp / (b2.mwatt / b2.mm2),
)
fiber = cleo.light.Light(
    coords=(1.75, 1.75, 0) * b2.mm,
    light_model=cleo.light.fiber473nm(R0=0.2 * b2.mm),
    name="fiber",
)

spikes = cleo.ephys.SortedSpiking(
    name="spikes",
    r_perfect_detection=25 * b2.um,
    r_half_detection=50 * b2.um,
)
probe = cleo.ephys.Probe([1.75, 1.75, 0.5] * b2.mm)
probe.add_signals(spikes)

plotargs = {
    "colors": ["xkcd:fuchsia"],
    "xlim": (-2.5, 2.5),
    "ylim": (-2.5, 2.5),
    "zlim": (0, 1),
    "scatterargs": {"s": 1},
    "axis_scale_unit": b2.mm,
}

fig, ax = cleo.viz.plot(
    ng_exc,
    # ng_inh,
    **plotargs,
    devices=[probe, fiber],
)
fig.savefig(f"{results_dir}/100x100_static_setup.png")

sim = cleo.CLSimulator(net)
# sim.inject_stimulator(opto, exciteneurons, Iopto_var_name='I_opto')
sim.inject(simple_opsin, ng_exc, Iopto_var_name="I_opto")
sim.inject(fiber, ng_exc)
sim.inject(probe, ng_exc)


stim_vals = []
stim_t = []
spike_vals = []


class ReactiveLoopOpto(cleo.ioproc.LatencyIOProcessor):
    def __init__(self):
        super().__init__(sample_period=0.2 * b2.ms)

    # since this is open-loop, we don't use state_dict
    def process(self, state_dict, t_samp):
        i, t, z_t = state_dict["Probe"]["spikes"]
        if np.size(i) >= 3:
            if opto_on:
                opto_intensity = 0.15
            else:
                opto_intensity = 0
        else:
            opto_intensity = 0
        stim_vals.append(opto_intensity)
        stim_t.append(t_samp / b2.ms)
        spike_vals.append(np.size(i))
        opto_intensity *= b2.mwatt / b2.mm2
        # return output dict and time
        return ({"fiber": opto_intensity}, t_samp + delay_ms * b2.ms)


sim.set_io_processor(ReactiveLoopOpto())

if generate_video:
    vv = cleo.viz.VideoVisualizer(dt=0.5 * b2.ms, devices_to_plot=[probe, fiber])
    sim.inject(vv, ng_exc)


# %%


sim.run(15 * b2.ms)

# fig, ax = plt.subplots()
# ax.plot(excitespikes.t/msecond, excitespikes.i, '|',ms=.2,lw=1)
# ax.set(ylabel='neuron index', title='spiking')
# plt.savefig('spiking_100x100_15ms_v43.png')
# plt.close()
# print(sim.network.t)
t_end = time.time()
print(f"Time elapsed: {t_end - t_start:.1f} seconds")

# %%

stim_t_extended = np.copy(stim_t)
stim_t_extended = np.append(stim_t_extended, np.max(stim_t) + stim_t[1] - stim_t[0])
spike_counts, bin_edges = np.histogram(exc_spike_mon.t / b2.ms, bins=stim_t_extended)

indexes_half_radius = []

for i in range(N_exc):
    if (
        (
            (0.05 * ((np.floor(i / 100.0)) - 50) * b2.mm - 1.75 * b2.mm) ** 2
            + (0.05 * ((i % 100.0) - 50) * b2.mm - 1.75 * b2.mm) ** 2
        )
        ** (0.5)
    ) < 0.5 * b2.mm:
        indexes_half_radius.append(i)

# %%
np.savez_compressed(
    results_dir / "data.npz",
    t_spk_ms=exc_spike_mon.t / b2.ms,
    i_spk=exc_spike_mon.i,
    stim_t=stim_t,
    stim_vals=stim_vals,
    fiber_t=np.array(fiber.t / b2.ms),
    fiber_vals=np.array(fiber.values).squeeze(),
    spike_counts=spike_counts,
    bin_edges=bin_edges,
    spike_vals=spike_vals,
    stim_t_extended=stim_t_extended,
    indexes_half_radius=indexes_half_radius,
    numofexcneur=N_exc,
    exc_x_mm=ng_exc.x / b2.mm,
    exc_y_mm=ng_exc.y / b2.mm,
)

# %%
# plot results (old)
# from plot_single_expt_old import plot_all

# plot_all(
#     excitestate,
#     excitespikes,
#     N_exc,
#     stim_t,
#     stim_vals,
#     stim_t_extended,
#     indexes_half_radius,
#     spike_vals,
#     results_dir,
# )

# %%
# plot results
from plot_single_expt import plot_all

plot_all(results_dir)

# %%
if generate_video:
    from matplotlib import animation as animation

    fiber.max_Irr0_mW_per_mm2_viz = 5
    ani = vv.generate_Animation(plotargs, slowdown_factor=1000, figsize=[16, 12])
    ani.save(f"{results_dir}/100x100_neurons_15ms_animation.gif")
    plt.close()
