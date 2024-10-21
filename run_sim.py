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


po = 0.1
alpha = 5
p1 = 1
alpha1 = 7
# 14-15 are the time constants for synaptic currents
tauampa = 2 * b2.ms
taugaba = 5 * b2.ms
ex = 2.71828
# these two determine the number of neurons
numofexcneur = 10000
numofinhneur = 2500
excstartingval = np.random.uniform(-0.5, 0.5, numofexcneur)
inhstartingval = np.random.uniform(-0.5, 0, numofinhneur)
excreset = -5 * b2.volt
inhreset = 0 * b2.volt
g = 0.1
# i is the index of a neuron
# "The strengths of the connections were selected to balance the averaged excitatory current with the averaged inhibitory current in network"
# 37-38 show the structure of the recurrent currents, however this is not complete and need to be fixed
# Isynext_equations = '''Isynext = g/n(sum(se*exciteneurons.w)+1/ni*sum(si*inhneurons.w)) : amp'''
# Isyninh_equations = '''1/n*sum(se*we) : amp'''

# sum() is handled by (summed)
# ni is handled N_incoming
# N_outgoing (?)

# %%
# define the equations and variables for the excitatory and inhibitory neurons
exceqs = """dv/dt = (I_summed - v/gexc)/C : volt
C: farad 
I_summed=Iext+Isynext+Isynweakext+Isyninh+I_default+I_opto: amp
I_opto : amp
Iext = stimulus_exc(t,i)*amp: amp
I_default: amp
Isynext: amp
Isyninh: amp
Isynweakext: amp
gexc: ohm
x = .05*((floor(i/100.0)) - 50)*mmeter : meter
y = .05*((i%100.0) - 50)*mmeter : meter
z: meter
sampled_neurons: boolean
excthreshold: volt"""

inheqs = """dv/dt = (Iext + Isynext - v/ginh)/C :volt
C: farad
# Iext = stimulus_inh(t,i)*amp : amp
Iext : amp
Isynext: amp
Isyninh : amp
ginh: ohm
x: meter
y: meter
z: meter
excthreshold: volt"""

resetv = "v =  excreset"
excthreshold = "v > excthreshold"
# NOTE: Iext is currently set to 100 across all neurons, however this is not true for the final model, and needs to be fixed (2ms input)
exciteneurons = b2.NeuronGroup(
    numofexcneur, exceqs, threshold=excthreshold, reset=resetv
)
inhneurons = b2.NeuronGroup(
    numofinhneur, inheqs, threshold=excthreshold, reset="v = 0*volt"
)

random_sampled_neurons = rnd.sample(range(0, numofexcneur - 1), 1000)
sampled_neurons_bool = np.zeros((numofexcneur), dtype=bool)
sampled_neurons_bool[random_sampled_neurons] = True
# exciteneurons.x = '2*((floor(i/50.0)) - 50)' #value for 2500 exc neuron
# exciteneurons.y = '2*((i%50.0) - 50)' #value for 2500 exc neurons
# exciteneurons.x = '.1*1*((floor(i/10.0)) - 5)*mmeter' #value for 100 exc neuron
# exciteneurons.y = '.1*1*((i%10.0) - 5)*mmeter' #value for 100 exc neurons
# exciteneurons.x = '.1*((floor(i/100.0)) - 50)*mmeter' #value for 10000 exc neuron
# exciteneurons.y = '.1*((i%100.0) - 50)*mmeter' #value for 10000 exc neurons
exciteneurons.z = np.random.uniform(0.45, 0.55, numofexcneur) * b2.mm
# Initial conditions for neurons
exciteneurons.v = excstartingval * b2.volt
exciteneurons.C = np.ones(numofexcneur) * 1 * b2.farad
exciteneurons.gexc = np.ones(numofexcneur) * (0.1) * b2.ohm
# exciteneurons.Iext = np.ones(numofexcneur)*100*amp;
# exciteneurons.Iext = 'stimulus_exc(t,i)*amp'
# exciteneurons.I_default = np.ones(numofexcneur)*100*amp;
exciteneurons.excthreshold = np.random.uniform(0.5, 2, numofexcneur) * b2.volt
exciteneurons.sampled_neurons = sampled_neurons_bool

# inhneurons.x = '8*((floor(i/12.5)) - 50)' #value for 625 inh neurons
# inhneurons.y = '8*((i % 12.5) - 50)' #value for 625 inh neurons
inhneurons.x = ".1*((floor(i/50)) - 25)*mmeter"
inhneurons.y = ".1*((i % 50) - 25)*mmeter"
inhneurons.z = ".5*mmeter"
inhneurons.v = inhstartingval * b2.volt
inhneurons.C = np.ones(numofinhneur) * 1 * b2.farad
inhneurons.ginh = np.ones(numofinhneur) * (0.2) * b2.ohm
# inhneurons.Iext = np.ones(numofinhneur)*100*amp;
# stimulus_inh = TimedArray(np.hstack([[0, 0, c, 0, 0]
#                                 for c in np.random.rand(numofinhneur)]),
#                                dt=2*ms)
# inhneurons.Iext = 'stimulus_inh(t)*amp'
inhneurons.excthreshold = np.random.uniform(0, 1, numofinhneur) * b2.volt

# %%
# configure initial stimulus
stim_radius = 0.5 * b2.mm
stim_arr_exc = np.zeros((2, numofexcneur))  # T x N
dist_from_center = np.sqrt(exciteneurons.x_**2 + exciteneurons.y_**2) * b2.meter
i2stim = dist_from_center < stim_radius
print(f"Number of excitatory neurons stimulated: {np.sum(i2stim)}")

stim_level = 675
stim_arr_exc[0, i2stim] = stim_level
print(f"stimulating at {stim_level} amps")

stimulus_exc = b2.TimedArray(stim_arr_exc, dt=2 * b2.ms)


# stim_arr_inh = np.zeros((2, numofinhneur))  # T x N
# dist_from_center = np.sqrt(inhneurons.x_**2 + inhneurons.y_**2) * b2.meter
# i2stim = dist_from_center < stim_radius
# print(f"Number of inhibitory neurons stimulated: {np.sum(i2stim)}")

# stim_arr_inh[0, i2stim] = 500

# stimulus_inh = b2.TimedArray(stim_arr_inh, dt=2 * b2.ms)


# %%
# synapses
# 84-88 defines the weak excitatory synapses
sexcweak = b2.Synapses(
    exciteneurons,
    exciteneurons,
    """w=40000 : 1
dse/dt = -se/tauampa : amp (clock-driven)
Isynweakext_post= g/N_incoming*se*w : amp (summed)
""",
    on_pre="se = se + .5*amp",
)
sexcweak.connect(
    condition="i!=j",
    # original paper used Manhattan distance
    # p="""po*exp((-abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post)/(.05*mmeter))/((alpha**2)))""",
    # Euclidean distance instead:
    p="""po*exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / (.05*mm * alpha**2))""",
)
# sexcweak.w = .2;

# 90-96 defines the strong excitatory synapses, note that the total number needs to be capped at 500 to replicate the paper
sexcstrong = b2.Synapses(
    exciteneurons,
    exciteneurons,
    """w=80000 : 1
dse/dt = -se/tauampa: amp (clock-driven)
Isynext_post = w*g/N_incoming*se : amp (summed)
""",
    on_pre="se = se + .5*amp",
)

sexcstrong.connect(
    condition="i!=j and sampled_neurons_pre and sampled_neurons_post",
    # p="""p1*exp((100-(abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post)/(.05*mmeter)))/((alpha1**2)))""",
    p="""p1*exp((100 - sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / (.05*mm)) / alpha1**2)""",
)

# sexcstrong.w = 1;

# 99-113 defines the inhibitory to excitatory, excitatory to inhibitory, and inhibitory-inhibitory synapses
inhtoexcsynapse = b2.Synapses(
    inhneurons,
    exciteneurons,
    """w=-10000000 : 1
dsi/dt = -si/taugaba : amp (clock-driven)
Isyninh_post = w*g/N_incoming/N_outgoing*si : amp (summed)
""",
    on_pre="si = si + .5*amp",
)

inhtoexcsynapse.connect(
    condition="i!=j",
    # p="""po*exp((-abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post)/(.05*mmeter))/((alpha**2)))""",
    p="""po*exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / (.05*mm * alpha**2))""",
)

exctoinhsynapse = b2.Synapses(
    exciteneurons,
    inhneurons,
    """w=10000000 : 1
dsi/dt = -si/taugaba : amp (clock-driven)
Isynext_post = w*g/N_incoming/N_outgoing*si : amp (summed)
""",
    on_pre="si = si + .5*amp",
)

exctoinhsynapse.connect(
    condition="i!=j",
    # p="""po*exp((-abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post)/(.05*mmeter))/((alpha**2)))""",
    p="""po*exp(-sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) / (.05*mm * alpha**2))""",
)

# exctoinhsynapse.w = 2; #this needs to be changed

# a starting point to define the synaptic currents (S7), please not that this is a state variable of excite neurons
# exciteneurons.Isynext = 'g/numofexcneur*(sum(sexcstrong[:, j].se*sexcstrong[:, j].w)+1/numofinhneur*sum(inhneurons[:, j].si*inhneurons[:, j].w)) '

excitestate = b2.StateMonitor(exciteneurons, ["v", "Iext", "Isynext"], record=True)
excitespikes = b2.SpikeMonitor(exciteneurons)

# exciteneurons = SpikeGeneratorGroup(numofexcneur, indices, times)

net = b2.Network(
    exciteneurons,
    inhneurons,
    sexcweak,
    sexcstrong,
    exctoinhsynapse,
    inhtoexcsynapse,
    excitestate,
    excitespikes,
)


# In[27]:


incoming_connections = sexcweak.N_incoming_post
np.delete(incoming_connections, np.where(incoming_connections == 0))


# In[3]:
simple_opsin = cleo.opto.ProportionalCurrentOpsin(
    name="simple_opsin",
    # handpicked gain to make firing rate roughly comparable to EIF
    I_per_Irr=-2400000 * b2.amp,
)
fiber = cleo.light.Light(
    coords=(1.75, 1.75, 0) * b2.mm,
    light_model=cleo.light.fiber473nm(R0=0.2 * b2.mm),
    name="fiber",
)

plotargs = {
    "colors": ["xkcd:fuchsia"],
    "xlim": (-2.5, 2.5),
    "ylim": (-2.5, 2.5),
    "zlim": (0, 1),
    "scatterargs": {"s": 1},
}
scatterargs = {
    "s": 1,
}

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
    "scatterargs": {"s": 1},  # to adjust neuron marker size
    "axis_scale_unit": b2.mm,
}

fig, ax = cleo.viz.plot(
    exciteneurons,
    **plotargs,
    devices=[probe, fiber],
)
fig.savefig(f"{results_dir}/100x100_static_setup.png")

sim = cleo.CLSimulator(net)
# sim.inject_stimulator(opto, exciteneurons, Iopto_var_name='I_opto')
sim.inject(simple_opsin, exciteneurons, Iopto_var_name="I_opto")
sim.inject(fiber, exciteneurons)
sim.inject(probe, exciteneurons)


stim_vals = []
stim_t = []
spike_vals = []


class ReactiveLoopOpto(cleo.ioproc.LatencyIOProcessor):
    def __init__(self):
        super().__init__(sample_period_ms=0.2)

    # since this is open-loop, we don't use state_dict
    def process(self, state_dict, time_ms):
        i, t, z_t = state_dict["Probe"]["spikes"]
        if np.size(i) >= 3:
            if opto_on:
                opto_intensity = 0.15
            else:
                opto_intensity = 0
        else:
            opto_intensity = 0
        stim_vals.append(opto_intensity)
        stim_t.append(time_ms)
        spike_vals.append(np.size(i))
        # return output dict and time
        return ({"fiber": opto_intensity}, time_ms + delay_ms)


sim.set_io_processor(ReactiveLoopOpto())

if generate_video:
    vv = cleo.viz.VideoVisualizer(dt=0.5 * b2.ms, devices_to_plot=[probe, fiber])
    sim.inject(vv, exciteneurons)


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
spike_counts, bin_edges = np.histogram(excitespikes.t / b2.ms, bins=stim_t_extended)

indexes_half_radius = []

for i in range(numofexcneur):
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
    t_spk_ms=excitespikes.t / b2.ms,
    i_spk=excitespikes.i,
    stim_t=stim_t,
    stim_vals=stim_vals,
    fiber_t=np.array(fiber.t_ms),
    fiber_vals=np.array(fiber.values).squeeze(),
    spike_counts=spike_counts,
    bin_edges=bin_edges,
    spike_vals=spike_vals,
    stim_t_extended=stim_t_extended,
    indexes_half_radius=indexes_half_radius,
    numofexcneur=numofexcneur,
    exc_x_mm=exciteneurons.x / b2.mm,
    exc_y_mm=exciteneurons.y / b2.mm,
)

# %%
# plot results (old)
# from plot_single_expt_old import plot_all

# plot_all(
#     excitestate,
#     excitespikes,
#     numofexcneur,
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
