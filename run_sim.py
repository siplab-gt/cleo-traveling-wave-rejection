#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from brian2 import *
import numpy as np
import itertools
import math
from cleosim import *
import random as rnd

prefs.codegen.target = "numpy"


# In[2]:


# TO DO
# Fix strong connections- only 500
# Modify weights to match paper- see comments on line 33
# get current calculations to work

start_scope()

# Lines 89-27 define default values. 9-16 are used in the generation of synapses
po = 0.1
alpha = 5
p1 = 1
alpha1 = 7
# 14-15 are the time constants for synaptic currents
tauampa = 2 * ms
taugaba = 5 * ms
ex = 2.71828
# these two determine the number of neurons
numofexcneur = 10000
numofinhneur = 2500
excstartingval = np.random.uniform(-0.5, 0.5, numofexcneur)
inhstartingval = np.random.uniform(-0.5, 0, numofinhneur)
excreset = -5 * volt
inhreset = 0 * volt
g = 0.1
generate_video = False


# i is the index of a neuron
# "The strengths of the connections were selected to balance the averaged excitatory current with the averaged inhibitory current in network"
# 37-38 show the structure of the recurrent currents, however this is not complete and need to be fixed
# Isynext_equations = '''Isynext = g/n(sum(se*exciteneurons.w)+1/ni*sum(si*inhneurons.w)) : amp'''
# Isyninh_equations = '''1/n*sum(se*we) : amp'''

# sum() is handled by (summed)
# ni is handled N_incoming
# N_outgoing (?)

# 41-58 define the equations and variables for the excitatory and inhibitory neurons
stimulus_exc = TimedArray(
    np.hstack(
        [
            [
                [
                    2000
                    * (
                        floor((index) / 100.0) > 44
                        and floor((index) / 100.0) < 56
                        and ((index) % 100.0) > 44
                        and ((index) % 100.0) < 56
                    )
                ],
                [0],
            ]
            for index in range(numofexcneur)
        ]
    ),
    dt=2 * ms,
)

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
Iext: amp
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
exciteneurons = NeuronGroup(numofexcneur, exceqs, threshold=excthreshold, reset=resetv)
inhneurons = NeuronGroup(
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
exciteneurons.z = np.random.uniform(0.45, 0.55, numofexcneur) * mmeter
# Initial conditions for neurons
exciteneurons.v = excstartingval * volt
exciteneurons.C = np.ones(numofexcneur) * 1 * farad
exciteneurons.gexc = np.ones(numofexcneur) * (0.1) * ohm
# exciteneurons.Iext = np.ones(numofexcneur)*100*amp;
# exciteneurons.Iext = 'stimulus_exc(t,i)*amp'
# exciteneurons.I_default = np.ones(numofexcneur)*100*amp;
exciteneurons.excthreshold = np.random.uniform(0.5, 2, numofexcneur) * volt
exciteneurons.sampled_neurons = sampled_neurons_bool

# inhneurons.x = '8*((floor(i/12.5)) - 50)' #value for 625 inh neurons
# inhneurons.y = '8*((i % 12.5) - 50)' #value for 625 inh neurons
inhneurons.x = ".1*((floor(i/50)) - 25)*mmeter"
inhneurons.y = ".1*((i % 50) - 25)*mmeter"
inhneurons.z = ".5*mmeter"
inhneurons.v = inhstartingval * volt
inhneurons.C = np.ones(numofinhneur) * 1 * farad
inhneurons.ginh = np.ones(numofinhneur) * (0.2) * ohm
# inhneurons.Iext = np.ones(numofinhneur)*100*amp;
# stimulus_inh = TimedArray(np.hstack([[0, 0, c, 0, 0]
#                                 for c in np.random.rand(numofinhneur)]),
#                                dt=2*ms)
# inhneurons.Iext = 'stimulus_inh(t)*amp'
inhneurons.excthreshold = np.random.uniform(0, 1, numofinhneur) * volt
# 84-88 defines the weak excitatory synapses
sexcweak = Synapses(
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
    p="""po*exp((-abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post)/(.05*mmeter))/((alpha**2)))""",
)
# sexcweak.w = .2;

# 90-96 defines the strong excitatory synapses, note that the total number needs to be capped at 500 to replicate the paper
sexcstrong = Synapses(
    exciteneurons,
    exciteneurons,
    """w=80000 : 1
dse/dt = -se/tauampa: amp (clock-driven)
Isynext_post = g/N_incoming*se*w : amp (summed)
""",
    on_pre="se = se + .5*amp",
)

sexcstrong.connect(
    condition="i!=j and sampled_neurons_pre and sampled_neurons_post",
    p="""p1*exp((100-(abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post)/(.05*mmeter)))/((alpha1**2)))""",
)

# sexcstrong.w = 1;

# 99-113 defines the inhibitory to excitatory, excitatory to inhibitory, and inhibitory-inhibitory synapses
inhtoexcsynapse = Synapses(
    inhneurons,
    exciteneurons,
    """w=-10000000 : 1
dsi/dt = -si/taugaba : amp (clock-driven)
Isyninh_post = g/N_incoming/N_outgoing*si : amp (summed)
""",
    on_pre="si = si + .5*amp",
)

inhtoexcsynapse.connect(
    condition="i!=j",
    p="""po*exp((-abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post)/(.05*mmeter))/((alpha**2)))""",
)

exctoinhsynapse = Synapses(
    exciteneurons,
    inhneurons,
    """w=10000000 : 1
dsi/dt = -si/taugaba : amp (clock-driven)
Isyninh_post = w*g/N_incoming/N_outgoing*si : amp (summed)
""",
    on_pre="si = si + .5*amp",
)

exctoinhsynapse.connect(
    condition="i!=j",
    p="""po*exp((-abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post)/(.05*mmeter))/((alpha**2)))""",
)

# exctoinhsynapse.w = 2; #this needs to be changed

# a starting point to define the synaptic currents (S7), please not that this is a state variable of excite neurons
# exciteneurons.Isynext = 'g/numofexcneur*(sum(sexcstrong[:, j].se*sexcstrong[:, j].w)+1/numofinhneur*sum(inhneurons[:, j].si*inhneurons[:, j].w)) '

excitestate = StateMonitor(exciteneurons, ["v", "Iext", "Isynext"], record=True)
excitespikes = SpikeMonitor(exciteneurons)

# exciteneurons = SpikeGeneratorGroup(numofexcneur, indices, times)

net = Network(
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


from cleosim.coordinates import assign_coords_rand_rect_prism
from cleosim.opto import *
from cleosim.electrodes import Probe, SortedSpiking

opto = OptogeneticIntervention(
    name="simple_opto",
    # handpicked gain to make firing rate roughly comparable to ELIF
    opsin_model=ProportionalCurrentModel(Iopto_per_mW_per_mm2=-240000000000 * amp),
    light_model_params=default_blue,
    location=(1.75, 1.75, 0) * mm,
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

spikes = SortedSpiking(
    "spikes",
    perfect_detection_radius=25 * umeter,
    half_detection_radius=50 * umeter,
    save_history=True,
)
probe = Probe(
    "probe",
    coords=[1.75, 1.75, 0.5] * mm,
    signals=[spikes],
)


cleosim.visualization.plot(
    exciteneurons,
    colors=["xkcd:fuchsia"],
    xlim=(-2.5, 2.5),
    ylim=(-2.5, 2.5),
    zlim=(0, 1),
    devices_to_plot=[opto, probe],
    figsize=[16, 12],
    scatterargs=scatterargs,
)


# In[ ]:


sim = CLSimulator(net)
sim.inject_stimulator(opto, exciteneurons, Iopto_var_name="I_opto")
sim.inject_recorder(probe, exciteneurons)


# In[ ]:


from cleosim.processing import LatencyIOProcessor

stim_vals = []
stim_t = []
spike_vals = []


class ReactiveLoopOpto(LatencyIOProcessor):
    def __init__(self):
        super().__init__(sample_period_ms=0.2)

    # since this is open-loop, we don't use state_dict
    def process(self, state_dict, time_ms):
        i, t, z_t = state_dict["probe"]["spikes"]
        if np.size(i) >= 6:
            opto_intensity = 5
        else:
            opto_intensity = 0
        stim_vals.append(opto_intensity)
        stim_t.append(time_ms)
        spike_vals.append(np.size(i))
        # return output dict and time
        return ({"simple_opto": opto_intensity}, time_ms)


sim.set_io_processor(ReactiveLoopOpto())


# In[ ]:


from cleosim.visualization import VideoVisualizer

if generate_video:
    vv = VideoVisualizer(dt=0.5 * ms, devices_to_plot="all")
    sim.inject_device(vv, exciteneurons)


# In[ ]:


sim.run(15 * ms)


# In[ ]:


fig, ax = plt.subplots()
ax.plot(excitespikes.t / msecond, excitespikes.i, "|", ms=0.2, lw=1)
ax.set(ylabel="neuron index", title="spiking")
plt.savefig("spiking_100x100_15ms_v21.png")
print(sim.network.t)


# In[ ]:


stim_t_extended = stim_t
stim_t_extended.append(np.max(stim_t) + stim_t[1] - stim_t[0])
spike_counts, bin_edges = np.histogram(excitespikes.t / ms, bins=stim_t_extended)

indexes_half_radius = []

for i in range(numofexcneur):
    if (
        (
            (0.05 * ((floor(i / 100.0)) - 50) * mmeter - 1.75 * mm) ** 2
            + (0.05 * ((i % 100.0) - 50) * mmeter - 1.75 * mm) ** 2
        )
        ** (0.5)
    ) < 0.5 * mmeter:
        indexes_half_radius.append(i)

spikes_to_keep = np.isin(excitespikes.i, indexes_half_radius)
t_spikes_to_keep = excitespikes.t[spikes_to_keep]

spike_counts_in_radius, bin_edges = np.histogram(
    t_spikes_to_keep / ms, bins=stim_t_extended
)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# ax1.plot(stim_t[0:(len(stim_t)-1)],spike_counts)
ax1.plot(stim_t[0 : (len(stim_t) - 1)], spike_counts_in_radius)
ax1.set(ylabel="spikes per ms", title="Spikes within .5 mm of Probe")
ax2.plot(stim_t[0 : (len(stim_t) - 1)], spike_vals)
ax2.set(ylabel="spikes per ms", title="Spikes Detected by Probe")
ax3.plot(stim_t[0 : (len(stim_t) - 1)], stim_vals)
ax3.set(ylabel=r"$Irr_0$ (mm/mW$^2$)", title="Optogenetic Stimulus", xlabel="time (ms)")
plt.savefig("spiking_and_stim_v21.png")


# %%

if generate_video:
    opto.max_Irr0_mW_per_mm2_viz = 5
    ani = vv.generate_Animation(plotargs, slowdown_factor=1000, figsize=[16, 12])


# In[ ]:


from matplotlib import animation as animation

ani.save("100x100_neurons_15ms_animation_v21.gif")


# In[ ]:


def visualize_activity_at_time(state, num_of_neurons, sampletime):
    sidelength = (np.sqrt(num_of_neurons)).astype(int)
    timeslice = np.zeros([sidelength, sidelength])
    for i in range(sidelength):
        for j in range(sidelength):
            neuronactivity = state.v[i * sidelength + j]
            timeslice[i, j] = neuronactivity[sampletime]

    fig, ax = plt.subplots()
    im = ax.imshow(timeslice, cmap=plt.get_cmap("jet"))


visualize_activity_at_time(state=excitestate, num_of_neurons=numofexcneur, sampletime=2)
# plt.savefig('100by100_activity_at_2_ms.png')
visualize_activity_at_time(state=excitestate, num_of_neurons=numofexcneur, sampletime=5)
# plt.savefig('100by100_activity_at_5_ms.png')
visualize_activity_at_time(state=excitestate, num_of_neurons=numofexcneur, sampletime=8)
# plt.savefig('100by100_activity_at_8_ms.png')


# In[ ]:


def visualize_spiking_per_ms(spikes, num_of_neurons, sampletime):
    sidelength = (np.sqrt(num_of_neurons)).astype(int)
    spiketimes = spikes.t
    neuronspikes = spikes.i
    x_coordinates = []
    y_coordinates = []
    for i in range(len(spiketimes)):
        if spiketimes[i] >= sampletime * ms and spiketimes[i] < ((sampletime + 1) * ms):
            x_coordinates.append(
                0.05 * ((floor(neuronspikes[i] / sidelength)) - sidelength / 2)
            )
            y_coordinates.append(
                0.05 * ((neuronspikes[i] % sidelength) - sidelength / 2)
            )
    fig, ax = plt.subplots()
    ax.plot(x_coordinates, y_coordinates, "ks", ms=0.4)
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])


visualize_spiking_per_ms(spikes=excitespikes, num_of_neurons=numofexcneur, sampletime=2)
# plt.savefig('100by100_spiking_at_2_ms.png')
visualize_spiking_per_ms(spikes=excitespikes, num_of_neurons=numofexcneur, sampletime=8)
# plt.savefig('100by100_spiking_at_8_ms.png')
