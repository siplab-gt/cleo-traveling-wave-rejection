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
from cleo import *
import random as rnd
import pickle
import matplotlib.pyplot as plt
prefs.codegen.target = "numpy"


# In[2]:

start_scope()

#Lines 89-27 define default values. 9-16 are used in the generation of synapses
po = .1;
alpha = 5;
p1 = 1;
alpha1 = 7;
#14-15 are the time constants for synaptic currents
tauampa = 2*ms;
taugaba = 5*ms;
ex = 2.71828;
#these two determine the number of neurons
numofexcneur = 10000; 
numofinhneur = 2500;

excstartingval = np.random.uniform(-.5,.5,numofexcneur);
inhstartingval = np.random.uniform(-.5,0,numofinhneur);
excreset = -5*volt;
inhreset = 0*volt;
g = .1;



#i is the index of a neuron
# "The strengths of the connections were selected to balance the averaged excitatory current with the averaged inhibitory current in network"
#37-38 show the structure of the recurrent currents, however this is not complete and need to be fixed
#Isynext_equations = '''Isynext = g/n(sum(se*exciteneurons.w)+1/ni*sum(si*inhneurons.w)) : amp'''
#Isyninh_equations = '''1/n*sum(se*we) : amp'''

# sum() is handled by (summed)
# ni is handled N_incoming 
# N_outgoing (?)

# define the equations and variables for the excitatory and inhibitory neurons
stimulus_exc = TimedArray(np.hstack([[[2000*(floor((index)/100.0)>44 and floor((index)/100.0)<56 and ((index)%100.0)>44 and ((index)%100.0)<56)], [0]]
                                for index in range(numofexcneur)]),
                                dt=2*ms)

exceqs = '''dv/dt = (I_summed - v/gexc)/C : volt
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
excthreshold: volt'''

inheqs = '''dv/dt = (Iext + Isynext - v/ginh)/C :volt
C: farad
Iext: amp
Isynext: amp
Isyninh : amp
ginh: ohm
x: meter
y: meter
z: meter
excthreshold: volt'''

resetv = 'v =  excreset';
excthreshold = 'v > excthreshold';

#NOTE: Iext is currently set to 100 across all neurons, however this is not true for the final model, and needs to be fixed (2ms input)
exciteneurons = NeuronGroup(numofexcneur, exceqs, threshold= excthreshold, reset= resetv)
inhneurons = NeuronGroup(numofinhneur, inheqs, threshold= excthreshold, reset='v = 0*volt')

random_sampled_neurons=rnd.sample(range(0,numofexcneur-1),1000)
sampled_neurons_bool=np.zeros((numofexcneur),dtype=bool)
sampled_neurons_bool[random_sampled_neurons]=True
#exciteneurons.x = '2*((floor(i/50.0)) - 50)' #value for 2500 exc neuron
#exciteneurons.y = '2*((i%50.0) - 50)' #value for 2500 exc neurons
#exciteneurons.x = '.1*1*((floor(i/10.0)) - 5)*mmeter' #value for 100 exc neuron
#exciteneurons.y = '.1*1*((i%10.0) - 5)*mmeter' #value for 100 exc neurons
#exciteneurons.x = '.1*((floor(i/100.0)) - 50)*mmeter' #value for 10000 exc neuron
#exciteneurons.y = '.1*((i%100.0) - 50)*mmeter' #value for 10000 exc neurons
exciteneurons.z=np.random.uniform(.45,.55, numofexcneur)*mmeter
#Initial conditions for neurons
exciteneurons.v = excstartingval*volt;
exciteneurons.C = np.ones(numofexcneur)*1*farad;
exciteneurons.gexc = np.ones(numofexcneur)*(.1)*ohm;
#exciteneurons.Iext = np.ones(numofexcneur)*100*amp;
#exciteneurons.Iext = 'stimulus_exc(t,i)*amp'
#exciteneurons.I_default = np.ones(numofexcneur)*100*amp;
exciteneurons.excthreshold = np.random.uniform(.5, 2, numofexcneur)*volt;
exciteneurons.sampled_neurons= sampled_neurons_bool

#inhneurons.x = '8*((floor(i/12.5)) - 50)' #value for 625 inh neurons
#inhneurons.y = '8*((i % 12.5) - 50)' #value for 625 inh neurons
inhneurons.x = '.1*((floor(i/50)) - 25)*mmeter'
inhneurons.y = '.1*((i % 50) - 25)*mmeter'
inhneurons.z = '.5*mmeter'
inhneurons.v = inhstartingval*volt;
inhneurons.C = np.ones(numofinhneur)*1*farad;
inhneurons.ginh = np.ones(numofinhneur)*(.2)*ohm;
#inhneurons.Iext = np.ones(numofinhneur)*100*amp;
#stimulus_inh = TimedArray(np.hstack([[0, 0, c, 0, 0]
#                                 for c in np.random.rand(numofinhneur)]),
#                                dt=2*ms)
#inhneurons.Iext = 'stimulus_inh(t)*amp'
inhneurons.excthreshold = np.random.uniform(0,1, numofinhneur)*volt;

#84-88 defines the weak excitatory synapses
sexcweak = Synapses(exciteneurons, exciteneurons, 
'''w=40000 : 1
dse/dt = -se/tauampa : amp (clock-driven)
Isynweakext_post= g/N_incoming*se*w : amp (summed)
''',  on_pre ='se = se + .5*amp')
sexcweak.connect(condition='i!=j',
             p = '''po*exp((-abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post)/(.05*mmeter))/((alpha**2)))''')
#sexcweak.w = .2;

#90-96 defines the strong excitatory synapses, note that the total number needs to be capped at 500 to replicate the paper
sexcstrong = Synapses(exciteneurons, exciteneurons, 
'''w=80000 : 1
dse/dt = -se/tauampa: amp (clock-driven)
Isynext_post = g/N_incoming*se*w : amp (summed)
''',  on_pre ='se = se + .5*amp')

sexcstrong.connect(condition='i!=j and sampled_neurons_pre and sampled_neurons_post',
             p = '''p1*exp((100-(abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post)/(.05*mmeter)))/((alpha1**2)))''')

#sexcstrong.w = 1;

#99-113 defines the inhibitory to excitatory, excitatory to inhibitory, and inhibitory-inhibitory synapses
inhtoexcsynapse = Synapses(inhneurons, exciteneurons, '''w=-10000000 : 1
dsi/dt = -si/taugaba : amp (clock-driven)
Isyninh_post = g/N_incoming/N_outgoing*si : amp (summed)
''',  on_pre ='si = si + .5*amp')

inhtoexcsynapse.connect(condition='i!=j',
             p = '''po*exp((-abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post)/(.05*mmeter))/((alpha**2)))''')

exctoinhsynapse = Synapses(exciteneurons, inhneurons, '''w=10000000 : 1
dsi/dt = -si/taugaba : amp (clock-driven)
Isyninh_post = w*g/N_incoming/N_outgoing*si : amp (summed)
''',  on_pre ='si = si + .5*amp')

exctoinhsynapse.connect(condition='i!=j',
             p = '''po*exp((-abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post)/(.05*mmeter))/((alpha**2)))''')

#exctoinhsynapse.w = 2; #this needs to be changed

#a starting point to define the synaptic currents (S7), please not that this is a state variable of excite neurons
#exciteneurons.Isynext = 'g/numofexcneur*(sum(sexcstrong[:, j].se*sexcstrong[:, j].w)+1/numofinhneur*sum(inhneurons[:, j].si*inhneurons[:, j].w)) '

excitestate = StateMonitor(exciteneurons, ['v', 'Iext', 'Isynext'], record=True)
excitespikes = SpikeMonitor(exciteneurons)

#exciteneurons = SpikeGeneratorGroup(numofexcneur, indices, times)

net= Network(exciteneurons,inhneurons,sexcweak,sexcstrong, exctoinhsynapse, inhtoexcsynapse, excitestate, excitespikes)


# In[27]:


incoming_connections=sexcweak.N_incoming_post
np.delete(incoming_connections,np.where(incoming_connections==0))


# In[3]:


from cleo.coords import assign_coords_rand_rect_prism
from cleo.opto import *
from cleo.ephys import Probe, SortedSpiking

simple_opsin = ProportionalCurrentOpsin(
    name="simple_opsin",
    # handpicked gain to make firing rate roughly comparable to EIF
    I_per_Irr=-240000000000*amp,
)
fiber = Light(
    coords=(1.75, 1.75, 0) * mm,
    light_model=fiber473nm(),
    name="fiber",
)

plotargs = {
    "colors": ["xkcd:fuchsia"],
    "xlim": (-2.5, 2.5),
    "ylim": (-2.5, 2.5),
    "zlim": (0, 1),
    "scatterargs": {"s": 1} 
}
scatterargs = {
    "s":1,
}

spikes = SortedSpiking(name='spikes',
    r_perfect_detection=25 * umeter,
    r_half_detection=50 * umeter,
    save_history=True,
)
probe = ephys.Probe([1.75, 1.75, 0.5] * mm)
probe.add_signals(spikes)

plotargs = {
    "colors": ["xkcd:fuchsia"],
    "xlim" : (-2.5, 2.5),
    "ylim" : (-2.5, 2.5),
    "zlim": (0, 1),
    "scatterargs": {"s": 1},  # to adjust neuron marker size
    "axis_scale_unit": mmeter,
}

#cleo.viz.plot(
#    exciteneurons,
#    **plotargs,
#    devices="all",
#)
#plt.savefig('100x100_static_setup.png')

sim = CLSimulator(net)
#sim.inject_stimulator(opto, exciteneurons, Iopto_var_name='I_opto')
sim.inject(simple_opsin, exciteneurons, Iopto_var_name='I_opto')
sim.inject(fiber, exciteneurons)
sim.inject(probe, exciteneurons)


from cleo.ioproc import LatencyIOProcessor
stim_vals = []
stim_t = []
spike_vals = []
class ReactiveLoopOpto(LatencyIOProcessor):
    def __init__(self):
        super().__init__(sample_period_ms=.2)

    # since this is open-loop, we don't use state_dict
    def process(self, state_dict, time_ms):
        i, t, z_t = state_dict['Probe']['spikes']
        if np.size(i)>=3:
            opto_intensity = 0
            #opto_intensity = 5 #0 to turn opto off
        else:
            opto_intensity = 0
        stim_vals.append(opto_intensity)
        stim_t.append(time_ms)
        spike_vals.append(np.size(i))
        # return output dict and time
        return ({"fiber": opto_intensity}, time_ms)

sim.set_io_processor(ReactiveLoopOpto())

###

#from cleo.viz import VideoVisualizer

#vv = VideoVisualizer(dt=.5 * ms, devices_to_plot="all")
#sim.inject(vv, exciteneurons)


####


sim.run(15*ms)


####


#fig, ax = plt.subplots()
#ax.plot(excitespikes.t/msecond, excitespikes.i, '|',ms=.2,lw=1)
#ax.set(ylabel='neuron index', title='spiking')
#plt.savefig('spiking_100x100_15ms_v43.png')
#plt.close()
#print(sim.network.t)

####

stim_t_extended=stim_t
stim_t_extended.append(np.max(stim_t)+stim_t[1]-stim_t[0])
spike_counts,bin_edges = np.histogram(excitespikes.t/ms,bins=stim_t_extended)

indexes_half_radius=[]

for i in range(numofexcneur):
    if (((.05*((floor(i/100.0)) - 50)*mmeter-1.75*mm)**2+(.05*((i%100.0) - 50)*mmeter-1.75*mm)**2)**(.5)) < .5*mmeter:
        indexes_half_radius.append(i)

spikes_to_keep = np.isin(excitespikes.i, indexes_half_radius) 
t_spikes_to_keep = excitespikes.t[spikes_to_keep]

spike_counts_in_radius,bin_edges = np.histogram(t_spikes_to_keep/ms,bins=stim_t_extended)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
#ax1.plot(stim_t[0:(len(stim_t)-1)],spike_counts)
ax1.plot(stim_t[0:(len(stim_t)-1)],spike_counts_in_radius)
ax1.set(ylabel='spikes per ms',title='Spikes within .5 mm of Probe')
ax2.plot(stim_t[0:(len(stim_t)-1)],spike_vals)
ax2.set(ylabel='spikes per ms',title='Spikes Detected by Probe')
ax3.plot(stim_t[0:(len(stim_t)-1)], stim_vals)
ax3.set(ylabel=r'$Irr_0$ (mm/mW$^2$)', title='Optogenetic Stimulus', xlabel='time (ms)');
plt.savefig('spiking_and_stim_v43_updated_cleo.png')
plt.close()

fig, (ax_one) = plt.subplots(1, 1, sharex=True)
line_radius=ax_one.plot(stim_t[0:(len(stim_t)-1)],spike_counts_in_radius,'k-')
line_detect=ax_one.plot(stim_t[0:(len(stim_t)-1)],spike_vals,'k-',color='.5')
ax_one.set(ylabel='Spikes per ms',title='Spiking Activity')
#ax_one.legend([line_radius,line_detect],['Spikes within .5 mm of Probe','Spikes Detected by Probe'])
ylim=ax_one.get_ylim()
for i in range(len(stim_vals)-1):
    if stim_vals[i]>0:
        ax_one.plot([stim_t[i], stim_t[i+1]], [ylim[1], ylim[1] ],'b-',ms=4)

plt.savefig('spiking_and_stim_one_plot_v43_updated_cleo.png')
plt.savefig('spiking_and_stim_one_plot_v43_updated_cleo.pdf')
plt.close()

import pickle
class save_plotting_data():
    def __init__(self, stim_t,spike_counts_in_radius,spike_vals,stim_vals):
        self.time_axis=stim_t[0:(len(stim_t)-1)]
        self.grond_truth_spikes=spike_counts_in_radius
        self.recorded_spikes=spike_vals
        self.opto_stim=stim_vals
def save_object(obj):
    try:
        with open("opto_off_v43_updated_cleo.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
obj=save_plotting_data(stim_t=stim_t,spike_counts_in_radius=spike_counts_in_radius,spike_vals=spike_vals,stim_vals=stim_vals)        
save_object(obj)

# In[ ]:


opto.max_Irr0_mW_per_mm2_viz = 5
#ani = vv.generate_Animation(plotargs, slowdown_factor=1000, figsize=[16,12])


# In[ ]:


from matplotlib import animation as animation
#ani.save('100x100_neurons_15ms_animation_v43_updated_cleo_opto_on.gif')
plt.close()

# In[ ]:


def visualize_activity_at_time(state,num_of_neurons,sampletime):
    sidelength = (np.sqrt(num_of_neurons)).astype(int)
    timeslice = np.zeros([sidelength,sidelength])
    for i in range(sidelength):
        for j in range(sidelength):
            neuronactivity=state.v[i*sidelength+j]
            timeslice[i,j]=neuronactivity[sampletime]

    fig, ax = plt.subplots()
    im = ax.imshow(timeslice, cmap=plt.get_cmap('jet'))
    ax.set(ylabel='y position (mm)', title='Action Potential at {} ms'.format(sampletime), xlabel='X position (mm)');

visualize_activity_at_time(state=excitestate,num_of_neurons=numofexcneur,sampletime=2)
#plt.savefig('100by100_activity_at_2_ms.png')
visualize_activity_at_time(state=excitestate,num_of_neurons=numofexcneur,sampletime=5)
#plt.savefig('100by100_activity_at_5_ms.png')
visualize_activity_at_time(state=excitestate,num_of_neurons=numofexcneur,sampletime=8)
#plt.savefig('100by100_activity_at_8_ms.png')


# In[ ]:


def visualize_spiking_per_ms(spikes,num_of_neurons,sampletime):
    sidelength = (np.sqrt(num_of_neurons)).astype(int)
    spiketimes=spikes.t
    neuronspikes=spikes.i
    x_coordinates=[]
    y_coordinates=[]
    for i in range(len(spiketimes)):
        if spiketimes[i]>=sampletime*ms and spiketimes[i]<((sampletime+1)*ms):
            x_coordinates.append(.05*((floor(neuronspikes[i]/sidelength)) - sidelength/2))
            y_coordinates.append(.05*((neuronspikes[i]%sidelength) - sidelength/2))         
    fig, ax = plt.subplots()
    ax.plot(x_coordinates,y_coordinates,'ks',ms=1.6)
    ax.set_xlim([-2.5,2.5])
    ax.set_ylim([-2.5,2.5])
    ax.set(ylabel='y position (mm)', title='Spiking Activity at {} ms'.format(sampletime), xlabel='X position (mm)');

visualize_spiking_per_ms(spikes=excitespikes,num_of_neurons=numofexcneur,sampletime=0)
plt.savefig('100by100_spiking_at_0_ms_v43_updated_cleo_opto_on.png')
plt.savefig('100by100_spiking_at_0_ms_v43_updated_cleo_opto_on.png')
plt.close()

#visualize_spiking_per_ms(spikes=excitespikes,num_of_neurons=numofexcneur,sampletime=1)
#plt.savefig('100by100_spiking_at_1_ms_v21.png')

visualize_spiking_per_ms(spikes=excitespikes,num_of_neurons=numofexcneur,sampletime=4)
plt.savefig('100by100_spiking_at_4_ms_v43_updated_cleo_opto_on.png')
plt.savefig('100by100_spiking_at_4_ms_v43_updated_cleo_opto_on.pdf')
plt.close()

visualize_spiking_per_ms(spikes=excitespikes,num_of_neurons=numofexcneur,sampletime=8)
plt.savefig('100by100_spiking_at_8_ms_v43_updated_cleo_opto_on.png')
plt.savefig('100by100_spiking_at_8_ms_v43_updated_cleo_opto_on.pdf')
plt.close()

visualize_spiking_per_ms(spikes=excitespikes,num_of_neurons=numofexcneur,sampletime=10)
#plt.savefig('100by100_spiking_at_10_ms_v21.png')
visualize_spiking_per_ms(spikes=excitespikes,num_of_neurons=numofexcneur,sampletime=12)
plt.savefig('100by100_spiking_at_12_ms_v43_updated_cleo_opto_on.png')
plt.savefig('100by100_spiking_at_12_ms_v43_updated_cleo_opto_on.pdf')
plt.close()

#
#Identify spiking by index
#find x y value of index and distance from probe
#
def visualize_spiking_per_ms_smoothed(spikes,num_of_neurons,sampletime,smooth_std):
    sidelength = (np.sqrt(num_of_neurons)).astype(int)
    spiketimes=spikes.t
    neuronspikes=spikes.i
    #x_coordinates=[]
    #y_coordinates=[]
    x_coordinates=np.zeros(len(spiketimes))
    y_coordinates=np.zeros(len(spiketimes))
    fig, ax = plt.subplots()
    figsize=(1.3, 1.3)
    firing_rate_smoothed=np.zeros(len(spiketimes))
    for i in range(len(spiketimes)):
        if spiketimes[i]>=(sampletime-1.25)*ms and spiketimes[i]<((sampletime+1.25)*ms):
            x_coordinates[i]=.05*((floor(neuronspikes[i]/sidelength)) - sidelength/2)
            y_coordinates[i]=.05*((neuronspikes[i]%sidelength) - sidelength/2)
            index_times=[]
            counter=0
            for j in range(len(spiketimes)):
                if neuronspikes[j]==neuronspikes[i]:
                    index_times.append(j)
                    counter=counter+1
                    if i==j:
                        index_index=counter
            firing_rate_smoothed[i]=np.sum((2.718**(-.5*((np.array(spiketimes[i])-np.array(spiketimes[index_times]))/smooth_std)**2))/(smooth_std*2.506)/1000)
            #ax.plot(x_coordinate,y_coordinate,'ks',ms=8*firing_rate_smoothed)#was 1.6 with 10 ms standard deviation smoothing; 3.2 with 20 ms
            #ax.plot(x_coordinate,y_coordinate, 's', ms=2, c=plt.cm.Greys(8*firing_rate_smoothed))
    plt.scatter(x_coordinates,y_coordinates, s=2, c=firing_rate_smoothed, cmap='Greys')
    ax.set_xlim([-2.5,2.5])
    ax.set_ylim([-2.5,2.5])
    ax.set(ylabel='y position (mm)', title='Spiking Activity at {} ms'.format(sampletime), xlabel='X position (mm)')

visualize_spiking_per_ms_smoothed(spikes=excitespikes,num_of_neurons=numofexcneur,sampletime=0, smooth_std=.2)
plt.savefig('100by100_spiking_at_0_ms_v43_updated_cleo_smoother.png')
plt.savefig('100by100_spiking_at_0_ms_v43_updated_cleo_smoother.pdf')
plt.close()
visualize_spiking_per_ms_smoothed(spikes=excitespikes,num_of_neurons=numofexcneur,sampletime=4, smooth_std=.2)
plt.savefig('100by100_spiking_at_4_ms_v43_updated_cleo_smoother.png')
plt.savefig('100by100_spiking_at_4_ms_v43_updated_cleo_smoother.pdf')
plt.close()
visualize_spiking_per_ms_smoothed(spikes=excitespikes,num_of_neurons=numofexcneur,sampletime=8, smooth_std=.2)
plt.savefig('100by100_spiking_at_8_ms_v43_updated_cleo_smoother.png')
plt.savefig('100by100_spiking_at_8_ms_v43_updated_cleo_smoother.pdf')
plt.close()
visualize_spiking_per_ms_smoothed(spikes=excitespikes,num_of_neurons=numofexcneur,sampletime=12, smooth_std=.2)
plt.savefig('100by100_spiking_at_12_ms_v43_updated_cleo_smoother.png')
plt.savefig('100by100_spiking_at_12_ms_v43_updated_cleo_smoother.pdf')
plt.close()