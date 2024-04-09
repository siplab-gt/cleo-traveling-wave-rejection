#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-paper')

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

class save_plotting_data():
    def __init__(self, stim_t,spike_counts_in_radius,spike_vals,stim_vals):
        self.time_axis=stim_t
        self.grond_truth_spikes=spike_counts_in_radius
        self.recorded_spikes=spike_vals
        self.opto_stim=stim_vals

obj_with_opto_on = load_object("./opto_on_v40.pickle")
obj_with_opto_off = load_object("./opto_off_v41.pickle")

light_473nm = '#72b5f2'

#%%
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(5.25, 1.35), constrained_layout=True)
#line_radius=ax_one.plot(stim_t[0:(len(stim_t)-1)],spike_counts_in_radius,'k-')
line_detect=ax.plot(obj_with_opto_on.time_axis,obj_with_opto_on.recorded_spikes,'k-', label='with opto')
line_detect=ax.plot(obj_with_opto_off.time_axis,obj_with_opto_off.recorded_spikes,'-', color='.5', label='without opto')
ax.set(ylabel='Spikes/ms', xlabel='Time (ms)', title='Spikes detected at electrode')
#ax_one.legend([line_radius,line_detect],['Spikes within .5 mm of Probe','Spikes Detected by Probe'])
ylim=ax.get_ylim()
xlim=ax.get_xlim()
# print(obj_with_opto_on.opto_stim)
for i in range(len(obj_with_opto_on.opto_stim)-1):
    if obj_with_opto_on.opto_stim[i]>0:
        opto_line = ax.plot([obj_with_opto_on.time_axis[i], obj_with_opto_on.time_axis[i+1]], [ylim[1], ylim[1] ], color=light_473nm, ms=4)
opto_line[0].set(label='light on')

# ax.plot([xlim[0], xlim[1]], [3, 3 ],'b--')
ax.set_xlim(xlim)
sns.despine(fig)
# fig.legend(loc='center left', bbox_to_anchor=(.9, .5))
fig.legend(loc='center')
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig('spiking_and_stim_opto_compare.svg')

#%%
fig2, (ax_2) = plt.subplots(1, 1, sharex=True)
#line_radius=ax_one.plot(stim_t[0:(len(stim_t)-1)],spike_counts_in_radius,'k-')
line_detect=ax_2.plot(obj_with_opto_on.time_axis,obj_with_opto_on.grond_truth_spikes,'k-')
line_detect=ax_2.plot(obj_with_opto_off.time_axis,obj_with_opto_off.grond_truth_spikes,'k-',color='.5')
ax.set(ylabel='Spikes per ms',title='Spiking Activity')
#ax_one.legend([line_radius,line_detect],['Spikes within .5 mm of Probe','Spikes Detected by Probe'])
ylim=ax_2.get_ylim()
xlim=ax_2.get_xlim()
print(obj_with_opto_on.opto_stim)
for i in range(len(obj_with_opto_on.opto_stim)-1):
    if obj_with_opto_on.opto_stim[i]>0:
        ax_2.plot([obj_with_opto_on.time_axis[i], obj_with_opto_on.time_axis[i+1]], [ylim[1], ylim[1] ],'b-',ms=4)
ax_2.plot([xlim[0], xlim[1]], [3, 3 ],'b--')
ax_2.set_xlim(xlim)
# plt.savefig('spiking_and_stim_opto_compare_v26_gt.png')