#!/usr/bin/env python
# coding: utf-8

# %%
import time
from pathlib import Path

import brian2.only as b2
import cleo
import matplotlib.pyplot as plt
from brian2 import np

from cleo_pe1 import config, model

b2.prefs.codegen.target = "numpy"
cleo.utilities.style_plots_for_paper()

# %%
t_start = time.time()

# %%
cfg = config.SimulationConfig()
b2.prefs.codegen.target = cfg.target
net, objs = model.load_model(cfg)
ng_exc = objs["ng_exc"]

# In[3]:
simple_opsin = cleo.opto.ProportionalCurrentOpsin(
    name="simple_opsin",
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
fig.savefig(cfg.results_dir / "100x100_static_setup.png")

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
            if cfg.opto_on:
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
        return ({"fiber": opto_intensity}, t_samp + cfg.delay_ms * b2.ms)


sim.set_io_processor(ReactiveLoopOpto())

if cfg.generate_video:
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
exc_spike_mon = objs["exc_spike_mon"]

stim_t_extended = np.copy(stim_t)
stim_t_extended = np.append(stim_t_extended, np.max(stim_t) + stim_t[1] - stim_t[0])
spike_counts, bin_edges = np.histogram(exc_spike_mon.t / b2.ms, bins=stim_t_extended)

indexes_half_radius = []

for i in range(cfg.N_exc):
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
    cfg.results_dir / "data.npz",
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
    numofexcneur=cfg.N_exc,
    exc_x_mm=ng_exc.x / b2.mm,
    exc_y_mm=ng_exc.y / b2.mm,
)

# %%
# plot results
from src.plot_single_expt import plot_all

plot_all(cfg.results_dir)

# %%
if cfg.generate_video:
    from matplotlib import animation as animation

    fiber.max_Irr0_mW_per_mm2_viz = 5
    ani = vv.generate_Animation(plotargs, slowdown_factor=1000, figsize=[16, 12])
    ani.save(cfg.results_dir / "100x100_neurons_15ms_animation.gif")
    plt.close()
