from dataclasses import asdict

import brian2.only as b2
from brian2 import np

from cleo_pe1.config import SimulationConfig

# "hi"
exceqs = """dv/dt = (I_summed - v*g_exc)/C_exc : volt
I_summed = I_stim + I_exc_strong + I_exc_weak + I_inh + I_opto : amp
I_opto : amp
stimulated = sqrt(x**2 + y**2) < stim_radius : boolean
# normalize by original resistance (0.1 Ω)
I_stim = int(stimulated) * int(t < stim_duration) 
    * stim_level * ((1 / g_exc) / (0.1 * ohm))
    * individual_stim_strength
    * amp : amp
individual_stim_strength : 1
I_exc_weak : amp
I_exc_strong : amp
I_inh : amp
x = .05*((floor(i/100.0)) - 50)*mmeter : meter
y = .05*((i%100.0) - 50)*mmeter : meter
z: meter
thresh: volt"""

inheqs = """dv/dt = (I_stim + I_exc - v*g_inh)/C_inh :volt
# I_stim = stimulus_inh(t,i)*amp : amp
I_stim : amp
I_exc: amp
x = .1*((floor(i/50)) - 25) * mm : meter
y = .1*((i % 50) - 25) * mm : meter
z = .5 * mm : meter
thresh: volt"""

# normalize by N/N_post to get average N_incoming
syn_model = """
    ds/dt = -s/tau : amp (clock-driven)
    I_NAME_post= W_EXPR / (N / N_post) * s : amp (summed)
"""
on_pre = "s += 0.5*amp"
# original paper used Manhattan distance
# p="p0*exp(-abs(x_pre-x_post)/(.05*mmeter)*abs(y_pre-y_post) / sigma**2)",
# Euclidean distance instead:
connect_prob = "p0 * exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2) / sigma**2)"


def load_model(cfg: SimulationConfig):
    rng = np.random.default_rng(cfg.seed)
    cfg_needed_on_load = {}
    for k in ["sigma", "p0", "p1"]:
        cfg_needed_on_load[k] = getattr(cfg, k)
    b2.prefs.codegen.target = cfg.target
    ng_exc = b2.NeuronGroup(
        cfg.N_exc,
        exceqs,
        threshold="v > thresh",
        reset="v = -5*volt",
    )
    ng_inh = b2.NeuronGroup(
        cfg.N_inh,
        inheqs,
        threshold="v > thresh",
        reset="v = 0*volt",
    )

    ng_exc.z = rng.uniform(0.45, 0.55, cfg.N_exc) * b2.mm
    ng_exc.individual_stim_strength = rng.uniform(0, 1, cfg.N_exc)

    # Initial conditions and firing thresholds for neurons
    ng_exc.v = rng.uniform(*cfg.exc_v_init_lim, cfg.N_exc) * b2.volt
    ng_exc.thresh = rng.uniform(*cfg.exc_thresh_lim, cfg.N_exc) * b2.volt

    ng_inh.v = rng.uniform(*cfg.inh_v_init_lim, cfg.N_inh) * b2.volt
    ng_inh.thresh = rng.uniform(*cfg.inh_thresh_lim, cfg.N_inh) * b2.volt

    # synapses
    syn_e2e_weak = b2.Synapses(
        ng_exc,
        ng_exc,
        syn_model.replace("NAME", "exc_weak")
        .replace("W_EXPR", "w_base")
        .replace("tau", "tau_ampa"),
        on_pre=on_pre,
        namespace=cfg_needed_on_load,
    )
    syn_e2e_weak.connect(
        condition="i!=j",
        p=connect_prob,
    )
    print(f"Number of weak synapses: {len(syn_e2e_weak)}")

    # strong synapses
    i_strong_neurons = rng.choice(range(cfg.N_exc), cfg.N_exc_strong, replace=False)
    x_strong = ng_exc.x[i_strong_neurons]
    y_strong = ng_exc.y[i_strong_neurons]
    p_strong_connect = cfg.p1 * np.exp(
        (
            (10 * cfg.unit_len) ** 2
            - (x_strong - x_strong[:, None]) ** 2
            - (y_strong - y_strong[:, None]) ** 2
        )
        / cfg.sigma_strong**2
    )
    np.fill_diagonal(p_strong_connect, 0)
    assert p_strong_connect.shape == (cfg.N_exc_strong, cfg.N_exc_strong)
    strong_connections = (
        rng.uniform(size=(cfg.N_exc_strong, cfg.N_exc_strong)) < p_strong_connect
    )
    i_syn_strong, j_syn_strong = np.where(strong_connections)
    j_syn_strong
    idx_strong_to_keep = rng.choice(
        range(len(i_syn_strong)), cfg.N_syn_strong, replace=False
    )
    assert len(idx_strong_to_keep) == cfg.N_syn_strong
    i_syn_strong = i_syn_strong[idx_strong_to_keep]
    j_syn_strong = j_syn_strong[idx_strong_to_keep]
    assert len(i_syn_strong) == cfg.N_syn_strong
    # map back from index among strong neurons to index among all neurons
    i_syn_strong_orig = i_strong_neurons[i_syn_strong]
    j_syn_strong_orig = i_strong_neurons[j_syn_strong]
    assert len(i_syn_strong_orig) == len(j_syn_strong_orig) == cfg.N_syn_strong

    # %%
    syn_e2e_strong = b2.Synapses(
        ng_exc,
        ng_exc,
        syn_model.replace("NAME", "exc_strong")
        .replace("W_EXPR", "w_base * strong_weak_ratio")
        .replace("tau", "tau_ampa"),
        on_pre=on_pre,
        namespace=cfg_needed_on_load,
    )
    syn_e2e_strong.connect(i=i_syn_strong_orig, j=j_syn_strong_orig)
    print(f"Number of strong synapses (should be 500): {len(syn_e2e_strong)}")

    # %%
    syn_i2e = b2.Synapses(
        ng_inh,
        ng_exc,
        syn_model.replace("NAME", "inh")
        .replace("W_EXPR", "-inh_exc_w_ratio * w_base * (1 + strong_weak_ratio)")
        .replace("tau", "tau_gaba"),
        on_pre=on_pre,
        namespace=cfg_needed_on_load,
    )
    syn_i2e.connect(
        condition="i!=j",
        p=connect_prob,
    )

    syn_e2i = b2.Synapses(
        ng_exc,
        ng_inh,
        syn_model.replace("NAME", "exc")
        .replace("W_EXPR", "inh_exc_w_ratio * w_base * (1 + strong_weak_ratio)")
        .replace("tau", "tau_ampa"),
        on_pre=on_pre,
        namespace=cfg_needed_on_load,
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

    return net, {
        "ng_exc": ng_exc,
        "ng_inh": ng_inh,
        "syn_e2e_weak": syn_e2e_weak,
        "syn_e2e_strong": syn_e2e_strong,
        "exc_st_mon": exc_st_mon,
        "exc_spike_mon": exc_spike_mon,
    }
