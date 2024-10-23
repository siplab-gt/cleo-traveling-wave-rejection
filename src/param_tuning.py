from dataclasses import asdict

import brian2.only as b2
from brian2 import np
from config import SimulationConfig
from model import load_model


def tune_weights(cfg: SimulationConfig):
    b2.prefs.codegen.target = cfg.target
    cfg.stim_level = 0  # remove stimulus
    cfg.inh_exc_w_ratio = 0  # remove inhibition
    net, objs = load_model(cfg)
    sweak = objs["syn_e2e_weak"]
    sstrong = objs["syn_e2e_strong"]
    ng = objs["ng_exc"]

    # instead of measuring single neurons, I'll take advantage of random initializations and thresholds
    # to get a smoother, non-integer measure
    # find neurons with 5 weak presynaptic neurons
    i_weak = np.unique(sweak.i["N_incoming == 5"])
    j_weak = np.setdiff1d(np.unique(sweak.j["N_incoming == 5"]), i_weak)
    print(f"{len(i_weak)=}, {len(j_weak)=}")

    # find strong neurons with 1 presynaptic neuron
    i_strong = np.setdiff1d(
        np.unique(sstrong.i["N_incoming == 1"]), np.concatenate((i_weak, j_weak))
    )
    j_strong = np.setdiff1d(
        np.unique(sstrong.j["N_incoming == 1"]),
        np.concatenate((i_weak, j_weak, i_strong)),
    )
    print(f"{len(i_strong)=}, {len(j_strong)=}")

    i_record = np.concatenate((i_weak, i_strong, j_weak, j_strong))
    i_all_others = np.setdiff1d(ng.i, i_record)
    assert len(i_record) + len(i_all_others) == cfg.N_exc == len(ng), (
        len(i_record),
        len(i_all_others),
        len(ng),
    )

    # set the voltage to just above threshold for presynaptic neurons
    # and make their threshold high so they just spike once
    ng.thresh[np.r_[i_weak, i_strong]] = 19120731 * b2.volt
    ng.v[i_weak] = ng.thresh[i_weak] * 1.01
    ng.v[i_strong] = ng.thresh[i_strong] * 1.01
    # ensure other neurons won't spike
    ng.v[i_all_others] = -18010630 * b2.volt

    # ng_test = b2.NeuronGroup(7, exceqs, threshold="v > thresh", reset="v = -5*volt")
    # ng_test.thresh = np.mean(cfg.exc_thresh_lim) * b2.volt
    # print(ng_test.thresh)
    # ng_test.v[:5] = ng_test.thresh[:5] * 1.001
    # print(ng_test.v)

    # stim_arr_exc = np.zeros((2, cfg.N_exc))  # T x N
    # stimulus_exc = b2.TimedArray(stim_arr_exc, dt=2 * b2.ms)

    # syn_weak = b2.Synapses(
    #     ng_test,
    #     ng_test,
    #     syn_model.replace("NAME", "exc_weak"),
    #     on_pre=on_pre,
    #     namespace={"w": cfg.w_base * cfg.g_weak, "tau": cfg.tau_ampa},
    # )
    # syn_weak.connect(i=range(5), j=[5] * 5)

    # syn_strong = b2.Synapses(
    #     ng_test,
    #     ng_test,
    #     syn_model.replace("NAME", "exc_strong"),
    #     on_pre=on_pre,
    #     namespace={"w": cfg.w_base * cfg.g_strong, "tau": cfg.tau_ampa},
    # )
    # syn_strong.connect(i=[0], j=[6])

    # spmon = b2.SpikeMonitor(ng_test)

    # net = b2.Network(b2.collect())
    spmon = objs["exc_spike_mon"]

    net.store()

    # 5 spikes to 1 spike
    def evaluate_weights(w_base, sw_ratio):
        cfg.w_base = w_base
        cfg.strong_weak_ratio = sw_ratio
        net.run(10 * b2.ms, namespace=asdict(cfg))

        n_weak_pre_spikes = np.sum(np.isin(spmon.i, i_weak))
        assert n_weak_pre_spikes == len(i_weak), f"{n_weak_pre_spikes=}, {len(i_weak)=}"

        n_strong_pre_spikes = np.sum(np.isin(spmon.i, i_strong))
        assert n_strong_pre_spikes == len(
            i_strong
        ), f"{n_strong_pre_spikes=}, {len(i_strong)=}"

        # should have no other spikes
        assert np.sum(np.isin(spmon.i, i_all_others)) == 0

        n_weak_post_spikes = np.sum(np.isin(spmon.i, j_weak))
        n_strong_post_spikes = np.sum(np.isin(spmon.i, j_strong))
        result = n_weak_post_spikes / len(j_weak), n_strong_post_spikes / len(j_strong)
        print(f"{w_base=}, {sw_ratio=}, {result=}")
        net.restore()
        return result

    def binary_search(fn, low, high, target, tolerance=0.01):
        while low < high:
            mid = (low + high) / 2
            spike_count = fn(mid)
            if abs(spike_count - target) < tolerance:
                return mid
            elif spike_count < target:
                low = mid
            else:
                high = mid
        return low

    w_base = binary_search(
        lambda w_base: evaluate_weights(w_base, cfg.strong_weak_ratio)[0], 0, 1e6, 1
    )

    strong_weak_ratio = binary_search(
        lambda sw_ratio: evaluate_weights(w_base, sw_ratio)[1], 0, 100, 1
    )
    return w_base, strong_weak_ratio


if __name__ == "__main__":
    cfg = SimulationConfig()
    print(tune_weights(cfg))
