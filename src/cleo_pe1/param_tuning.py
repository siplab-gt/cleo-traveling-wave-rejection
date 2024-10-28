from dataclasses import asdict

import brian2.only as b2
from brian2 import np
from config import SimulationConfig, realistic_cfg
from cleo_pe1.model import load_model


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

    spmon = objs["exc_spike_mon"]

    net.store()

    # aiming for 1 spike per postsynaptic neuron
    def evaluate_weights(w_base, sw_ratio, i, j):
        cfg.w_base = w_base
        cfg.strong_weak_ratio = sw_ratio

        # set the voltage to just above threshold for presynaptic neurons
        # and make their threshold high so they just spike once
        ng.thresh[i] = 1912073120061116 * b2.volt
        ng.v[i] = ng.thresh[i] * 1.01
        # ensure other neurons won't spike
        i_others = np.setdiff1d(ng.i, np.r_[i, j])
        ng.v[i_others] = -1801063018501224 * b2.volt

        net.run(10 * b2.ms, namespace=asdict(cfg))

        n_pre_spikes = np.sum(np.isin(spmon.i, i))
        assert n_pre_spikes == len(i), f"{n_pre_spikes=}, {len(i)=}"

        # should have no other spikes
        assert np.sum(np.isin(spmon.i, i_others)) == 0

        n_post_spikes = np.sum(np.isin(spmon.i, j))

        result = n_post_spikes / len(j)
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

    print("w_base:")
    w_base = binary_search(
        lambda w_base: evaluate_weights(w_base, cfg.strong_weak_ratio, i_weak, j_weak),
        0,
        1e6,
        1,
    )

    print("strong_weak_ratio:")
    strong_weak_ratio = binary_search(
        lambda sw_ratio: evaluate_weights(w_base, sw_ratio, i_strong, j_strong),
        0,
        100,
        1,
    )
    return (
        w_base,
        strong_weak_ratio,
        {"syn_weak": sweak, "syn_strong": sstrong, "ng": ng},
    )


if __name__ == "__main__":
    # cfg = SimulationConfig()
    cfg = realistic_cfg()
    w_base, strong_weak_ratio, objs = tune_weights(cfg)
    print(f"{w_base=}, {strong_weak_ratio=}")

    sstrong = objs["syn_strong"]
    sweak = objs["syn_weak"]
    w_strong_unnormalized = w_base * strong_weak_ratio / (sstrong.N / cfg.N_exc)
    w_weak_unnormalized = w_base / (sweak.N / cfg.N_exc)
    swratio_unnormalized = strong_weak_ratio * sweak.N / sstrong.N
    print(
        "This strong-weak ratio represents the total average influence on an excitatory neuron. "
        "A more interpretable strong-weak ratio, without this normalization, is "
    )
    print(f"{swratio_unnormalized=:.3f}")
