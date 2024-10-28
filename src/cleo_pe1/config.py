from dataclasses import asdict, dataclass, field
from pathlib import Path

import brian2 as b2


@dataclass
class SimulationConfig:
    # experiment
    opto_on: bool = False
    delay_ms: int = 0
    results_base_dir: Path = Path("results")
    generate_video: bool = True
    generate_3d_video: bool = False
    target: str = "numpy"
    seed: int = 18051844

    # underlying model
    p0: float = 0.1  # for weak connections
    unit_len: b2.Quantity = field(default_factory=lambda: 50 * b2.um)
    sigma_no_len: float = 5
    """using the arbitrary coordinate system from the paper. `sigma` applies length"""
    sigma_strong_no_len: float = 7
    """using the arbitrary coordinate system from the paper. `sigma_strong` applies length"""
    p1: int = 1  # for strong connections
    tau_ampa: b2.Quantity = field(default_factory=lambda: 2 * b2.ms)
    tau_gaba: b2.Quantity = field(default_factory=lambda: 5 * b2.ms)
    N_exc: int = 10000
    N_inh: int = 2500
    N_exc_strong: int = 1000
    N_syn_strong: int = 500
    # g_weak: float = 0.1
    # g_strong: float = 2
    w_base: float = 1525
    strong_weak_ratio: float = 0.25
    inh_exc_w_ratio: float = 1
    g_exc: b2.Quantity = field(default_factory=lambda: b2.siemens)
    g_inh: b2.Quantity = field(default_factory=lambda: b2.siemens)
    C_exc: b2.Quantity = field(default_factory=lambda: b2.farad)
    C_inh: b2.Quantity = field(default_factory=lambda: b2.farad)
    exc_v_init_lim: tuple = (-0.5, 1)
    exc_thresh_lim: tuple = (0.5, 2)
    inh_v_init_lim: tuple = (-0.5, 0.5)
    inh_thresh_lim: tuple = (0, 1)
    stim_level: float = 7.5e-17
    stim_radius: b2.Quantity = field(default_factory=lambda: 0.25 * b2.mm)
    stim_duration: b2.Quantity = field(default_factory=lambda: 2 * b2.ms)

    @property
    def results_dir(self):
        if self.opto_on:
            exp_name = f"opto_on_delay{self.delay_ms}ms"
        else:
            exp_name = "opto_off"
        results_dir = self.results_base_dir / exp_name
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        return results_dir

    @property
    def sigma(self):
        return self.sigma_no_len * self.unit_len

    @property
    def sigma_strong(self):
        return self.sigma_strong_no_len * self.unit_len

    def save_to_file(self):
        with open(self.results_dir / "config.txt", "w") as f:
            f.write(str(asdict(self)).replace(", ", ",\n"))


def realistic_cfg(**kwargs):
    tau = 15 * b2.ms
    params = {
        "g_exc": 1 / (100 * b2.Mohm),
        "g_inh": 1 / (200 * b2.Mohm),
    }
    params["C_exc"] = tau * params["g_exc"]
    params["C_inh"] = tau * params["g_inh"]
    params["w_base"] = 5.969e-7
    params["strong_weak_ratio"] = 0.109
    return SimulationConfig(**(params | kwargs))
