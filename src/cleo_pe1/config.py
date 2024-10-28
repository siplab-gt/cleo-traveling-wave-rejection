from dataclasses import asdict, dataclass, field
from pathlib import Path

import brian2 as b2


@dataclass
class SimulationConfig:
    # experiment
    opto_on: bool = False
    delay_ms: int = 0
    results_base_dir: Path = Path("results")
    generate_video: bool = False
    exp_name: str = "AUTOMATICALLY GENERATED"
    results_dir: Path = "AUTOMATICALLY GENERATED"
    target: str = "numpy"

    # underlying model
    p0: float = 0.1  # for weak connections
    unit_len: b2.Quantity = field(default_factory=lambda: 50 * b2.um)
    sigma: b2.Quantity = 5
    sigma_strong: b2.Quantity = 7
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

    def __post_init__(self):
        self.sigma = self.sigma * self.unit_len
        self.sigma_strong = self.sigma_strong * self.unit_len
        if self.opto_on:
            self.exp_name = f"opto_on_delay{self.delay_ms}ms"
        else:
            self.exp_name = "opto_off"
        self.results_dir = self.results_base_dir / self.exp_name
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)

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
