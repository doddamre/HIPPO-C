import copy
import importlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

batman = importlib.import_module("batman")
HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[2] 

#Reference planet HIP 67522 c (same values as Task A)
ref = batman.TransitParams()
ref.t0 = 0.0                  # time of mid-transit [days]
ref.per = 14.334892           # orbital period [days]
ref.rp = 0.0528               # planet radius [stellar radii]
ref.a = 19.14                 # semi-major axis [stellar radii]
ref.inc = 89.20               # inclination [deg]
ref.ecc = 0.077
ref.w = 186.6                 # argument of periastron [deg]
ref.u = [0.322, 0.221]        # limb darkening, same as Task A
ref.limb_dark = "quadratic"

#Hypothetical planet with same orbital parameters but half radius
hypo = copy.deepcopy(ref)
hypo.rp = ref.rp/2      # planet radius 

#Hypothetical planet with same orbital parameters but double radius
hypo2 = copy.deepcopy(ref)
hypo2.rp = ref.rp * 2      # planet radius [stellar radii]

t = np.linspace(-0.25, 0.25, 1000)   # days around mid-transit
flux_ref = batman.TransitModel(ref, t).light_curve(ref)
flux_half = batman.TransitModel(hypo, t).light_curve(hypo)
flux_double = batman.TransitModel(hypo2, t).light_curve(hypo2)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t * 24, flux_ref, color="tab:blue",
        label=f"HIP 67522 c (Rp/R* = {ref.rp:.4f})")
ax.plot(t * 24, flux_half, color="tab:orange",
        label=f"Hypothetical planet, half radius (Rp/R* = {hypo.rp:.4f})")
ax.plot(t * 24, flux_double, color="tab:red",
        label=f"Hypothetical planet, double radius (Rp/R* = {hypo2.rp:.4f})")
ax.set_xlabel("Time from mid-transit [hours]")
ax.set_ylabel("Relative flux")
ax.set_title("HIP 67522 c vs. planets with different radii")
ax.legend()
fig.tight_layout()
fig.savefig(REPO_ROOT / "assignment2_taskC.png", dpi=150)
plt.show()