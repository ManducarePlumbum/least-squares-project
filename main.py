import numpy as np
import json
from data_organiser import DATA
from Least_squares.grad_desc import GRADIENT_DESCENT
import matplotlib.pyplot as plt


raw = "data/thermistor_boiling_data.csv"

with open("allan_variance/boiling_allan.json", "r") as f:
    loaded = json.load(f)
integraltime = loaded["minimal allan"]["time"] * 1000

data_temps = np.linspace(2, 96, 86)
means, sem = DATA(raw, integraltime, data_temps).data_assembly()

print("temp range[", means[:, 1].min(), means[:, 1].max(), "]")
print("resist range[", means[:, 0].min(), means[:, 0].max(), "]")

# More robust initialization:
log_R = np.log(means[:, 0])
# Fit log(R) = log(R0) + alpha * T using linear regression
coeffs = np.polyfit(means[:, 1], log_R, 1)
init_alpha = np.clip(coeffs[0], -0.5, -0.001)
init_R0 = np.clip(np.exp(coeffs[1]), 1000, 50000)
print(f"Initial guesses: alpha={init_alpha:.6f}, R0={init_R0:.2f}")

# Now run exponential GD with these initials
alpha, R0, chis = GRADIENT_DESCENT(
    means[:, 1],
    means[:, 0],
    sem[:, 1],
    sem[:, 0],
    init_alpha,
    init_R0,
    1e-12,
    "Exponential",
    10000,
).grad_run()

print(alpha, R0)
print(chis[0])
print(chis[-1])
plt.plot(chis)
plt.show()

plt.plot(means[:, 1], R0 * np.exp(alpha * means[:, 1]))
plt.errorbar(means[:, 1], means[:, 0], yerr=sem[:, 0], fmt=".")
plt.show()
