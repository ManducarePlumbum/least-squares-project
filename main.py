import numpy as np
import json
from data_organiser import DATA
from Least_squares.grad_desc import GRADIENT_DESCENT
from Least_squares.chisquare import CHI_SQUARE

raw = "data/thermistor_boiling_data.csv"

with open("allan_variance/boiling_allan.json", "r") as f:
    loaded = json.load(f)
integraltime = loaded["minimal allan"]["time"] * 1000

data_temps = np.linspace(2, 96, 94)
means, sem = DATA(raw, integraltime, data_temps).data_assembly()

print("temp range[", means[:, 1].min(), means[:, 1].max(), "]")
print("resist range[", means[:, 0].min(), means[:, 0].max(), "]")


# SIMPLE linear fit for initial parameters (no GD, no ERROR class)
Temp = means[:, 1]
R_log = np.log(means[:, 0])
p = np.polyfit(Temp, R_log, 1)  # p[0] = slope (alpha), p[1] = intercept (ln_R0)
init_alpha = p[0]
init_R0 = np.exp(p[1])

print(f"Initial guesses: alpha={init_alpha:.6f}, R0={init_R0:.2f}")

# Now run exponential GD with these initials
alpha, R0 = GRADIENT_DESCENT(
    means[:, 1],
    means[:, 0],
    sem[:, 1],
    sem[:, 0],
    init_alpha,
    init_R0,
    0.001,
    "Exponential",
    1000,
).grad_run()

print(f"Final: alpha={alpha:.6f}, R0={R0:.2f}")
print(
    CHI_SQUARE(
        means[:, 1], means[:, 0], sem[:, 1], sem[:, 0], alpha, R0, "Exponential"
    ).chi_square()
)
