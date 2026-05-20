import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from data_organiser import DATA


raw = "data/thermistor_boiling_data.csv"

with open("allan_variance/boiling_allan.json", "r") as f:
    loaded = json.load(f)
integraltime = loaded["minimal allan"]["time"] * 1000

data_temps = np.linspace(2, 96, 96)
means, sem = DATA(raw, integraltime, data_temps).data_assembly()


def function(x, a, b):
    return b * np.exp(a * x)


log_R = np.log(means[:, 0])
coeffs = np.polyfit(means[:, 1], log_R, 1)
init_alpha = coeffs[0]
init_R0 = np.exp(coeffs[1])

df_dT = init_alpha * init_R0 * np.exp(init_alpha * means[:, 1])
err_eff = np.sqrt(sem[:, 0] ** 2 + df_dT**2 * sem[:, 1] ** 2)
popt, pcov = curve_fit(
    function, means[:, 1], means[:, 0], p0=[init_alpha, init_R0], sigma=sem[:, 0]
)

perr = np.sqrt(np.diag(pcov))
print("parameters", popt)
print("uncertainties", perr)


fit = function(means[:, 1], *popt)
residuals = means[:, 0] - fit
chi_square = np.sum((residuals / err_eff) ** 2)

print(chi_square)

fit_label = f"Fit: R = ({popt[1]:.2f} ± {perr[1]:.2f}) · exp(({popt[0]:.4f} ± {perr[0]:.4f}) · T)"
plt.errorbar(means[:, 1], means[:, 0], yerr=sem[:, 0], fmt="o", label="data")
plt.plot(means[:, 1], fit, label=fit_label)
plt.legend()
plt.grid(True)
plt.show()
