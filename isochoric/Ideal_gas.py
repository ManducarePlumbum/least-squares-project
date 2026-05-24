import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

time_offset = 1.5

# Read with pandas - it handles ragged rows better
vernier_data = pd.read_csv("../data_2/Run 2.csv", skiprows=1, header=None)

# Run 12: columns 23-24 (0-indexed: 22, 23)
time_1 = vernier_data[22].dropna().astype(float).values
time_1 += time_offset
pressure_1 = vernier_data[23].dropna().astype(float).values

# Run 15: columns 29-30 (0-indexed: 28, 29)
time_2 = vernier_data[28].dropna().astype(float).values
pressure_2 = vernier_data[29].dropna().astype(float).values

thermis_thermo_data = np.genfromtxt(
    "../data_2/raiser_temp_run_data.csv", delimiter=",", skip_header=1
)

R_1 = 9902
U_out = 3.3

temperatures = thermis_thermo_data[:, 2] + 273
thermistor = R_1 * (thermis_thermo_data[:, 1] / (U_out - thermis_thermo_data[:, 1]))
therm_time = thermis_thermo_data[:, 0] - time_offset

## calibration parameters
with open("../calibration_params.json", "r") as f:
    loaded = json.load(f)
alpha = loaded["alpha"]["best_fit"]
alpha_std = loaded["alpha"]["std"]
R_0 = loaded["R_0"]["best_fit"]
R_0_std = loaded["R_0"]["std"]

with open("../allan_variance/allan_isochoric.json") as f:
    load = json.load(f)
integration_time = load["minimal allan"]["time"] * 1000


def Thermistor_temp(data):
    T = np.log(data) / (alpha * np.log(R_0))
    return T


print(thermistor)
gas_temp = Thermistor_temp(thermistor)

data_points = np.linspace(np.min(gas_temp), np.max(gas_temp), 20)


def process_data_point(temp, time, integration_time, T_crit):
    first_accept = np.where(temp >= T_crit)[0][0]
    t_0 = time[first_accept]
    t_1 = t_0 + integration_time

    bool_mask = (time >= t_0) & (time <= t_1)

    data_point_temp = temp[bool_mask]
    data_point_time = time[bool_mask]

    data_mean = np.array([np.mean(data_point_time), np.mean(data_point_temp)])
    data_SEM = np.array([np.std(data_point_time), np.std(data_point_temp)])

    return data_mean, data_SEM


def process_multiple_data_points(temp, time, integration_time, T_crit_values):
    results = np.array(
        [
            process_data_point(temp, time, integration_time, T_crit)
            for T_crit in T_crit_values
        ]
    )
    data_means = np.array([r[0] for r in results])
    data_SEM = np.array([r[1] for r in results])
    return data_means, data_SEM


gas_mean, gas_sem = process_multiple_data_points(
    gas_temp, therm_time, integration_time, data_points
)

print(gas_temp)
plt.plot(gas_mean[0], gas_mean[1])
plt.show()
