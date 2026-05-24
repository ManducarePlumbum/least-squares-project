import numpy as np
import json

data = np.genfromtxt("../data_2/Volume_Estimation.csv", delimiter=",", skip_header=1)

pressure = data[:, -1]
time = data[:, -2]

pres = pressure[~np.isnan(pressure)]
t = time[~np.isnan(time)]

P_i = pres[:5]
P_f = pres[-5:]

P_i = np.array([np.mean(P_i), np.std(P_i) / np.sqrt(len(P_i))])
P_f = np.array([np.mean(P_f), np.std(P_f) / np.sqrt(len(P_f))])

del_V = 0.02

V = del_V * (P_i[0] / (P_f[0] - P_i[0]))

print(V)
del_V_std = 0.00025
dV_dP_f = (del_V * P_f[0]) / (P_f[0] - P_i[0]) ** 2
dV_dP_i = (-del_V * P_i[0]) / (P_f[0] - P_i[0]) ** 2
dV_del_V = P_i[0] / (P_f[0] - P_i[0])

V_std = np.sqrt(
    dV_dP_f**2 * P_f[1] ** 2 + dV_dP_i**2 * P_i[1] ** 2 + dV_del_V**2 * del_V_std**2
)

result = {
    "P_f": {"mean": P_f[0], "Sem": P_f[1]},
    "P_i": {"mean": P_i[0], "Sem": P_i[1]},
    "del_V": {"mean": del_V, "Std": del_V_std},
    "V": {"mean": V, "std": V_std},
}

with open("volume.json", "w") as f:
    json.dump(result, f, indent=4)
