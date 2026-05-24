import numpy as np
import matplotlib.pyplot as plt
import time
import json


def allan(t, y, name, plot=True):
    t1 = time.time()
    dt = t[1] - t[0]
    N = np.size(y)
    m = np.arange(2, np.floor(N / 2) + 1, dtype=np.int32)  # Number of subgroups
    allans = np.zeros(np.size(m))  # Array where Allan variances are stored
    tau = np.zeros(np.size(m))
    allan_i = 0  # Initiates index for allans

    for mi in m:
        k = np.floor(N / mi)  # Size of subgroup
        tau[allan_i] = k * dt
        if tau[allan_i] == tau[allan_i - 1]:
            tau[allan_i] = 0
            continue
        means_mi = np.zeros(mi)  # Means of subgroups array

        for s in np.arange(mi):
            means_mi[s] = np.mean(y[int(s * k) : int((s + 1) * k)])
        diffs_squares = (means_mi[1:] - means_mi[:-1]) ** 2
        allans[allan_i] = 0.5 * np.mean(diffs_squares)
        allan_i = allan_i + 1

    allans = np.trim_zeros(allans)[1:]
    tau = np.trim_zeros(tau)[1:]

    if plot == True:
        print("plotting")
        plt.ion()
        plt.close("all")
        plt.figure(1)
        plt.plot(t, y, "k", linewidth=0.8)
        plt.title("Time series")
        plt.xlabel("Time")
        plt.ylabel("y")
        plt.figure(2)
        allan_plot = plt.subplot(111)
        allan_plot.loglog(tau, np.sqrt(allans), "k", linewidth=0.8)
        plt.title("Allan Variance Plot")
        plt.xlabel("Integration Time [secs]")
        plt.ylabel("Allan std")
        plt.savefig(name)
        plt.show()

    upper = 10000
    valid_idx = np.where(tau <= upper)[0]
    min_idx = valid_idx[np.argmin(allans[valid_idx])]
    allan_min = tau[min_idx]

    print("Allan variance plot processing time: %0.3e" % (time.time() - t1))
    print("the minimised allan variance below 10 seconds is:", allan_min)
    return np.array((tau, allans))


data = np.genfromtxt(
    "../data/thermistor_boiling_data.csv", delimiter=",", skip_header=1
)
t = data[:, 0] / 1000
p = data[:, 1]

res = allan(t, p, "allan_boiling.png", plot=True)

upper = 10
allans = res[1]
tau = res[0]
valid_idx = np.where(tau <= upper)[0]
min_idx = valid_idx[np.argmin(allans[valid_idx])]
allan_min = allans[min_idx]

result = {
    "minimal allan": {"allan variance": allan_min, "time": tau[min_idx]},
}

with open("boiling_allan.json", "w") as f:
    json.dump(result, f, indent=4)


data = np.genfromtxt("../data/thermistor_Allan_data.csv", delimiter=",", skip_header=1)
t = data[:, 0] / 1000
p = data[:, 1]

res = allan(t, p, "allan.png", plot=True)

upper = 10
allans = res[1]
tau = res[0]
valid_idx = np.where(tau <= upper)[0]
min_idx = valid_idx[np.argmin(allans[valid_idx])]
allan_min = allans[min_idx]

result = {
    "minimal allan": {"allan variance": allan_min, "time": tau[min_idx]},
}

with open("allan.json", "w") as f:
    json.dump(result, f, indent=4)

data = np.genfromtxt("../data_2/raiser_temp_run_data.csv", delimiter=",", skip_header=1)
t = data[:, 0] / 1000
p = data[:, 1]

res = allan(t, p, "allan_isochoric.png", plot=True)

upper = 10
allans = res[1]
tau = res[0]
valid_idx = np.where(tau <= upper)[0]
min_idx = valid_idx[np.argmin(allans[valid_idx])]
allan_min = allans[min_idx]

result = {
    "minimal allan": {"allan variance": allan_min, "time": tau[min_idx]},
}

with open("allan_isochoric.json", "w") as f:
    json.dump(result, f, indent=4)
