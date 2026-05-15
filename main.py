import numpy as np
import matplotlib as plt

data = np.genfromtxt("data/thermistor_boiling_data.csv", delimiter=",", skip_header=1)

print(len(data[:, 0]))
