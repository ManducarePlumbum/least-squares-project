import numpy as np
import json


class DATA:
    def __init__(
        self, raw_data: str, integral_time: float, data_points: np.ndarray
    ) -> None:
        self.full_data = np.genfromtxt(raw_data, delimiter=",", skip_header=1)
        self.integration_time = integral_time
        self.data_points = data_points
        self.voltage_to_Resistance()
        self.time = self.full_data[:, 0]
        self.temp = self.full_data[:, 2]

    def voltage_to_Resistance(self):
        R1 = 10**4
        U_out = 3.3

        self.full_data[:, 1] = R1 * (
            self.full_data[:, 1] / (U_out - self.full_data[:, 1])
        )

    def data_point(self, T_crit):
        first_accept = np.where(self.temp >= T_crit)[0][0]
        t_0 = self.time[first_accept]
        t_1 = t_0 + self.integration_time
        bool_mask = (self.time >= t_0) & (self.time <= t_1)
        data_point = self.full_data[bool_mask]

        data_mean = np.array([np.mean(data_point[:, 1]), np.mean(data_point[:, 2])])
        data_SEM = np.array([np.std(data_point[:, 1]), np.std(data_point[:, 2])])

        return data_mean, data_SEM

    def data_assembly(self):

        results = [self.data_point(i) for i in self.data_points]

        data_means = np.array([r[0] for r in results])
        data_SEM = np.array([r[1] for r in results])

        return data_means, data_SEM


if __name__ == "__main__":
    with open("allan_variance/boiling_allan.json", "r") as f:
        loaded = json.load(f)
    integraltime = loaded["minimal allan"]["time"] * 1000
    raw_data = "data/thermistor_boiling_data.csv"
    data_points = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    print(DATA(raw_data, integraltime, data_points).data_assembly())
