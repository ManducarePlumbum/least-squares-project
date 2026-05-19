import numpy as np

"""
we estimate the error as in:
https://ned.ipac.caltech.edu/level5/Sept01/Orear/Orear21.html
"""


class ERROR:
    def __init__(
        self,
        x_mean: float,
        y_mean: float,
        x_std: float,
        y_std: float,
        a: float,
        b: float,
        func: str,
    ) -> None:
        self.x_err = x_std
        self.y_err = y_std
        self.x = x_mean
        self.y = y_mean
        self.a = a
        self.b = b
        self.func = func

    def df_dx(self):
        if self.func == "Linear":
            return self.a
        elif self.func == "Exponential":
            return self.a * self.b * np.exp(self.a * self.x)
        else:
            raise ValueError

    def eff_std(self):
        eff_std = np.sqrt(self.y_err**2 + self.df_dx() ** 2 * self.x_err**2) + 1e-10
        return eff_std


if __name__ == "__main__":
    x = 3
    y = 5
    x_std = 0.1
    y_std = 0.15
    a = 2
    b = 1
    print(ERROR(x, y, x_std, y_std, a, b, "Linear").eff_std())
    print(ERROR(x, y, x_std, y_std, a, b, "Exponential").eff_std())
