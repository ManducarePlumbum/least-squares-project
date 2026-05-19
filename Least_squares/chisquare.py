import numpy as np

from .error import ERROR


class CHI_SQUARE:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_std: np.ndarray,
        y_std: np.ndarray,
        a: float,
        b: float,
        func: str,
    ) -> None:
        self.x = x
        self.x_std = x_std
        self.y = y
        self.y_std = y_std
        self.a = a
        self.b = b
        self.func = func

    def function(self):

        if self.func == "Linear":
            return self.a * self.x + self.b
        elif self.func == "Exponential":
            exponent = np.clip(self.a * self.x, -50, 50)
            return self.b * np.exp(exponent)
        else:
            raise ValueError

    def chi_square(self):
        diff = self.y - self.function()
        weights = np.array(
            [
                1.0
                / (
                    ERROR(
                        self.x[i],
                        self.y[i],
                        self.x_std[i],
                        self.y_std[i],
                        self.a,
                        self.b,
                        self.func,
                    ).eff_std()
                    ** 2
                    + 1e-10
                )
                for i in range(len(self.x))
            ]
        )
        chi_sq = np.sum(weights * diff**2)
        return chi_sq
