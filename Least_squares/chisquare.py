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
            return self.b * np.exp(self.a * self.x)
        else:
            raise ValueError

    def chi_square(self):
        diff = self.y - self.function()
        W = np.array([[0.0 for _ in range(len(self.x))] for _ in range(len(self.x))])
        for i in range(len(W[:, 0])):
            W[i, i] = (
                1
                / ERROR(
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
        chi_sq = diff.T @ W @ diff
        return chi_sq
