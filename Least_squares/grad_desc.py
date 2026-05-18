import numpy as np
from chisquare import CHI_SQUARE

print("Hello World")


class GRADIENT_DESCENT:
    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        a: float,
        b: float,
        alpha: float,
        func: str,
    ) -> None:

        self.x = np.array([np.mean(x_data[:, i]) for i in range(len(x_data[:, 0]))])
        self.x_std = np.array([np.std(x_data[:, i]) for i in range(len(x_data[:, 0]))])
        self.y = np.array([np.mean(y_data[:, i]) for i in range(len(x_data[:, 0]))])
        self.y_std = np.array([np.std(y_data[:, i]) for i in range(len(x_data[:, 0]))])
        self.a = a
        self.b = b
        self.alpha = alpha
        self.func = func

    def d_chi_d_param(self):
        pass

    def grad_step(self):
        pass

    def grad_run(self):
        pass
