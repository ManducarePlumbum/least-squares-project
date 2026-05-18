import numpy as np

from error import ERROR


class CHI_SQUARE:
    def __init__(
        self, x_data: np.ndarray, y_data: np.ndarray, a: float, b: float, func: str
    ) -> None:
        self.x = np.array([np.mean(x_data[:, i]) for i in range(len(x_data[:, 0]))])
        self.x_std = np.array([np.std(x_data[:, i]) for i in range(len(x_data[:, 0]))])
        self.y = np.array([np.mean(y_data[:, i]) for i in range(len(x_data[:, 0]))])
        self.y_std = np.array([np.std(y_data[:, i]) for i in range(len(x_data[:, 0]))])
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
            W[i, i] = ERROR(
                self.x[i],
                self.y[i],
                self.x_std[i],
                self.y_std[i],
                self.a,
                self.b,
                self.func,
            ).eff_std()
        chi_sq = diff.T @ W @ diff
        return chi_sq


if __name__ == "__main__":
    # Linear test
    x_linear = np.tile(np.arange(10), (10, 1))
    y_linear = np.array(
        [
            [1.2, 2.9, 4.8, 7.2, 9.1, 10.8, 12.9, 14.7, 16.8, 19.1],
            [0.9, 3.1, 5.1, 6.9, 8.9, 11.1, 13.1, 15.0, 17.1, 18.9],
            [1.1, 3.0, 5.0, 7.1, 9.0, 11.0, 13.0, 14.9, 17.0, 19.0],
            [1.3, 2.8, 4.9, 7.0, 9.2, 10.9, 12.8, 15.1, 16.9, 19.2],
            [0.8, 3.2, 5.2, 6.8, 8.8, 11.2, 13.2, 14.8, 17.2, 18.8],
            [1.0, 3.3, 5.3, 7.3, 9.3, 10.7, 13.3, 15.2, 16.8, 19.3],
            [1.4, 2.9, 4.7, 7.1, 9.1, 11.1, 12.9, 14.9, 17.0, 19.1],
            [0.9, 3.1, 5.0, 6.9, 9.0, 11.0, 13.0, 15.0, 17.1, 19.0],
            [1.1, 3.0, 5.1, 7.0, 8.9, 10.9, 13.1, 14.7, 16.9, 18.9],
            [1.2, 2.9, 4.9, 7.2, 9.2, 11.0, 12.8, 15.1, 17.0, 19.2],
        ]
    )

    a = 1
    b = 3

    test1 = CHI_SQUARE(x_linear, y_linear, a, b, "Linear").chi_square()
    print(test1)

    # Exponential test

    x_exp = np.tile(np.arange(10), (10, 1))
    y_exp = np.array(
        [
            [2.1, 2.7, 3.5, 4.8, 6.5, 8.9, 12.1, 16.5, 22.4, 30.5],
            [1.9, 2.8, 3.7, 5.0, 6.7, 9.1, 12.4, 16.8, 22.7, 30.8],
            [2.0, 2.6, 3.6, 4.9, 6.6, 9.0, 12.2, 16.7, 22.6, 30.7],
            [2.2, 2.9, 3.4, 4.7, 6.4, 8.8, 12.0, 16.4, 22.3, 30.4],
            [1.8, 2.7, 3.8, 5.1, 6.8, 9.2, 12.5, 16.9, 22.8, 30.9],
            [2.0, 2.5, 3.6, 4.9, 6.5, 8.9, 12.1, 16.6, 22.5, 30.6],
            [2.1, 2.8, 3.5, 4.6, 6.7, 9.3, 12.3, 16.8, 22.7, 30.7],
            [1.9, 2.7, 3.7, 5.0, 6.3, 9.0, 12.0, 16.5, 22.4, 30.8],
            [2.0, 2.9, 3.4, 4.8, 6.6, 8.8, 12.4, 16.7, 22.6, 30.5],
            [2.2, 2.6, 3.6, 5.1, 6.8, 9.1, 12.2, 16.9, 22.8, 30.6],
        ]
    )

    test2 = CHI_SQUARE(x_exp, y_exp, a, b, "Exponential").chi_square()
    print(test2)
