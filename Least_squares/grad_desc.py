import numpy as np
from .chisquare import CHI_SQUARE
from .error import ERROR


class GRADIENT_DESCENT:
    def __init__(
        self,
        x_means: np.ndarray,
        y_means: np.ndarray,
        x_std: np.ndarray,
        y_std: np.ndarray,
        init_a: float,
        init_b: float,
        alpha: float,
        func: str,
        Iterations: int,
    ) -> None:

        self.x = x_means
        self.x_std = x_std
        self.y = y_means
        self.y_std = y_std
        self.a = init_a
        self.b = init_b
        self.alpha = alpha
        self.func = func
        self.Iterations = Iterations

    def function(self):

        if self.func == "Linear":
            return self.a * self.x + self.b
        elif self.func == "Exponential":
            exponent = np.clip(self.a * self.x, -50, 50)
            return self.b * np.exp(exponent)
        else:
            raise ValueError

    def jacobian(self):
        if self.func == "Linear":
            df_da = self.x
            df_db = 1.0
            return np.array([[df_da[i], df_db] for i in range(len(self.x))])
        elif self.func == "Exponential":
            # Clip exponent
            exponent = np.clip(self.a * self.x, -50, 50)
            exp_val = np.exp(exponent)
            df_da = self.x * self.b * exp_val
            df_db = exp_val
            # Clip Jacobian to prevent NaN propagation
            df_da = np.clip(df_da, -1e15, 1e15)
            df_db = np.clip(df_db, -1e15, 1e15)
            return np.array([[df_da[i], df_db[i]] for i in range(len(self.x))])
        else:
            raise ValueError

    def grad_step(self):

        self.a = np.clip(self.a, -1.0, 0)  # Force negative alpha
        self.b = np.clip(self.b, 100, 1e6)  # Reasonable R0 range

        # Compute function with safeguards
        f = self.function()

        if np.any(np.isnan(f)):
            self.alpha *= 0.5
            return

        diff = self.y - f

        # Build weight matrix with regularization
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

        weights = np.nan_to_num(weights, nan=1e-6, posinf=1e-6, neginf=1e-6)

        jac = self.jacobian()

        # Gradient descent with safeguard
        try:
            grad = -2 * jac.T @ (weights * diff)

            # Clip gradients to prevent explosion
            grad = np.clip(grad, -1e6, 1e6)

            # Update parameters with adaptive step size
            step_a = self.alpha * grad[0]
            step_b = self.alpha * grad[1]

            # Apply updates with bounds
            self.a -= step_a
            self.b -= step_b

            # Re-apply bounds
            self.a = np.clip(self.a, -1.0, 0)
            self.b = np.clip(self.b, 100, 1e6)
        except Exception as e:
            # If gradient calculation fails, reduce learning rate
            print(f"Gradient calculation failed: {e}, reducing learning rate")
            self.alpha *= 0.5

    def grad_run(self):
        chis = np.array([0.0 for _ in range(self.Iterations)])
        prev_loss = CHI_SQUARE(
            self.x, self.y, self.x_std, self.y_std, self.a, self.b, self.func
        ).chi_square()
        for i in range(self.Iterations):
            self.grad_step()
            curr_loss = CHI_SQUARE(
                self.x, self.y, self.x_std, self.y_std, self.a, self.b, self.func
            ).chi_square()
            if curr_loss > prev_loss * 1.01:
                self.alpha *= 0.5
            chis[i] = curr_loss
            prev_loss = curr_loss

        return self.a, self.b, chis


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Small exponential test dataset (30 points)
    # True model: y = 5.0 * exp(-0.3 * x) + noise
    x_test = np.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
            13.5,
            14.0,
            14.5,
        ]
    )

    y_test = np.array(
        [
            5.12,
            4.28,
            3.71,
            3.15,
            2.89,
            2.41,
            2.05,
            1.88,
            1.52,
            1.31,
            1.19,
            0.98,
            0.89,
            0.71,
            0.68,
            0.59,
            0.48,
            0.41,
            0.39,
            0.31,
            0.28,
            0.25,
            0.21,
            0.19,
            0.15,
            0.14,
            0.12,
            0.11,
            0.09,
            0.08,
        ]
    )

    # Constant uncertainty estimates (adjust if your code needs heteroscedastic errors)
    np.random.seed(42)
    x_std_test = 0.05 * np.random.random(len(x_test))
    y_std_test = 0.15 * np.random.random(len(x_test))

    # True parameters for reference½
    true_a, true_b = -0.3, 5.0

    a, b, chis = GRADIENT_DESCENT(
        x_test,
        y_test,
        x_std_test,
        y_std_test,
        -0.02,
        4.5,
        0.000001,
        "Exponential",
        10000,
    ).grad_run()
    print(a, b)
    print(chis[0])
    print(chis[-1])
    plt.plot(chis)
    plt.show()

    plt.plot(x_test, b * np.exp(a * x_test))
    plt.errorbar(x_test, y_test, yerr=y_std_test, xerr=x_std_test, fmt="o")
    plt.show()
