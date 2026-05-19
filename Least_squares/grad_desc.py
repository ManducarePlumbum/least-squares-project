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

        # Add constraints
        self.a_max = 1000  # Alpha must be ≤ 0 for cooling
        self.a_min = -1000  # Reasonable lower bound
        self.b_min = 100  # R0 positive and reasonable

    def function(self):

        if self.func == "Linear":
            return self.a * self.x + self.b
        elif self.func == "Exponential":
            exponent = np.clip(self.a * self.b, -100, 100)
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
            exponent = np.clip(self.a * self.x, -100, 100)
            exp_val = np.exp(exponent)
            df_da = self.x * self.b * exp_val
            df_db = exp_val
            # Check for invalid values
            df_da = np.nan_to_num(df_da, nan=0.0, posinf=1e10, neginf=-1e10)
            df_db = np.nan_to_num(df_db, nan=0.0, posinf=1e10, neginf=0.0)
            return np.array([[df_da[i], df_db[i]] for i in range(len(self.x))])
        else:
            raise ValueError

    def grad_step(self):

        # Apply constraints before step
        self.a = np.clip(self.a, self.a_min, self.a_max)
        self.b = np.maximum(self.b_min, self.b)

        # Compute function with safeguards
        f = self.function()
        diff = self.y - f

        # Build weight matrix with regularization
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

        jac = self.jacobian()

        # Compute gradient
        try:
            grad = jac.T @ W @ diff
            # Clip gradient
            grad = np.clip(grad, -1e6, 1e6)

            # Update parameters
            self.a += self.alpha * grad[0]
            self.b += self.alpha * grad[1]

            # Re-apply constraints
            self.a = np.clip(self.a, self.a_min, self.a_max)
            self.b = np.maximum(self.b_min, self.b)

        except Exception as e:
            print(f"Gradient step failed: {e}")
            pass

    def grad_run(self):
        for i in range(self.Iterations):
            old_a, old_b = self.a, self.b
            self.grad_step()

            # Check for convergence
            if abs(self.a - old_a) < 1e-8 and abs(self.b - old_b) < 1e-5:
                print(f"Converged at iteration {i}")
                break

            # Check for NaN
            if np.isnan(self.a) or np.isnan(self.b):
                print(f"NaN at iteration {i}, reverting to last good values")
                self.a, self.b = old_a, old_b
                break

            # Print progress
            if i % 100 == 0:
                print(f"Iter {i}: a={self.a:.6f}, b={self.b:.2f}")

        return self.a, self.b
