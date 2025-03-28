import numpy as np


class LinearRegression:
    """
    Implementa uma regressão linear simples.
    """

    def __init__(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("x and y must be NumPy arrays.")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        if len(x) < 2:
            raise ValueError("x and y must have at least two data points.")

        self.x = x
        self.y = y
        self._correlation_coefficient = self._correlacao()
        self._inclination = self._inclinacao()
        self._intercept = self._interceptacao()

    def _correlacao(self):
        """Calcula o coeficiente de correlação de Pearson (r)."""
        covariacao = np.cov(self.x, self.y, bias=True)[0, 1]
        variancia_x = np.var(self.x)
        variancia_y = np.var(self.y)

        if variancia_x == 0 or variancia_y == 0:
            raise ValueError(
                "Variance of x or y is zero. Cannot calculate correlation."
            )

        return covariacao / np.sqrt(variancia_x * variancia_y)

    def _inclinacao(self):
        """Calcula a inclinação (m) da reta de regressão."""
        std_x = np.std(self.x)
        std_y = np.std(self.y)
        return self._correlation_coefficient * (std_y / std_x)

    def _interceptacao(self):
        """Calcula o intercepto (b) da reta de regressão."""
        return np.mean(self.y) - self._inclination * np.mean(self.x)

    def predict(self, x_new):
        """Faz previsões para novos valores de x."""
        if not isinstance(x_new, (int, float, np.ndarray)):
            raise TypeError("x_new must be a number or a NumPy array.")
        return self._intercept + (self._inclination * x_new)
