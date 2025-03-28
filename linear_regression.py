import numpy as np
import logging

# Configura o logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class LinearRegression:
    """
    Implementa uma regressão linear simples.
    """

    def __init__(self, x, y):
        logger.info("Inicializando LinearRegression.")
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("x and y devem ser arrays NumPy.")
        if len(x) != len(y):
            raise ValueError("x e y devem ter o mesmo comprimento.")
        if len(x) < 2:
            raise ValueError("x e y devem ter pelo menos dois pontos de dados.")

        self.x = x
        self.y = y
        try:
            self._correlation_coefficient = self._correlacao()
            logger.info("Coeficiente de correlação calculado com sucesso.")
            self._inclination = self._inclinacao()
            logger.info("Inclinação calculada com sucesso.")
            self._intercept = self._interceptacao()
            logger.info("Intercepto calculado com sucesso.")
        except ValueError as e:
            logger.error(f"Erro durante o cálculo dos coeficientes: {e}")
            raise  # Propaga a exceção para ser tratada em níveis superiores

    def _correlacao(self):
        """Calcula o coeficiente de correlação de Pearson (r)."""
        logger.debug("Calculando coeficiente de correlação...")
        covariacao = np.cov(self.x, self.y, bias=True)[0, 1]
        variancia_x = np.var(self.x)
        variancia_y = np.var(self.y)

        if variancia_x == 0 or variancia_y == 0:
            raise ValueError(
                "Variância de x ou y é zero. Impossível calcular a correlação."
            )

        return covariacao / np.sqrt(variancia_x * variancia_y)

    def _inclinacao(self):
        """Calcula a inclinação (m) da reta de regressão."""
        logger.debug("Calculando inclinação...")
        std_x = np.std(self.x)
        std_y = np.std(self.y)
        return self._correlation_coefficient * (std_y / std_x)

    def _interceptacao(self):
        """Calcula o intercepto (b) da reta de regressão."""
        logger.debug("Calculando intercepto...")
        return np.mean(self.y) - self._inclination * np.mean(self.x)

    def predict(self, x_new):
        """Faz previsões para novos valores de x."""
        logger.info("Fazendo previsões...")
        if not isinstance(x_new, (int, float, np.ndarray)):
            raise TypeError("x_new deve ser um número ou um array NumPy.")
        try:
            resultado = self._intercept + (self._inclination * np.array(x_new))
            logger.debug(f"Previsão calculada: {resultado}")
            return resultado
        except Exception as e:
            logger.error(f"Erro durante a predição: {e}")
            raise