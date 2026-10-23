"""Problema multiobjetivo para o pymoo. A avaliação em si fica em ``AvaliadorAP``."""

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from ..simulacao import FalhaDeSimulacao
from .avaliador import CandidatoInvalido


class ProblemaOtimizacaoAP(ElementwiseProblem):
    """Objetivos (ambos minimizados): erro relativo do APD90 e RMSE da forma de onda.

    Somente falhas ESPERADAS (integrador que não converge, sem AP) viram penalidade.
    Erros de programação (KeyError, TypeError...) propagam e interrompem a execução,
    em vez de produzir uma otimização inteira de penalidades sem aviso.
    """

    def __init__(self, avaliador, limites, *, penalidade=1e3, **kwargs):
        self.avaliador = avaliador
        self.penalidade = float(penalidade)
        faixa = np.asarray([limites[nome] for nome in avaliador.nomes_parametros], dtype=float)
        # kwargs permite passar, por exemplo, elementwise_runner (avaliação em paralelo).
        super().__init__(n_var=len(faixa), n_obj=2, n_ieq_constr=0, xl=faixa[:, 0], xu=faixa[:, 1], **kwargs)

    def avaliar_candidato(self, x):
        """Atalho para inspecionar um candidato (devolve o dicionário completo de métricas)."""
        return self.avaliador.avaliar_candidato(x)

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            metricas = self.avaliador.avaliar_candidato(x)
            out["F"] = np.array(
                [metricas["erro_apd90_relativo"], metricas["erro_forma_onda"]], dtype=float
            )
        except (FalhaDeSimulacao, CandidatoInvalido, FloatingPointError, OverflowError, ZeroDivisionError):
            out["F"] = np.array([self.penalidade, self.penalidade], dtype=float)
