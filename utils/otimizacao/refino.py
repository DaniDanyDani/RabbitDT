"""Refino local em regime (quase) estacionário. Usa apenas scipy (sem pymoo)."""

import numpy as np
from scipy.optimize import minimize

from ..simulacao import FalhaDeSimulacao
from .avaliador import CandidatoInvalido


def refinar_em_regime_estacionario(
    avaliador,
    x0,
    limites,
    *,
    num_pulsos,
    sobrescritas_simulacao=None,
    peso_apd90=1.0,
    max_avaliacoes=300,
    penalidade=1e3,
    registrar=None,
):
    """Nelder-Mead nos parâmetros normalizados, com ``num_pulsos`` batimentos por avaliação.

    Minimiza ``J = peso_apd90 * erro_apd90 + erro_forma_onda``. A busca inicial do NSGA-II usa
    poucos pulsos; como o estado ainda deriva depois disso, o APD90 em regime pode diferir em
    vários por cento. Este refino corrige essa diferença a partir da melhor solução da busca.

    Devolve dict com ``x``, ``J``, ``metricas`` (da melhor avaliação), ``n_avaliacoes`` e ``historico``.
    """
    nomes = avaliador.nomes_parametros
    inferior = np.array([limites[nome][0] for nome in nomes], dtype=float)
    superior = np.array([limites[nome][1] for nome in nomes], dtype=float)
    faixa = superior - inferior
    u0 = np.clip((np.asarray(x0, dtype=float) - inferior) / faixa, 0.0, 1.0)

    sobrescritas = dict(sobrescritas_simulacao or {})
    sobrescritas["num_pulsos"] = int(num_pulsos)
    melhor = {"J": np.inf, "x": inferior + u0 * faixa, "metricas": None}
    historico = []

    def objetivo(u):
        x = inferior + np.clip(u, 0.0, 1.0) * faixa
        try:
            metricas = avaliador.avaliar_candidato(x, **sobrescritas)
        except (FalhaDeSimulacao, CandidatoInvalido, FloatingPointError, OverflowError, ZeroDivisionError):
            historico.append((penalidade, np.nan, np.nan))
            return penalidade
        J = peso_apd90 * metricas["erro_apd90_relativo"] + metricas["erro_forma_onda"]
        historico.append((J, metricas["erro_apd90_relativo"], metricas["erro_forma_onda"]))
        if J < melhor["J"]:
            melhor.update(J=float(J), x=x.copy(), metricas=metricas)
        if registrar is not None:
            registrar(len(historico), J, metricas)
        return J

    n = len(u0)
    simplex = np.tile(u0, (n + 1, 1))
    for i in range(n):
        passo = 0.05 if u0[i] + 0.05 <= 1.0 else -0.05
        simplex[i + 1, i] = u0[i] + passo

    minimize(
        objetivo,
        u0,
        method="Nelder-Mead",
        bounds=[(0.0, 1.0)] * n,
        options={
            "maxfev": int(max_avaliacoes),
            "xatol": 1e-3,
            "fatol": 1e-5,
            "adaptive": True,
            "initial_simplex": simplex,
        },
    )
    if melhor["metricas"] is None:
        raise CandidatoInvalido("O refino não encontrou nenhum candidato válido.")
    return {
        "x": melhor["x"],
        "J": melhor["J"],
        "metricas": melhor["metricas"],
        "n_avaliacoes": len(historico),
        "historico": historico,
    }
