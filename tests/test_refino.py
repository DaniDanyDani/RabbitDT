"""O refino local deve encontrar o mínimo de uma função conhecida (avaliador falso, sem o modelo)."""

import numpy as np

from utils.otimizacao.avaliador import CandidatoInvalido
from utils.otimizacao.refino import refinar_em_regime_estacionario


class _AvaliadorFalso:
    nomes_parametros = ("a", "b", "c")
    alvo = np.array([1.3, 0.7, 4.0])

    def avaliar_candidato(self, x, **sobrescritas):
        assert sobrescritas["num_pulsos"] == 7  # o refino repassa o número de pulsos
        if x[2] > 9.0:
            raise CandidatoInvalido("região inviável")
        erro = float(np.sqrt(np.mean(((x - self.alvo) / self.alvo) ** 2)))
        return {"erro_apd90_relativo": erro, "erro_forma_onda": 0.5 * erro, "apd90_sim_ms": 60.0}


def test_refino_encontra_o_minimo_dentro_dos_limites():
    limites = {"a": (0.1, 1.9), "b": (0.1, 1.9), "c": (0.4, 10.0)}
    resultado = refinar_em_regime_estacionario(
        _AvaliadorFalso(), x0=[1.0, 1.0, 5.0], limites=limites, num_pulsos=7, max_avaliacoes=400
    )
    assert np.allclose(resultado["x"], _AvaliadorFalso.alvo, rtol=0.02)
    assert resultado["J"] < 0.02
    assert 0 < resultado["n_avaliacoes"] <= 400


def test_refino_respeita_os_limites():
    limites = {"a": (0.1, 1.0), "b": (0.1, 1.9), "c": (0.4, 10.0)}  # o alvo de 'a' (1,3) está fora
    resultado = refinar_em_regime_estacionario(
        _AvaliadorFalso(), x0=[0.5, 1.0, 5.0], limites=limites, num_pulsos=7, max_avaliacoes=300
    )
    assert resultado["x"][0] <= 1.0 + 1e-12
