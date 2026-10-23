"""Testes da escolha de compromisso e da detecção de parâmetros no limite."""

import numpy as np
import pytest

from utils.otimizacao.selecao import (
    chave_compromisso,
    escolher_compromisso,
    mascara_validos,
    ordenar_por_compromisso,
    parametros_no_limite,
)


def test_dentro_da_tolerancia_vence_menor_erro_de_forma():
    F = np.array([
        [0.01, 0.20],   # APD90 ótimo, forma ruim
        [0.04, 0.05],   # dentro da tolerância de 5%, forma boa  <- esperado
        [0.20, 0.01],   # forma ótima, APD90 fora da tolerância
    ])
    assert escolher_compromisso(F, tolerancia_apd90=0.05) == 1


def test_fora_da_tolerancia_usa_a_soma_dos_erros():
    F = np.array([[0.30, 0.10], [0.20, 0.10], [0.10, 0.50]])
    assert escolher_compromisso(F, tolerancia_apd90=0.05) == 1


def test_front_inteiro_ruim_nao_e_tratado_como_bom():
    """Com a escala relativa antiga, qualquer ponto de um front ruim parecia 'ótimo'."""
    F = np.array([[0.80, 0.30], [0.82, 0.29]])
    chaves = [chave_compromisso(f, 0.05) for f in F]
    assert all(c[0] == 1 for c in chaves)  # ambas fora da tolerância


def test_penalizados_e_nao_finitos_sao_ignorados():
    F = np.array([[1e3, 1e3], [np.nan, 0.1], [0.03, 0.04]])
    assert mascara_validos(F, 1e3).tolist() == [False, False, True]
    assert ordenar_por_compromisso(F, tolerancia_apd90=0.05).tolist() == [2]


def test_levanta_erro_quando_todas_as_solucoes_falharam():
    with pytest.raises(RuntimeError):
        escolher_compromisso(np.array([[1e3, 1e3]]), tolerancia_apd90=0.05)


def test_front_com_uma_unica_solucao_1d_tambem_funciona():
    assert escolher_compromisso(np.array([0.02, 0.03]), tolerancia_apd90=0.05) == 0


def test_parametros_no_limite():
    limites = {"a": (0.0, 10.0), "b": (0.0, 10.0), "c": (0.0, 10.0)}
    valores = {"a": 0.05, "b": 5.0, "c": 9.9}
    assert parametros_no_limite(valores, limites) == [("a", "inferior"), ("c", "superior")]
