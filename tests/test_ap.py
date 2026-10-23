"""Testes das métricas de AP com sinais sintéticos de resposta conhecida."""

from pathlib import Path

import numpy as np
import pytest

from utils.ap import (
    calcular_apd90,
    calcular_apd90_detalhado,
    carregar_ap,
    erro_forma_onda,
    erro_forma_onda_alinhada,
    instante_ativacao,
    normalizar_ap,
)
from utils.otimizacao.configuracao import BASELINE_ALVO_MS, INICIO_BUSCA_PICO_ALVO_MS

LARGURA_SUBIDA = 0.4
LARGURA_QUEDA = 12.0


def _ap_sintetico(passo_ms, t_ativ=40.0, t_queda=80.0, duracao=300.0):
    """AP liso: subida logística em t_ativ e queda logística centrada em t_queda."""
    t = np.arange(0.0, duracao + passo_ms / 2, passo_ms)
    subida = 1.0 / (1.0 + np.exp(-(t - t_ativ) / LARGURA_SUBIDA))
    queda = 1.0 / (1.0 + np.exp((t - t_queda) / LARGURA_QUEDA))
    return t, subida * queda


def _referencias_finas(**kwargs):
    """Valores de referência obtidos numa grade 250x mais fina (sem erro de amostragem)."""
    t, v = _ap_sintetico(0.001, **kwargs)
    pico = float(v.max())
    t_pico = float(t[np.argmax(v)])
    t_ativ = float(t[np.argmax(np.gradient(v, t))])
    t_cruzamento = kwargs.get("t_queda", 80.0) + LARGURA_QUEDA * np.log(1.0 / (0.1 * pico) - 1.0)
    return t_ativ, t_pico, t_cruzamento


def test_ativacao_coincide_com_o_maximo_de_dvdt_e_com_meia_amplitude():
    t, v = _ap_sintetico(0.25)
    t_ativ_ref, _, _ = _referencias_finas()
    por_dvdt = instante_ativacao(t, v, janela_linha_base_ms=(0, 25), inicio_busca_ms=25, metodo="dvdt_max")
    por_meia = instante_ativacao(t, v, janela_linha_base_ms=(0, 25), inicio_busca_ms=25, metodo="meia_amplitude")
    assert abs(por_dvdt - t_ativ_ref) < 0.15
    assert abs(por_meia - 40.0) < 0.15


def test_apd90_confere_com_valor_analitico_nas_duas_referencias():
    t, v = _ap_sintetico(0.25)
    t_ativ_ref, t_pico_ref, t_cruzamento = _referencias_finas()
    kw = {"janela_linha_base_ms": (0, 25), "inicio_busca_pico_ms": 25}
    apd_ativ = calcular_apd90(t, v, referencia="ativacao", **kw)
    apd_pico = calcular_apd90(t, v, referencia="pico", **kw)
    assert abs(apd_ativ - (t_cruzamento - t_ativ_ref)) < 0.2
    assert abs(apd_pico - (t_cruzamento - t_pico_ref)) < 0.4
    assert apd_ativ > apd_pico  # a ativação antecede o pico


def test_apd90_nao_depende_da_resolucao_temporal():
    kw = {"janela_linha_base_ms": (0, 25), "inicio_busca_pico_ms": 25}
    apds = [calcular_apd90(*_ap_sintetico(passo), **kw) for passo in (0.1, 0.25, 0.5)]
    assert max(apds) - min(apds) < 0.3


def test_apd90_e_nan_quando_nao_repolariza_na_janela():
    t, v = _ap_sintetico(0.25, t_queda=400.0)  # a queda acontece depois do fim do registro
    detalhes = calcular_apd90_detalhado(t, v, janela_linha_base_ms=(0, 25), inicio_busca_pico_ms=25)
    assert not detalhes["repolarizou"]
    assert np.isnan(detalhes["apd90"])


def test_apd90_valida_a_referencia():
    t, v = _ap_sintetico(0.25)
    with pytest.raises(ValueError):
        calcular_apd90(t, v, referencia="outra")


def test_normalizacao_limita_a_faixa_de_0_a_1():
    t, v = _ap_sintetico(0.25)
    v_norm, linha_base, pico = normalizar_ap(t, 30.0 + 90.0 * v, janela_linha_base_ms=(0, 25), inicio_busca_pico_ms=25)
    assert np.isclose(v_norm.max(), 1.0) and np.isclose(v_norm.min(), 0.0)
    assert np.isclose(linha_base, 30.0, atol=1e-6)
    assert np.isclose(pico, 30.0 + 90.0 * v.max())


def test_forma_de_onda_sem_alinhamento_e_dominada_pelo_deslocamento_e_com_alinhamento_nao():
    """Mesma forma de onda, upstroke em 40 ms (alvo) e 65 ms (simulado, como o estímulo em 50 ms)."""
    t_alvo, v_alvo = _ap_sintetico(0.25, t_ativ=40.0, t_queda=80.0)
    t_sim, v_sim = _ap_sintetico(0.25, t_ativ=65.0, t_queda=105.0)
    alvo_norm, _, _ = normalizar_ap(t_alvo, v_alvo, janela_linha_base_ms=(0, 25), inicio_busca_pico_ms=25)
    sim_norm, _, _ = normalizar_ap(t_sim, v_sim, janela_linha_base_ms=(0, 50), inicio_busca_pico_ms=50)

    sem_alinhar = erro_forma_onda(t_sim, sim_norm, t_alvo, alvo_norm)
    ativ_alvo = instante_ativacao(t_alvo, alvo_norm, janela_linha_base_ms=(0, 25), inicio_busca_ms=25)
    ativ_sim = instante_ativacao(t_sim, sim_norm, janela_linha_base_ms=(0, 50), inicio_busca_ms=50)
    alinhado = erro_forma_onda_alinhada(
        t_sim, sim_norm, ativ_sim, t_alvo, alvo_norm, ativ_alvo, antes_ms=10.0, depois_ms=150.0
    )
    assert sem_alinhar > 0.1
    assert alinhado < 0.01


def test_erro_alinhado_cresce_quando_a_forma_realmente_difere():
    t_alvo, v_alvo = _ap_sintetico(0.25, t_ativ=40.0, t_queda=80.0)
    t_sim, v_sim = _ap_sintetico(0.25, t_ativ=65.0, t_queda=145.0)  # APD 40 ms maior
    alvo_norm, _, _ = normalizar_ap(t_alvo, v_alvo, janela_linha_base_ms=(0, 25), inicio_busca_pico_ms=25)
    sim_norm, _, _ = normalizar_ap(t_sim, v_sim, janela_linha_base_ms=(0, 50), inicio_busca_pico_ms=50)
    erro = erro_forma_onda_alinhada(t_sim, sim_norm, 65.0, t_alvo, alvo_norm, 40.0, antes_ms=10.0, depois_ms=150.0)
    assert erro > 0.1


def test_ap_experimental_do_usuario_se_o_arquivo_existir():
    """Mantém a verificação original (APD90 = 66,79 ms medido a partir do pico) quando o arquivo está presente."""
    arquivo = Path(__file__).resolve().parents[1] / "data" / "ap_mayra.txt"
    if not arquivo.is_file():
        pytest.skip("data/ap_mayra.txt não encontrado (coloque o registro experimental na pasta data/).")
    t, v = carregar_ap(arquivo)
    kw = {"janela_linha_base_ms": BASELINE_ALVO_MS, "inicio_busca_pico_ms": INICIO_BUSCA_PICO_ALVO_MS}
    apd90_pico = calcular_apd90(t, v, referencia="pico", **kw)
    apd90_ativacao = calcular_apd90(t, v, referencia="ativacao", **kw)
    v_norm, _, _ = normalizar_ap(t, v, **kw)
    assert len(t) == 1201
    assert t[0] == 0.0 and t[-1] == 300.0
    assert np.isclose(apd90_pico, 66.79, atol=0.1)
    assert apd90_pico <= apd90_ativacao <= apd90_pico + 15.0
    assert np.isclose(v_norm.max(), 1.0)
    assert np.isclose(v_norm.min(), 0.0)
