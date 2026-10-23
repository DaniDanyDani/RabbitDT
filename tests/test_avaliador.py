"""Testes do avaliador com o modelo de Mahajan (alvo sintético gerado pelo próprio modelo)."""

import numpy as np
import pytest

from utils.ap import normalizar_ap
from utils.modelo_celular import get_default_params
from utils.otimizacao import configuracao as cfg
from utils.otimizacao.avaliador import AvaliadorAP, CandidatoInvalido
from utils.simulacao import simular_ultimo_ap

NOMES = cfg.PARAMETROS_OTIMIZAVEIS
PADRAO = get_default_params()
X_PADRAO = np.array([PADRAO[n] for n in NOMES])
CONFIG = {
    "num_pulsos": 2, "bcl_ms": 450.0, "estimulo_amplitude": -15.0, "estimulo_duracao_ms": 3.0,
    "estimulo_offset_ms": 0.0, "janela_antes_estimulo_ms": 50.0, "janela_depois_estimulo_ms": 250.0,
    "dt_ap_ms": 0.25, "dt_previo_ms": 2.0, "max_step_ms": 2.0, "atol": 1e-6, "rtol": 1e-4,
}


def _x_com_fatores(**fatores):
    return np.array([PADRAO[n] * fatores.get(n, 1.0) for n in NOMES])


def _alvo_sintetico(x, ms_antes_do_estimulo):
    """Recorta e normaliza o AP simulado como um registro experimental (tempo 0..300 ms, 0..1)."""
    parametros = dict(PADRAO)
    parametros.update({n: float(v) for n, v in zip(NOMES, x)})
    config = dict(CONFIG, janela_depois_estimulo_ms=300.0 - ms_antes_do_estimulo + 50.0)
    sim = simular_ultimo_ap(parametros, **config)
    t, v = sim["tempo_ms"], sim["potencial_mV"]
    inicio = 50.0 - ms_antes_do_estimulo
    mascara = (t >= inicio - 1e-9) & (t <= inicio + 300.0 + 1e-9)
    t_alvo = t[mascara] - inicio
    v_norm, _, _ = normalizar_ap(t_alvo, v[mascara], janela_linha_base_ms=(0, ms_antes_do_estimulo),
                                 inicio_busca_pico_ms=ms_antes_do_estimulo)
    return t_alvo, v_norm


def _avaliador(t_alvo, v_alvo, ms_antes_do_estimulo):
    return AvaliadorAP(
        nomes_parametros=NOMES, valores_iniciais=PADRAO, tempo_alvo=t_alvo, potencial_alvo=v_alvo,
        configuracao_simulacao=CONFIG,
        baseline_alvo_ms=(0.0, ms_antes_do_estimulo), inicio_pico_alvo_ms=ms_antes_do_estimulo,
        baseline_sim_ms=cfg.BASELINE_SIM_MS, inicio_pico_sim_ms=cfg.INICIO_BUSCA_PICO_SIM_MS,
        referencia_apd=cfg.REFERENCIA_APD, metodo_ativacao=cfg.METODO_ATIVACAO,
        suavizar_pontos=cfg.SUAVIZAR_PONTOS, janela_forma_antes_ms=cfg.JANELA_FORMA_ANTES_MS,
        janela_forma_depois_apd=cfg.JANELA_FORMA_DEPOIS_APD, amplitude_minima_sim_mV=cfg.AMPLITUDE_MINIMA_SIM_MV,
    )


X_VERDADE = _x_com_fatores(gca=0.4, gtof=1.3)


def test_erro_e_praticamente_zero_nos_parametros_verdadeiros():
    t_alvo, v_alvo = _alvo_sintetico(X_VERDADE, ms_antes_do_estimulo=25.0)
    metricas = _avaliador(t_alvo, v_alvo, 25.0).avaliar_candidato(X_VERDADE)
    assert metricas["erro_apd90_relativo"] < 0.005
    assert metricas["erro_forma_onda"] < 0.01
    assert metricas["apd90_repolarizou"]


def test_erro_nao_depende_de_onde_o_upstroke_cai_no_registro_experimental():
    """O mesmo AP com 25 ms ou 40 ms de repouso antes do upstroke deve dar o mesmo erro (alinhamento)."""
    erros = []
    for antes in (25.0, 40.0):
        t_alvo, v_alvo = _alvo_sintetico(X_VERDADE, ms_antes_do_estimulo=antes)
        metricas = _avaliador(t_alvo, v_alvo, antes).avaliar_candidato(X_VERDADE)
        erros.append((metricas["erro_apd90_relativo"], metricas["erro_forma_onda"]))
    assert all(e[0] < 0.005 and e[1] < 0.01 for e in erros)


def test_parametros_errados_aumentam_os_dois_erros():
    t_alvo, v_alvo = _alvo_sintetico(X_VERDADE, ms_antes_do_estimulo=25.0)
    avaliador = _avaliador(t_alvo, v_alvo, 25.0)
    certo = avaliador.avaliar_candidato(X_VERDADE)
    errado = avaliador.avaliar_candidato(_x_com_fatores(gca=0.8, gtof=1.3))
    assert errado["erro_apd90_relativo"] > 10 * certo["erro_apd90_relativo"] + 0.05
    assert errado["erro_forma_onda"] > 10 * certo["erro_forma_onda"] + 0.01


def test_sem_ina_a_resposta_passiva_nao_conta_como_potencial_de_acao():
    t_alvo, v_alvo = _alvo_sintetico(X_VERDADE, ms_antes_do_estimulo=25.0)
    avaliador = _avaliador(t_alvo, v_alvo, 25.0)
    parametros = dict(PADRAO, gna=0.0)
    simulacao = simular_ultimo_ap(parametros, **CONFIG)
    with pytest.raises(CandidatoInvalido):
        avaliador.metricas_da_simulacao(simulacao)


def test_nome_de_parametro_inexistente_e_erro_de_programacao_e_nao_penalidade():
    t_alvo, v_alvo = _alvo_sintetico(X_VERDADE, ms_antes_do_estimulo=25.0)
    with pytest.raises(KeyError):
        AvaliadorAP(
            nomes_parametros=("gca", "parametro_que_nao_existe"), valores_iniciais=PADRAO,
            tempo_alvo=t_alvo, potencial_alvo=v_alvo, configuracao_simulacao=CONFIG,
            baseline_alvo_ms=(0, 25), inicio_pico_alvo_ms=25,
            baseline_sim_ms=cfg.BASELINE_SIM_MS, inicio_pico_sim_ms=cfg.INICIO_BUSCA_PICO_SIM_MS,
        )


def test_regime_estacionario_devolve_a_deriva_do_apd90():
    t_alvo, v_alvo = _alvo_sintetico(X_VERDADE, ms_antes_do_estimulo=25.0)
    metricas = _avaliador(t_alvo, v_alvo, 25.0).avaliar_regime_estacionario(X_VERDADE, num_pulsos=4)
    assert metricas["num_pulsos"] == 4
    assert np.isfinite(metricas["deriva_apd90_ms"])
