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


    det = calcular_apd90_detalhado(
    t,
    v,
    referencia="pico",
    **kw,
    )

    print("\n--- Diagnóstico do APD90 experimental ---")

    for chave in (
        "linha_base",
        "pico",
        "amplitude",
        "t_pico",
        "t_repolarizacao",
        "apd90",
        "repolarizou",
    ):
        print(f"{chave}: {det[chave]}")

    print(f"Passo temporal médio: {np.mean(np.diff(t)):.6f} ms")
    print(f"APD90 esperado: 66.790000 ms")
    print(f"Diferença: {det['apd90'] - 66.79:.6f} ms")

    # Índice do pico calculado pela função atual.
    pico_idx = int(np.argmin(np.abs(t - det["t_pico"])))

    # Definições alternativas do nível de repolarização.
    limiares = {
        "linha_base_medida": (
            det["linha_base"] + 0.1 * det["amplitude"]
        ),
        "linha_base_zero": (
            0.1 * det["pico"]
        ),
    }

    print("\n--- Comparação dos limiares de repolarização ---")

    for nome, limiar in limiares.items():
        cruzamentos = np.flatnonzero(
            (v[pico_idx:-1] > limiar)
            & (v[pico_idx + 1:] <= limiar)
        )

        if cruzamentos.size == 0:
            print(f"{nome}: cruzamento não encontrado")
            continue

        idx = pico_idx + int(cruzamentos[0])

        t1, t2 = t[idx], t[idx + 1]
        v1, v2 = v[idx], v[idx + 1]

        # Interpolação linear entre as amostras.
        t_rep = t1 + (limiar - v1) * (t2 - t1) / (v2 - v1)
        apd90 = t_rep - det["t_pico"]

        print(f"\n{nome}:")
        print(f"  Limiar: {limiar:.9f}")
        print(f"  Instante de repolarização: {t_rep:.9f} ms")
        print(f"  APD90: {apd90:.9f} ms")
        print(f"  Diferença para 66,79 ms: {apd90 - 66.79:.9f} ms")

    # Instante necessário para reproduzir o APD90 histórico.
    apd90_esperado = 66.79
    t_rep_esperado = det["t_pico"] + apd90_esperado

    # Tensão interpolada nesse instante.
    v_rep_esperado = float(np.interp(t_rep_esperado, t, v))

    # Limiar atualmente utilizado.
    limiar_atual = (
        det["linha_base"] + 0.1 * det["amplitude"]
    )

    # Linha de base equivalente para que o limiar
    # coincida com a tensão no instante esperado.
    linha_base_equivalente = (
        v_rep_esperado - 0.1 * det["pico"]
    ) / 0.9

    print("\n--- Investigação da referência histórica ---")
    print(f"Instante esperado: {t_rep_esperado:.9f} ms")
    print(f"Tensão nesse instante: {v_rep_esperado:.9f}")
    print(f"Limiar atual: {limiar_atual:.9f}")
    print(f"Linha de base atual: {det['linha_base']:.9f}")
    print(f"Linha de base equivalente: {linha_base_equivalente:.9f}")

    # Janela usada atualmente para estimar a linha de base.
    t_inicio, t_fim = kw["janela_linha_base_ms"]

    mascara = (t >= t_inicio) & (t <= t_fim)
    v_base = v[mascara]

    print("\n--- Estatísticas da janela de linha de base ---")
    print(f"Janela: {t_inicio} a {t_fim} ms")
    print(f"Número de amostras: {len(v_base)}")
    print(f"Linha de base atual: {det['linha_base']:.9f}")
    print(f"Linha de base equivalente: {linha_base_equivalente:.9f}")

    estatisticas = {
        "Média": np.mean(v_base),
        "Mediana": np.median(v_base),
        "Mínimo": np.min(v_base),
        "Percentil 10": np.percentile(v_base, 10),
        "Percentil 25": np.percentile(v_base, 25),
        "Percentil 75": np.percentile(v_base, 75),
        "Máximo": np.max(v_base),
    }

    for nome, valor in estatisticas.items():
        print(f"{nome}: {valor:.9f}")


    # Usar exatamente o mesmo intervalo da função estimar_linha_base:
    # início incluído e fim excluído.
    t_inicio, t_fim = kw["janela_linha_base_ms"]
    mascara = (t >= t_inicio) & (t < t_fim)

    linha_base_min = float(np.min(v[mascara]))

    # Manter o mesmo pico detectado pela implementação atual.
    pico_idx = int(np.argmin(np.abs(t - det["t_pico"])))
    pico = float(v[pico_idx])

    amplitude_min = pico - linha_base_min
    limiar_min = linha_base_min + 0.1 * amplitude_min

    # Procurar o primeiro cruzamento descendente após o pico.
    cruzamentos = np.flatnonzero(
        (v[pico_idx:-1] > limiar_min)
        & (v[pico_idx + 1:] <= limiar_min)
    )

    print("\n--- APD90 usando o mínimo da linha de base ---")
    print(f"Amostras na janela: {np.count_nonzero(mascara)}")
    print(f"Linha de base mínima: {linha_base_min:.9f}")
    print(f"Linha de base mediana: {det['linha_base']:.9f}")
    print(f"Limiar de repolarização: {limiar_min:.9f}")

    if cruzamentos.size:
        idx = pico_idx + int(cruzamentos[0])

        t1, t2 = t[idx], t[idx + 1]
        v1, v2 = v[idx], v[idx + 1]

        t_rep_min = float(
            t1 + (limiar_min - v1) * (t2 - t1) / (v2 - v1)
        )
        apd90_min = t_rep_min - float(t[pico_idx])

        print(f"Instante de repolarização: {t_rep_min:.9f} ms")
        print(f"APD90 calculado: {apd90_min:.9f} ms")
        print(f"Diferença para 66,79 ms: {apd90_min - 66.79:.9f} ms")
    else:
        print("O sinal não cruzou o limiar de repolarização.")


    apd90_ativacao = calcular_apd90(t, v, referencia="ativacao", **kw)
    v_norm, _, _ = normalizar_ap(t, v, **kw)
    assert len(t) == 1201
    assert t[0] == 0.0 and t[-1] == 300.0
    assert np.isclose(apd90_pico, 66.79, atol=0.1)
    assert apd90_pico <= apd90_ativacao <= apd90_pico + 15.0
    assert np.isclose(v_norm.max(), 1.0)
    assert np.isclose(v_norm.min(), 0.0)
