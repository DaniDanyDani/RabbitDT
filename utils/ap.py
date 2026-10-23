"""Leitura e métricas do potencial de ação (AP).

Convenções usadas em todo o projeto
-----------------------------------
* **Ativação**: instante do upstroke. Por padrão é o máximo de dV/dt (mesma
  definição de Gemmell et al., 2014); alternativamente, o cruzamento de 50% da
  amplitude (mais robusto a ruído).
* **APD90**: tempo entre a referência (ativação ou pico) e o cruzamento
  descendente de 10% da amplitude (90% de repolarização), com interpolação
  linear entre amostras.
* O potencial experimental pode estar normalizado (sem unidade). Por isso a
  forma de onda é comparada após normalização (linha de base 0, pico 1) e
  **alinhada no instante de ativação** (ver ``erro_forma_onda_alinhada``).
"""

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Leitura e validação
# ---------------------------------------------------------------------------
def _ler_matriz(caminho):
    """Lê tabela numérica aceitando separador por espaço/tab/vírgula/ponto e vírgula e 1 linha de cabeçalho."""
    ultimo_erro = None
    for delimitador in (None, ",", ";"):
        for pular in (0, 1):
            try:
                dados = np.loadtxt(
                    caminho, comments="#", delimiter=delimitador, skiprows=pular, ndmin=2
                )
            except ValueError as erro:
                ultimo_erro = erro
                continue
            if dados.shape[1] >= 2:
                return dados
    raise ValueError(
        f"O arquivo {caminho} deve ter pelo menos duas colunas numéricas: tempo e potencial. "
        f"Último erro de leitura: {ultimo_erro}"
    )


def carregar_ap(caminho):
    """Lê um arquivo de duas colunas: tempo (ms) e potencial."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de AP não encontrado: {caminho}")

    dados = _ler_matriz(caminho)
    tempo = np.asarray(dados[:, 0], dtype=float)
    potencial = np.asarray(dados[:, 1], dtype=float)
    mascara = np.isfinite(tempo) & np.isfinite(potencial)
    tempo, potencial = tempo[mascara], potencial[mascara]

    if len(tempo) < 3 or np.any(np.diff(tempo) <= 0):
        raise ValueError("O tempo do AP deve ter ao menos 3 pontos e ser estritamente crescente.")

    return tempo, potencial


def _validar_sinal(tempo, potencial):
    t = np.asarray(tempo, dtype=float)
    v = np.asarray(potencial, dtype=float)
    if t.ndim != 1 or v.ndim != 1 or len(t) != len(v) or len(t) < 3:
        raise ValueError("tempo e potencial devem ser vetores de mesmo tamanho com ao menos 3 pontos.")
    if np.any(np.diff(t) <= 0) or not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)):
        raise ValueError("tempo deve ser crescente e ambos os vetores devem conter valores finitos.")
    return t, v


# def estimar_linha_base(tempo, potencial, janela_ms):
#     """Estima o potencial de repouso pela mediana numa janela (início, fim)."""
#     t = np.asarray(tempo, dtype=float)
#     v = np.asarray(potencial, dtype=float)
#     inicio, fim = janela_ms
#     mascara = (t >= inicio) & (t < fim)
#     if not np.any(mascara):
#         raise ValueError(f"Não há amostras de AP na janela de linha de base {janela_ms} ms.")
#     return float(np.median(v[mascara]))

def estimar_linha_base(tempo, potencial, janela_ms):
    """Estima o potencial de repouso pelo mínimo na janela especificada."""
    t = np.asarray(tempo, dtype=float)
    v = np.asarray(potencial, dtype=float)

    inicio, fim = janela_ms
    mascara = (t >= inicio) & (t < fim)

    if not np.any(mascara):
        raise ValueError(
            f"Não há amostras de AP na janela de linha de base "
            f"{janela_ms} ms."
        )

    return float(np.min(v[mascara]))


def _suavizar(v, pontos):
    """Média móvel centrada (número ímpar de pontos), sem deslocar o sinal no tempo."""
    if pontos is None or pontos <= 1:
        return v
    pontos = int(pontos) | 1  # força ímpar
    meio = pontos // 2
    estendido = np.pad(v, meio, mode="edge")
    return np.convolve(estendido, np.ones(pontos) / pontos, mode="valid")


# ---------------------------------------------------------------------------
# Instante de ativação e APD90
# ---------------------------------------------------------------------------
def instante_ativacao(
    tempo,
    potencial,
    *,
    janela_linha_base_ms=(0.0, 25.0),
    inicio_busca_ms=25.0,
    metodo="dvdt_max",
    suavizar_pontos=3,
):
    """Instante (ms) do upstroke do AP, ou ``nan`` se não houver AP.

    ``metodo="dvdt_max"``: máximo de dV/dt entre ``inicio_busca_ms`` e o pico
    (com refinamento parabólico sub-amostra). ``metodo="meia_amplitude"``:
    primeiro cruzamento ascendente de 50% da amplitude.
    """
    t, v = _validar_sinal(tempo, potencial)
    linha_base = estimar_linha_base(t, v, janela_linha_base_ms)
    indices = np.flatnonzero(t >= inicio_busca_ms)
    if indices.size < 3:
        return float("nan")
    pico_idx = int(indices[np.argmax(v[indices])])
    amplitude = v[pico_idx] - linha_base
    if amplitude <= 0:
        return float("nan")
    trecho = np.arange(indices[0], pico_idx + 1)

    if metodo == "dvdt_max":
        dvdt = np.gradient(_suavizar(v, suavizar_pontos), t)
        k = int(trecho[np.argmax(dvdt[trecho])])
        if 0 < k < len(t) - 1:
            y0, y1, y2 = dvdt[k - 1], dvdt[k], dvdt[k + 1]
            denominador = y0 - 2.0 * y1 + y2
            if denominador < 0:
                delta = float(np.clip(0.5 * (y0 - y2) / denominador, -1.0, 1.0))
                return float(t[k] + delta * 0.5 * (t[k + 1] - t[k - 1]))
        return float(t[k])

    if metodo == "meia_amplitude":
        limiar = linha_base + 0.5 * amplitude
        cruzamentos = np.flatnonzero((v[trecho[:-1]] < limiar) & (v[trecho[1:]] >= limiar))
        if cruzamentos.size == 0:
            return float("nan")
        i = int(trecho[cruzamentos[0]])
        return float(t[i] + (limiar - v[i]) * (t[i + 1] - t[i]) / (v[i + 1] - v[i]))

    raise ValueError("metodo deve ser 'dvdt_max' ou 'meia_amplitude'.")


def calcular_apd90_detalhado(
    tempo,
    potencial,
    *,
    janela_linha_base_ms=(0.0, 25.0),
    inicio_busca_pico_ms=25.0,
    referencia="ativacao",
    metodo_ativacao="dvdt_max",
    suavizar_pontos=3,
):
    """Calcula APD90 e devolve também os instantes usados.

    Retorna dict com: ``apd90`` (ms), ``t_referencia``, ``t_ativacao``,
    ``t_pico``, ``t_repolarizacao``, ``linha_base``, ``pico``, ``amplitude`` e
    ``repolarizou`` (False quando o sinal não cruza 90% de repolarização antes
    do fim da janela; nesse caso ``apd90`` é ``nan``).

    ``referencia="ativacao"`` mede a partir do upstroke (padrão da literatura);
    ``referencia="pico"`` reproduz a definição antiga do projeto.
    """
    if referencia not in ("ativacao", "pico"):
        raise ValueError("referencia deve ser 'ativacao' ou 'pico'.")
    t, v = _validar_sinal(tempo, potencial)
    nan = float("nan")
    saida = {
        "apd90": nan, "t_referencia": nan, "t_ativacao": nan, "t_pico": nan,
        "t_repolarizacao": nan, "linha_base": nan, "pico": nan, "amplitude": nan,
        "repolarizou": False,
    }

    linha_base = estimar_linha_base(t, v, janela_linha_base_ms)
    indices = np.flatnonzero(t >= inicio_busca_pico_ms)
    if indices.size == 0:
        return saida
    pico_idx = int(indices[np.argmax(v[indices])])
    amplitude = float(v[pico_idx] - linha_base)
    saida.update(linha_base=linha_base, pico=float(v[pico_idx]), amplitude=amplitude,
                 t_pico=float(t[pico_idx]))
    if amplitude <= 0:
        return saida

    t_ativ = instante_ativacao(
        t, v,
        janela_linha_base_ms=janela_linha_base_ms,
        inicio_busca_ms=inicio_busca_pico_ms,
        metodo=metodo_ativacao,
        suavizar_pontos=suavizar_pontos,
    )
    saida["t_ativacao"] = t_ativ
    t_ref = t_ativ if referencia == "ativacao" else saida["t_pico"]
    if not np.isfinite(t_ref):
        return saida
    saida["t_referencia"] = float(t_ref)

    limiar = linha_base + 0.1 * amplitude
    cruzamentos = np.flatnonzero((v[pico_idx:-1] > limiar) & (v[pico_idx + 1:] <= limiar))
    if cruzamentos.size == 0:
        return saida
    idx = int(pico_idx + cruzamentos[0])
    t1, t2, v1, v2 = t[idx], t[idx + 1], v[idx], v[idx + 1]
    t_rep = float(t2) if v2 == v1 else float(t1 + (limiar - v1) * (t2 - t1) / (v2 - v1))
    saida.update(t_repolarizacao=t_rep, apd90=float(t_rep - t_ref), repolarizou=True)
    return saida


def calcular_apd90(
    tempo,
    potencial,
    *,
    janela_linha_base_ms=(0.0, 25.0),
    inicio_busca_pico_ms=25.0,
    referencia="ativacao",
    metodo_ativacao="dvdt_max",
    suavizar_pontos=3,
):
    """APD90 em ms (``nan`` se o sinal não repolariza 90% dentro da janela)."""
    return calcular_apd90_detalhado(
        tempo, potencial,
        janela_linha_base_ms=janela_linha_base_ms,
        inicio_busca_pico_ms=inicio_busca_pico_ms,
        referencia=referencia,
        metodo_ativacao=metodo_ativacao,
        suavizar_pontos=suavizar_pontos,
    )["apd90"]


# ---------------------------------------------------------------------------
# Normalização e comparação de formas de onda
# ---------------------------------------------------------------------------
def normalizar_ap(
    tempo,
    potencial,
    *,
    janela_linha_base_ms=(0.0, 25.0),
    inicio_busca_pico_ms=25.0,
):
    """Normaliza a forma de onda para linha de base 0 e pico 1.

    A normalização é útil quando a curva experimental não está em mV. Não deve
    ser usada para avaliar amplitude absoluta ou nível de repouso em mV.
    Retorna (potencial_normalizado, linha_base, pico).
    """
    t = np.asarray(tempo, dtype=float)
    v = np.asarray(potencial, dtype=float)
    linha_base = estimar_linha_base(t, v, janela_linha_base_ms)
    indices_validos = np.flatnonzero(t >= inicio_busca_pico_ms)
    if indices_validos.size == 0:
        raise ValueError("Não há pontos suficientes para encontrar o pico do AP.")
    pico = float(np.max(v[indices_validos]))
    amplitude = pico - linha_base
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("A amplitude do AP deve ser positiva para normalização.")
    normalizado = np.clip((v - linha_base) / amplitude, 0.0, 1.0)
    return normalizado, linha_base, pico


def erro_forma_onda(tempo_sim, potencial_sim_norm, tempo_alvo, potencial_alvo_norm):
    """RMSE entre formas de onda normalizadas no MESMO eixo de tempo absoluto.

    ATENÇÃO: não alinha as curvas. Se o upstroke do simulado e o do alvo
    ocorrem em instantes diferentes, o erro é dominado pelo deslocamento
    temporal. Para otimização use ``erro_forma_onda_alinhada``.
    """
    t_sim = np.asarray(tempo_sim, dtype=float)
    v_sim = np.asarray(potencial_sim_norm, dtype=float)
    t_alvo = np.asarray(tempo_alvo, dtype=float)
    v_alvo = np.asarray(potencial_alvo_norm, dtype=float)

    inicio = max(float(t_sim[0]), float(t_alvo[0]))
    fim = min(float(t_sim[-1]), float(t_alvo[-1]))
    mascara = (t_alvo >= inicio) & (t_alvo <= fim)
    if np.count_nonzero(mascara) < 3:
        raise ValueError("Simulado e alvo não têm intervalo temporal suficiente em comum.")

    v_sim_interp = np.interp(t_alvo[mascara], t_sim, v_sim)
    diferenca = v_sim_interp - v_alvo[mascara]
    return float(np.sqrt(np.mean(diferenca ** 2)))


def erro_forma_onda_alinhada(
    tempo_sim,
    potencial_sim_norm,
    t_ativacao_sim,
    tempo_alvo,
    potencial_alvo_norm,
    t_ativacao_alvo,
    *,
    antes_ms=10.0,
    depois_ms=None,
    passo_ms=None,
):
    """RMSE entre formas de onda normalizadas, alinhadas no instante de ativação.

    Ambas as curvas são reamostradas num eixo relativo à ativação
    ``tau = t - t_ativacao``, de ``-antes_ms`` até ``depois_ms`` (ou até onde
    as duas existirem). Assim, o erro mede morfologia e não o atraso entre o
    estímulo e o upstroke.
    """
    rel_sim = np.asarray(tempo_sim, dtype=float) - float(t_ativacao_sim)
    rel_alvo = np.asarray(tempo_alvo, dtype=float) - float(t_ativacao_alvo)
    v_sim = np.asarray(potencial_sim_norm, dtype=float)
    v_alvo = np.asarray(potencial_alvo_norm, dtype=float)

    inicio = max(-float(antes_ms), float(rel_sim[0]), float(rel_alvo[0]))
    fim = min(float(rel_sim[-1]), float(rel_alvo[-1]))
    if depois_ms is not None:
        fim = min(fim, float(depois_ms))
    if not fim > inicio:
        raise ValueError("Simulado e alvo não têm intervalo comum em torno da ativação.")

    passo = float(np.median(np.diff(rel_alvo))) if passo_ms is None else float(passo_ms)
    grade = np.arange(inicio, fim + 1e-9, passo)
    if grade.size < 3:
        raise ValueError("Janela de comparação com menos de 3 pontos.")
    diferenca = np.interp(grade, rel_sim, v_sim) - np.interp(grade, rel_alvo, v_alvo)
    return float(np.sqrt(np.mean(diferenca ** 2)))
