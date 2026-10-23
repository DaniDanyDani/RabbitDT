"""Rotinas para executar o Mahajan e extrair a janela do último potencial de ação."""

import warnings

import numpy as np

from .modelo_celular import simulate_cell


class FalhaDeSimulacao(RuntimeError):
    """O integrador falhou ou não cobriu a janela pedida (falha esperada para certos parâmetros).

    É diferente de erros de programação (KeyError, TypeError...), que devem
    interromper a execução em vez de virar penalidade silenciosa.
    """


def _criar_grade_de_saida(
    *,
    tempo_fim_ms,
    inicio_janela_ms,
    fim_janela_ms,
    dt_janela_ms,
    dt_previo_ms,
    num_pulsos,
    bcl_ms,
    estimulo_offset_ms,
    estimulo_duracao_ms,
):
    """Usa amostragem esparsa no condicionamento e densa na janela analisada."""
    if dt_janela_ms <= 0 or dt_previo_ms <= 0:
        raise ValueError("Os passos temporais devem ser positivos.")

    previa = np.arange(0.0, inicio_janela_ms, dt_previo_ms)
    n_janela = int(round((fim_janela_ms - inicio_janela_ms) / dt_janela_ms))
    densa = inicio_janela_ms + np.arange(n_janela + 1, dtype=float) * dt_janela_ms
    densa = densa[densa <= fim_janela_ms + 1e-9]

    # Inclui os limites dos pulsos para que o integrador pare exatamente nas
    # descontinuidades do estímulo, mesmo quando a saída de condicionamento é esparsa.
    limites_estimulo = []
    for i in range(num_pulsos):
        inicio = i * bcl_ms + estimulo_offset_ms
        fim = inicio + estimulo_duracao_ms
        if 0.0 <= inicio <= tempo_fim_ms:
            limites_estimulo.append(inicio)
        if 0.0 <= fim <= tempo_fim_ms:
            limites_estimulo.append(fim)

    grade = np.unique(np.concatenate([previa, densa, np.asarray(limites_estimulo, dtype=float)]))
    grade = grade[(grade >= 0.0) & (grade <= tempo_fim_ms + 1e-9)]
    if grade.size < 3 or not np.isclose(grade[-1], tempo_fim_ms):
        grade = np.unique(np.append(grade, tempo_fim_ms))
    return grade


def simular_ultimo_ap(
    parametros_celulares=None,
    *,
    num_pulsos=8,
    bcl_ms=450.0,
    estimulo_amplitude=-15.0,
    estimulo_duracao_ms=3.0,
    estimulo_offset_ms=0.0,
    janela_antes_estimulo_ms=50.0,
    janela_depois_estimulo_ms=250.0,
    dt_ap_ms=0.25,
    dt_previo_ms=2.0,
    max_step_ms=0.5,
    atol=1e-6,
    rtol=1e-6,
):
    """Simula pulsos repetidos e retorna apenas os 300 ms da janela do último AP.

    A janela começa `janela_antes_estimulo_ms` antes do último pulso; por padrão,
    o estímulo aparece no tempo local t=50 ms e a janela total dura 300 ms.
    Os parâmetros celulares não informados mantêm os padrões do modelo Mahajan.
    O estado inicial é o padrão do modelo: com poucos pulsos o resultado ainda não
    é o regime estacionário (a [Na]i, por exemplo, deriva por centenas de batimentos).
    Levanta ``FalhaDeSimulacao`` se o integrador não chegar ao fim da janela.

    Retorno: dict com `tempo_ms` (relativo à janela), `potencial_mV`, `estados`,
    `tempo_global_ms`, `inicio_janela_global_ms` e `estimulo_local_ms`.
    """
    if num_pulsos < 1:
        raise ValueError("num_pulsos deve ser >= 1.")
    if bcl_ms <= 0 or estimulo_duracao_ms < 0 or dt_ap_ms <= 0:
        raise ValueError("BCL e dt devem ser positivos; duração do estímulo não pode ser negativa.")
    if janela_antes_estimulo_ms < 0 or janela_depois_estimulo_ms <= 0:
        raise ValueError("A janela antes deve ser >= 0 e a janela depois deve ser > 0.")
    if estimulo_offset_ms < 0:
        raise ValueError("estimulo_offset_ms deve ser >= 0.")

    ultimo_estimulo_global = (num_pulsos - 1) * bcl_ms + estimulo_offset_ms
    inicio_janela = ultimo_estimulo_global - janela_antes_estimulo_ms
    fim_janela = ultimo_estimulo_global + janela_depois_estimulo_ms
    if inicio_janela < 0:
        raise ValueError("A simulação começa antes da janela solicitada; aumente num_pulsos ou reduza a janela anterior.")

    t_eval = _criar_grade_de_saida(
        tempo_fim_ms=fim_janela,
        inicio_janela_ms=inicio_janela,
        fim_janela_ms=fim_janela,
        dt_janela_ms=dt_ap_ms,
        dt_previo_ms=dt_previo_ms,
        num_pulsos=num_pulsos,
        bcl_ms=bcl_ms,
        estimulo_offset_ms=estimulo_offset_ms,
        estimulo_duracao_ms=estimulo_duracao_ms,
    )

    # simulate_cell avisa (warnings) quando o solver falha antes do fim; aqui a falha
    # é tratada abaixo como FalhaDeSimulacao, então o aviso só poluiria o log.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        t, V, states = simulate_cell(
            cell_params={} if parametros_celulares is None else parametros_celulares,
            t_eval=t_eval,
            stim_amplitude=estimulo_amplitude,
            stim_duration=estimulo_duracao_ms,
            stim_period=bcl_ms,
            stim_offset=estimulo_offset_ms,
            atol=atol,
            rtol=rtol,
            max_step=max_step_ms,
        )

    if len(t) == 0 or t[-1] < fim_janela - 1e-6:
        raise FalhaDeSimulacao(
            f"A integração terminou em {t[-1] if len(t) else 'nenhum instante'} ms; "
            f"era esperado chegar a {fim_janela} ms."
        )

    mascara = (t >= inicio_janela - 1e-9) & (t <= fim_janela + 1e-9)
    t_global = t[mascara]
    t_local = t_global - inicio_janela
    V_janela = V[mascara]
    estados_janela = {nome: valores[mascara] for nome, valores in states.items()}

    # Exige que a janela final esteja amostrada até os extremos esperados.
    if len(t_local) < 3 or not np.isclose(t_local[0], 0.0) or not np.isclose(t_local[-1], janela_antes_estimulo_ms + janela_depois_estimulo_ms):
        raise FalhaDeSimulacao("A janela final retornada não cobre integralmente o intervalo solicitado.")

    return {
        "tempo_ms": t_local,
        "potencial_mV": V_janela,
        "estados": estados_janela,
        "tempo_global_ms": t_global,
        "inicio_janela_global_ms": inicio_janela,
        "estimulo_local_ms": janela_antes_estimulo_ms,
    }
