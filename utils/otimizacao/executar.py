"""Orquestra a otimização.

Etapas: (1) lê o AP experimental; (2) NSGA-II com poucos pulsos (busca ampla e barata);
(3) valida as melhores soluções em regime (quase) estacionário; (4) refina a melhor
com pacing longo; (5) salva tabelas, JSON e gráficos em ``results/``.
"""

import csv
import json
import logging
import multiprocessing
import time
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from scipy.stats import qmc

from ..ap import carregar_ap
from ..modelo_celular import get_default_params
from ..simulacao import FalhaDeSimulacao
from . import configuracao as cfg
from .avaliador import AvaliadorAP, CandidatoInvalido
from .problema import ProblemaOtimizacaoAP
from .refino import refinar_em_regime_estacionario
from .selecao import chave_compromisso, mascara_validos, ordenar_por_compromisso, parametros_no_limite

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _salvar_duas_colunas(caminho, tempo, potencial, rotulos):
    caminho = Path(caminho)
    np.savetxt(
        caminho,
        np.column_stack([tempo, potencial]),
        delimiter="\t",
        header=f"{rotulos[0]}\t{rotulos[1]}",
        comments="# ",
        fmt="%.8f",
    )


def _configuracao_simulacao(num_pulsos):
    return {
        "num_pulsos": num_pulsos,
        "bcl_ms": cfg.BCL_MS,
        "estimulo_amplitude": cfg.ESTIMULO_AMPLITUDE,
        "estimulo_duracao_ms": cfg.ESTIMULO_DURACAO_MS,
        "estimulo_offset_ms": cfg.ESTIMULO_OFFSET_MS,
        "janela_antes_estimulo_ms": cfg.JANELA_ANTES_ESTIMULO_MS,
        "janela_depois_estimulo_ms": cfg.JANELA_DEPOIS_ESTIMULO_MS,
        "dt_ap_ms": cfg.DT_AP_MS,
        "dt_previo_ms": cfg.DT_SAIDA_PREVIA_MS,
        "max_step_ms": cfg.MAX_STEP_MS,
        "atol": cfg.ATOL,
        "rtol": cfg.RTOL,
    }


class _ExecutorProcessos:
    """Runner elementwise do pymoo baseado em multiprocessing.Pool (não depende da versão do pymoo)."""

    def __init__(self, pool):
        self.pool = pool

    def __call__(self, f, X):
        return list(self.pool.map(f, list(X)))

    def __getstate__(self):
        estado = self.__dict__.copy()
        estado.pop("pool", None)  # o Pool não é serializável e os workers não precisam dele
        return estado


class _Acompanhamento(Callback):
    """Registra o progresso e grava a população periodicamente (permite --retomar)."""

    def __init__(self, arquivo_checkpoint, periodo, penalidade):
        super().__init__()
        self.arquivo_checkpoint = Path(arquivo_checkpoint)
        self.periodo = max(1, int(periodo))
        self.penalidade = float(penalidade)
        self.inicio = time.time()
        self.avisou = False

    def notify(self, algorithm):
        try:
            geracao = int(algorithm.n_gen)
            if geracao != 1 and geracao % self.periodo != 0:
                return
            X = np.asarray(algorithm.pop.get("X"), dtype=float)
            F = np.asarray(algorithm.pop.get("F"), dtype=float)
            validos = mascara_validos(F, self.penalidade)
            n_avaliacoes = getattr(getattr(algorithm, "evaluator", None), "n_eval", -1)
            melhor_apd = float(F[validos, 0].min()) if validos.any() else float("nan")
            melhor_forma = float(F[validos, 1].min()) if validos.any() else float("nan")
            logger.info(
                "geração %4d | avaliações %7s | inválidos %3d/%d | melhor erro APD90 %.4f | "
                "melhor RMSE forma %.4f | %.0f s",
                geracao, n_avaliacoes, int((~validos).sum()), len(F), melhor_apd, melhor_forma,
                time.time() - self.inicio,
            )
            np.savez(self.arquivo_checkpoint, X=X, F=F, geracao=geracao)
        except Exception as erro:  # noqa: BLE001 - monitoramento nunca deve derrubar uma busca longa
            if not self.avisou:
                logger.warning("Falha no acompanhamento (ignorada daqui em diante): %s", erro)
                self.avisou = True


def _populacao_inicial(problema, x_padrao, tamanho, semente, arquivo_checkpoint, retomar):
    """Hipercubo latino com o modelo padrão semeado na primeira posição (ou população salva)."""
    if retomar and Path(arquivo_checkpoint).is_file():
        dados = np.load(arquivo_checkpoint)
        X = np.asarray(dados["X"], dtype=float)
        if X.shape == (tamanho, problema.n_var):
            logger.info("Retomando da população salva (geração %d).", int(dados["geracao"]))
            return np.clip(X, problema.xl, problema.xu)
        logger.warning("Checkpoint incompatível (formato %s); iniciando do zero.", X.shape)
    elif retomar:
        logger.warning("Checkpoint %s não encontrado; iniciando do zero.", arquivo_checkpoint)
    amostrador = qmc.LatinHypercube(d=problema.n_var, seed=semente)
    X = qmc.scale(amostrador.random(tamanho), problema.xl, problema.xu)
    X[0] = np.clip(x_padrao, problema.xl, problema.xu)
    return X


def _criar_terminacao(geracoes, populacao):
    opcoes = {
        "xtol": 1e-8,
        "cvtol": 1e-6,
        "ftol": cfg.PARADA_FTOL,
        "n_max_gen": geracoes,
        "n_max_evals": populacao * geracoes * 2,
    }
    try:
        return DefaultMultiObjectiveTermination(period=cfg.PARADA_PERIODO, **opcoes)
    except TypeError:  # versões do pymoo sem o argumento 'period'
        return DefaultMultiObjectiveTermination(**opcoes)


def _relatar_alvo(avaliador):
    detalhes = avaliador.detalhes_alvo
    por_ref = avaliador.apd90_alvo_por_referencia
    logger.info(
        "APD90 do alvo: %.3f ms (a partir da ativação) | %.3f ms (a partir do pico) | ativação em t = %.2f ms",
        por_ref["ativacao"], por_ref["pico"], detalhes["t_ativacao"],
    )
    logger.info("Referência usada na otimização: APD90 a partir de '%s'.", cfg.REFERENCIA_APD)
    passo = float(avaliador.tempo_alvo[1] - avaliador.tempo_alvo[0])
    if detalhes["t_ativacao"] - cfg.INICIO_BUSCA_PICO_ALVO_MS < 2 * passo:
        logger.warning(
            "A ativação detectada (t = %.2f ms) está colada no início da busca (INICIO_BUSCA_PICO_ALVO_MS = %.1f ms): "
            "o upstroke pode ter começado antes. Diminua esse valor (e ajuste BASELINE_ALVO_MS).",
            detalhes["t_ativacao"], cfg.INICIO_BUSCA_PICO_ALVO_MS,
        )
    if detalhes["t_ativacao"] <= cfg.BASELINE_ALVO_MS[1]:
        logger.warning(
            "O upstroke detectado (t = %.2f ms) está dentro da janela de linha de base %s ms: "
            "ajuste BASELINE_ALVO_MS e INICIO_BUSCA_PICO_ALVO_MS em configuracao.py.",
            detalhes["t_ativacao"], cfg.BASELINE_ALVO_MS,
        )


def _validar_candidatos(avaliador, X, ordem, *, top_k, num_pulsos, sobrescritas):
    """Re-simula as melhores soluções com muitos pulsos e mede a deriva do último batimento."""
    validados = []
    for posicao, i in enumerate(ordem[:top_k], start=1):
        try:
            metricas = avaliador.avaliar_regime_estacionario(X[i], num_pulsos=num_pulsos, **sobrescritas)
        except (FalhaDeSimulacao, CandidatoInvalido) as erro:
            logger.warning("  candidato %d (índice %d): falhou com %d pulsos (%s)", posicao, i, num_pulsos, erro)
            continue
        logger.info(
            "  candidato %d (índice %d): APD90 = %.2f ms (erro %.2f%%) | RMSE forma = %.4f | "
            "deriva do último batimento = %+.3f ms",
            posicao, i, metricas["apd90_sim_ms"], 100 * metricas["erro_apd90_relativo"],
            metricas["erro_forma_onda"], metricas["deriva_apd90_ms"],
        )
        validados.append((int(i), metricas))
    return validados


# ---------------------------------------------------------------------------
# Saídas
# ---------------------------------------------------------------------------
def _grafico_comparacao(pasta, avaliador, metricas_final, metricas_rapida, num_pulsos_final):
    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots()
    ax.plot(avaliador.tempo_alvo - avaliador.t_ativacao_alvo_ms, avaliador.potencial_alvo_norm,
            label="Experimental", linewidth=1.8)
    sim = metricas_final["simulacao"]
    ax.plot(sim["tempo_ms"] - metricas_final["t_ativacao_sim_ms"], metricas_final["potencial_sim_norm"],
            label=f"Mahajan otimizado ({metricas_final.get('num_pulsos', num_pulsos_final)} pulsos)", linewidth=1.8)
    if metricas_rapida is not None and metricas_rapida is not metricas_final:
        sim_r = metricas_rapida["simulacao"]
        ax.plot(sim_r["tempo_ms"] - metricas_rapida["t_ativacao_sim_ms"], metricas_rapida["potencial_sim_norm"],
                label="Mahajan, mesmos parâmetros da busca (poucos pulsos)", linewidth=1.2,
                linestyle="--", alpha=0.8)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, label="Ativação (dV/dt máx.)")
    depois = avaliador.janela_forma_depois_ms or 250.0
    ax.set_xlim(-avaliador.janela_forma_antes_ms - 5.0, depois + 20.0)
    ax.set_xlabel("Tempo desde a ativação (ms)")
    ax.set_ylabel("Potencial normalizado")
    ax.set_title("Comparação da forma de onda do potencial de ação (alinhada na ativação)")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pasta / "comparacao_ap.png", dpi=250)
    fig.savefig(pasta / "comparacao_ap.pdf")


def _grafico_pareto(pasta, F, melhor_idx):
    fig = Figure(figsize=(6, 4.5))
    ax = fig.subplots()
    valido = mascara_validos(F, cfg.PENALIDADE)
    ax.scatter(F[valido, 0], F[valido, 1], s=25, alpha=0.8, label="Soluções NSGA-II")
    ax.scatter(F[melhor_idx, 0], F[melhor_idx, 1], marker="*", s=140, label="Melhor compromisso (poucos pulsos)")
    ax.set_xlabel("Erro relativo do APD90")
    ax.set_ylabel("RMSE da forma normalizada")
    ax.set_title("Front de Pareto")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pasta / "front_pareto.png", dpi=250)


def _salvar_resultados(pasta, ctx):
    avaliador = ctx["avaliador"]
    nomes = avaliador.nomes_parametros
    padrao = ctx["valores_padrao"]
    limites = ctx["limites"]
    final = ctx["metricas_final"]
    rapida = ctx["metricas_rapida"]
    X, F = ctx["X_pareto"], ctx["F_pareto"]
    validados = {i: m for i, m in ctx["validados"]}

    parametros = {nome: float(final["parametros"][nome]) for nome in nomes}
    fatores = {nome: parametros[nome] / float(padrao[nome]) for nome in nomes}
    no_limite = parametros_no_limite(parametros, limites)

    simulacao = final["simulacao"]
    _salvar_duas_colunas(pasta / "AP_simulado_otimizado.txt", simulacao["tempo_ms"],
                         simulacao["potencial_mV"], ("tempo_ms", "potencial_mV"))
    _salvar_duas_colunas(pasta / "AP_simulado_otimizado_normalizado.txt", simulacao["tempo_ms"],
                         final["potencial_sim_norm"], ("tempo_ms", "potencial_normalizado"))

    colunas = (
        ["indice", "melhor_compromisso", "erro_apd90_relativo", "erro_forma_onda",
         "validado_regime", "erro_apd90_regime", "erro_forma_regime", "deriva_apd90_ms"]
        + list(nomes) + [f"fator_{nome}" for nome in nomes]
    )
    with (pasta / "solucoes_pareto.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()
        for i, (x, f) in enumerate(zip(X, F)):
            m = validados.get(i)
            linha = {
                "indice": i,
                "melhor_compromisso": bool(i == ctx["melhor_idx"]),
                "erro_apd90_relativo": float(f[0]),
                "erro_forma_onda": float(f[1]),
                "validado_regime": m is not None,
                "erro_apd90_regime": "" if m is None else float(m["erro_apd90_relativo"]),
                "erro_forma_regime": "" if m is None else float(m["erro_forma_onda"]),
                "deriva_apd90_ms": "" if m is None else float(m["deriva_apd90_ms"]),
            }
            linha.update({nome: float(valor) for nome, valor in zip(nomes, x)})
            linha.update({f"fator_{nome}": float(valor) / float(padrao[nome]) for nome, valor in zip(nomes, x)})
            escritor.writerow(linha)

    def _resumo(m):
        if m is None:
            return None
        return {
            "apd90_simulado_ms": float(m["apd90_sim_ms"]),
            "erro_apd90_relativo": float(m["erro_apd90_relativo"]),
            "erro_forma_onda_normalizada": float(m["erro_forma_onda"]),
            "num_pulsos": int(m.get("num_pulsos", ctx["num_pulsos"])),
        }

    resumo_json = {
        "metodo": ctx["metodo"],
        "objetivos": ["erro_apd90_relativo", "RMSE_forma_onda_normalizada_alinhada"],
        "referencia_apd": cfg.REFERENCIA_APD,
        "apd90_alvo_ms": float(avaliador.apd90_alvo_ms),
        "apd90_alvo_por_referencia_ms": avaliador.apd90_alvo_por_referencia,
        "apd90_simulado_ms": float(final["apd90_sim_ms"]),
        "erro_apd90_relativo": float(final["erro_apd90_relativo"]),
        "erro_apd90_percentual": float(100.0 * final["erro_apd90_relativo"]),
        "erro_forma_onda_normalizada": float(final["erro_forma_onda"]),
        "estimulo_local_ms": float(simulacao["estimulo_local_ms"]),
        "num_pulsos_busca": int(ctx["num_pulsos"]),
        "num_pulsos_solucao_final": int(final.get("num_pulsos", ctx["num_pulsos"])),
        "bcl_ms": cfg.BCL_MS,
        "solucao_final_validada_em_regime": bool(ctx["final_validada"]),
        "deriva_apd90_ultimo_batimento_ms": (
            float(final["deriva_apd90_ms"]) if "deriva_apd90_ms" in final else None
        ),
        "origem_solucao_final": ctx["origem"],
        "melhores_parametros": parametros,
        "melhores_parametros_fator_do_padrao": fatores,
        "parametros_no_limite": [{"parametro": nome, "limite": lado} for nome, lado in no_limite],
        "busca_rapida_mesmos_parametros_do_NSGA2": _resumo(rapida),
        "refino": ctx["refino_info"],
    }
    with (pasta / "melhores_parametros_metricas.json").open("w", encoding="utf-8") as arquivo:
        json.dump(resumo_json, arquivo, indent=2, ensure_ascii=False)

    _grafico_comparacao(pasta, avaliador, final, rapida, ctx["num_pulsos"])
    _grafico_pareto(pasta, F, ctx["melhor_idx"])
    return resumo_json, no_limite


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def executar_otimizacao(
    *,
    populacao=None,
    geracoes=None,
    processos=None,
    semente=None,
    num_pulsos=None,
    num_pulsos_validacao=None,
    top_k=None,
    refinar=None,
    max_avaliacoes_refino=None,
    retomar=False,
    arquivo_alvo=None,
    pasta_resultados=None,
):
    """Executa o pipeline completo. Argumentos ``None`` usam os valores de ``configuracao.py``."""
    populacao = cfg.POPULACAO if populacao is None else int(populacao)
    geracoes = cfg.GERACOES if geracoes is None else int(geracoes)
    processos = cfg.NUM_PROCESSOS if processos is None else int(processos)
    semente = cfg.SEMENTE if semente is None else int(semente)
    num_pulsos = cfg.NUM_PULSOS if num_pulsos is None else int(num_pulsos)
    num_pulsos_validacao = cfg.NUM_PULSOS_VALIDACAO if num_pulsos_validacao is None else int(num_pulsos_validacao)
    top_k = cfg.TOP_K_VALIDACAO if top_k is None else int(top_k)
    refinar = cfg.REFINAR_REGIME_ESTACIONARIO if refinar is None else bool(refinar)
    max_avaliacoes_refino = cfg.MAX_AVALIACOES_REFINO if max_avaliacoes_refino is None else int(max_avaliacoes_refino)
    pasta = Path(pasta_resultados) if pasta_resultados else cfg.PASTA_RESULTADOS
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo_alvo = Path(arquivo_alvo) if arquivo_alvo else cfg.ARQUIVO_AP_ALVO
    arquivo_checkpoint = pasta / "checkpoint_populacao.npz"

    logger.info("Carregando forma de onda experimental: %s", arquivo_alvo)
    tempo_alvo, potencial_alvo = carregar_ap(arquivo_alvo)

    valores_padrao = get_default_params()
    nomes = tuple(cfg.PARAMETROS_OTIMIZAVEIS)
    limites = cfg.calcular_limites_absolutos(valores_padrao)
    x_padrao = np.array([valores_padrao[nome] for nome in nomes], dtype=float)

    avaliador = AvaliadorAP(
        nomes_parametros=nomes,
        valores_iniciais=valores_padrao,
        tempo_alvo=tempo_alvo,
        potencial_alvo=potencial_alvo,
        configuracao_simulacao=_configuracao_simulacao(num_pulsos),
        baseline_alvo_ms=cfg.BASELINE_ALVO_MS,
        inicio_pico_alvo_ms=cfg.INICIO_BUSCA_PICO_ALVO_MS,
        baseline_sim_ms=cfg.BASELINE_SIM_MS,
        inicio_pico_sim_ms=cfg.INICIO_BUSCA_PICO_SIM_MS,
        referencia_apd=cfg.REFERENCIA_APD,
        metodo_ativacao=cfg.METODO_ATIVACAO,
        suavizar_pontos=cfg.SUAVIZAR_PONTOS,
        janela_forma_antes_ms=cfg.JANELA_FORMA_ANTES_MS,
        janela_forma_depois_apd=cfg.JANELA_FORMA_DEPOIS_APD,
        amplitude_minima_sim_mV=cfg.AMPLITUDE_MINIMA_SIM_MV,
    )
    _relatar_alvo(avaliador)
    logger.info("Parâmetros a otimizar: %s", ", ".join(nomes))
    logger.info("Protocolo: %d pulsos, BCL=%g ms; estímulo local em %g ms.",
                num_pulsos, cfg.BCL_MS, cfg.JANELA_ANTES_ESTIMULO_MS)
    for nome in nomes:
        logger.info("  %-6s padrão = %-10.5g limites = [%.5g, %.5g]", nome, valores_padrao[nome], *limites[nome])

    sobrescritas_validacao = {
        "max_step_ms": cfg.MAX_STEP_VALIDACAO_MS,
        "atol": cfg.ATOL_VALIDACAO,
        "rtol": cfg.RTOL_VALIDACAO,
    }

    # O modelo padrão serve de referência e também compila/carrega o cache do numba antes de
    # criarmos os processos de trabalho.
    try:
        referencia = avaliador.avaliar_candidato(x_padrao)
        logger.info(
            "Modelo padrão (%d pulsos): APD90 = %.1f ms (alvo %.1f ms, erro %.0f%%), RMSE da forma = %.3f",
            num_pulsos, referencia["apd90_sim_ms"], avaliador.apd90_alvo_ms,
            100 * referencia["erro_apd90_relativo"], referencia["erro_forma_onda"],
        )
    except (FalhaDeSimulacao, CandidatoInvalido) as erro:
        logger.warning("O modelo padrão não produziu um AP utilizável: %s", erro)

    # --- NSGA-II --------------------------------------------------------
    n_processos = max(1, min(processos, populacao))
    pool = multiprocessing.Pool(n_processos) if n_processos > 1 else None
    try:
        kwargs_problema = {"elementwise_runner": _ExecutorProcessos(pool)} if pool is not None else {}
        problema = ProblemaOtimizacaoAP(avaliador, limites, penalidade=cfg.PENALIDADE, **kwargs_problema)
        X_inicial = _populacao_inicial(problema, x_padrao, populacao, semente, arquivo_checkpoint, retomar)
        algoritmo = NSGA2(
            pop_size=populacao,
            sampling=Population.new("X", X_inicial),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )
        logger.info("Iniciando NSGA-II: população=%d, gerações<=%d, processos=%d.",
                    populacao, geracoes, n_processos)
        resultado = minimize(
            problema,
            algoritmo,
            _criar_terminacao(geracoes, populacao),
            seed=semente,
            verbose=False,
            save_history=False,
            callback=_Acompanhamento(arquivo_checkpoint, cfg.PERIODO_CHECKPOINT, cfg.PENALIDADE),
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    X_pareto = np.atleast_2d(np.asarray(resultado.X, dtype=float))
    F_pareto = np.atleast_2d(np.asarray(resultado.F, dtype=float))
    ordem = ordenar_por_compromisso(
        F_pareto, tolerancia_apd90=cfg.TOLERANCIA_APD90_RELATIVA, penalidade=cfg.PENALIDADE
    )
    if ordem.size == 0:
        raise RuntimeError(
            "Todas as soluções receberam penalidade. Verifique se o modelo produz AP com o protocolo "
            "de estímulo e se os limites dos parâmetros são adequados."
        )
    melhor_idx = int(ordem[0])
    metricas_rapida = avaliador.avaliar_candidato(X_pareto[melhor_idx])
    logger.info(
        "Melhor compromisso da busca (%d pulsos): APD90 = %.2f ms (erro %.2f%%), RMSE forma = %.4f",
        num_pulsos, metricas_rapida["apd90_sim_ms"], 100 * metricas_rapida["erro_apd90_relativo"],
        metricas_rapida["erro_forma_onda"],
    )

    # --- Validação em regime estacionário ---------------------------------
    logger.info("Validando as %d melhores soluções com %d pulsos...", min(top_k, ordem.size), num_pulsos_validacao)
    validados = _validar_candidatos(
        avaliador, X_pareto, ordem, top_k=top_k, num_pulsos=num_pulsos_validacao, sobrescritas=sobrescritas_validacao
    )

    def _chave(m):
        return chave_compromisso((m["erro_apd90_relativo"], m["erro_forma_onda"]), cfg.TOLERANCIA_APD90_RELATIVA)

    refino_info = None
    if validados:
        indice_final, metricas_final = min(validados, key=lambda par: _chave(par[1]))
        x_final = X_pareto[indice_final]
        origem = "NSGA-II + validação em regime estacionário"
        final_validada = True
    else:
        logger.warning("Nenhum candidato simulou com %d pulsos; resultado NÃO validado em regime estacionário.",
                       num_pulsos_validacao)
        indice_final, metricas_final, x_final = melhor_idx, metricas_rapida, X_pareto[melhor_idx]
        origem = "NSGA-II (busca rápida, não validada em regime estacionário)"
        final_validada = False

    # --- Refino em regime estacionário -------------------------------------
    if refinar and final_validada:
        logger.info("Refinando em regime estacionário (%d pulsos por avaliação, até %d avaliações)...",
                    num_pulsos_validacao, max_avaliacoes_refino)

        def _registrar(n, J, m):
            if n % 25 == 0:
                logger.info("  refino: %d avaliações | J = %.5f (APD90 %.2f ms)", n, J, m["apd90_sim_ms"])

        try:
            refino = refinar_em_regime_estacionario(
                avaliador, x_final, limites,
                num_pulsos=num_pulsos_validacao,
                sobrescritas_simulacao=sobrescritas_validacao,
                peso_apd90=cfg.PESO_APD90_REFINO,
                max_avaliacoes=max_avaliacoes_refino,
                penalidade=cfg.PENALIDADE,
                registrar=_registrar,
            )
            metricas_refinada = avaliador.avaliar_regime_estacionario(
                refino["x"], num_pulsos=num_pulsos_validacao, **sobrescritas_validacao
            )
            melhorou = _chave(metricas_refinada) < _chave(metricas_final)
            refino_info = {
                "avaliacoes": int(refino["n_avaliacoes"]),
                "J_final": float(refino["J"]),
                "melhorou_em_relacao_a_validacao": bool(melhorou),
                "erro_apd90_relativo_antes": float(metricas_final["erro_apd90_relativo"]),
                "erro_apd90_relativo_depois": float(metricas_refinada["erro_apd90_relativo"]),
                "erro_forma_antes": float(metricas_final["erro_forma_onda"]),
                "erro_forma_depois": float(metricas_refinada["erro_forma_onda"]),
            }
            logger.info(
                "Refino: APD90 %.2f ms (erro %.2f%%), RMSE forma %.4f | antes: erro %.2f%%, RMSE %.4f",
                metricas_refinada["apd90_sim_ms"], 100 * metricas_refinada["erro_apd90_relativo"],
                metricas_refinada["erro_forma_onda"], 100 * metricas_final["erro_apd90_relativo"],
                metricas_final["erro_forma_onda"],
            )
            if melhorou:
                metricas_final, x_final = metricas_refinada, refino["x"]
                origem = "NSGA-II + validação + refino em regime estacionário"
        except (FalhaDeSimulacao, CandidatoInvalido) as erro:
            logger.warning("Refino descartado: %s", erro)

    metodo = "NSGA-II (poucos pulsos) + validação em regime estacionário"
    if refino_info is not None:
        metodo += " + refino Nelder-Mead"
    resumo, no_limite = _salvar_resultados(
        pasta,
        {
            "avaliador": avaliador, "valores_padrao": valores_padrao, "limites": limites,
            "metricas_final": metricas_final, "metricas_rapida": metricas_rapida,
            "X_pareto": X_pareto, "F_pareto": F_pareto, "melhor_idx": melhor_idx,
            "validados": validados, "final_validada": final_validada, "origem": origem,
            "refino_info": refino_info, "num_pulsos": num_pulsos, "metodo": metodo,
        },
    )

    logger.info("")
    logger.info("Otimização concluída.")
    logger.info("APD90 alvo:       %.3f ms", resumo["apd90_alvo_ms"])
    logger.info("APD90 simulado:   %.3f ms (%d pulsos)", resumo["apd90_simulado_ms"], resumo["num_pulsos_solucao_final"])
    logger.info("Erro APD90:       %.2f%%", resumo["erro_apd90_percentual"])
    logger.info("Erro forma onda:  %.5f (RMSE normalizado e alinhado)", resumo["erro_forma_onda_normalizada"])
    if resumo["deriva_apd90_ultimo_batimento_ms"] is not None:
        logger.info("Deriva do APD90 no último batimento: %+.3f ms", resumo["deriva_apd90_ultimo_batimento_ms"])
    logger.info("Parâmetros escolhidos (valor | fator do padrão):")
    for nome in nomes:
        logger.info("  %-6s %.8g | x%.3f", nome, resumo["melhores_parametros"][nome],
                    resumo["melhores_parametros_fator_do_padrao"][nome])
    if no_limite:
        logger.warning("Parâmetros no limite da faixa (o ajuste pode estar restrito): %s",
                       ", ".join(f"{nome} ({lado})" for nome, lado in no_limite))
    logger.info("Resultados em: %s", pasta)
    return resumo
