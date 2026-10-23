"""Avaliação de candidatos contra o AP experimental.

Este módulo não depende do pymoo: assim a lógica do objetivo pode ser testada
(e reutilizada em validações e refinos) sem montar o problema de otimização.
"""

import numpy as np

from ..ap import calcular_apd90_detalhado, erro_forma_onda_alinhada, normalizar_ap
from ..simulacao import simular_ultimo_ap

# Somado ao erro de APD90 quando o AP simulado não repolariza dentro da janela (o APD90
# só é conhecido como limite inferior). Mantém a penalidade graduada em vez de "tudo ou nada".
PENALIDADE_APD90_NAO_REPOLARIZOU = 1.0


class CandidatoInvalido(Exception):
    """O candidato não gerou um AP utilizável. Falha esperada: vira penalidade na otimização."""


class AvaliadorAP:
    """Simula um candidato e mede (1) erro relativo de APD90 e (2) RMSE da forma de onda.

    As duas métricas usam a MESMA definição para o alvo e para o simulado: APD90 contado a
    partir da ativação (ou do pico, conforme ``referencia_apd``) e formas de onda
    normalizadas e alinhadas no instante de ativação.
    """

    def __init__(
        self,
        *,
        nomes_parametros,
        valores_iniciais,
        tempo_alvo,
        potencial_alvo,
        configuracao_simulacao,
        baseline_alvo_ms,
        inicio_pico_alvo_ms,
        baseline_sim_ms,
        inicio_pico_sim_ms,
        referencia_apd="ativacao",
        metodo_ativacao="dvdt_max",
        suavizar_pontos=3,
        janela_forma_antes_ms=10.0,
        janela_forma_depois_apd=2.0,
        amplitude_minima_sim_mV=50.0,
    ):
        self.nomes_parametros = tuple(nomes_parametros)
        self.valores_iniciais = dict(valores_iniciais)
        faltando = [nome for nome in self.nomes_parametros if nome not in self.valores_iniciais]
        if faltando:
            raise KeyError(f"Parâmetros inexistentes no modelo: {faltando}")

        self.configuracao_simulacao = dict(configuracao_simulacao)
        self.baseline_sim_ms = tuple(baseline_sim_ms)
        self.inicio_pico_sim_ms = float(inicio_pico_sim_ms)
        self.amplitude_minima_sim_mV = float(amplitude_minima_sim_mV)
        self.janela_forma_antes_ms = float(janela_forma_antes_ms)
        self._opcoes_apd = {
            "referencia": referencia_apd,
            "metodo_ativacao": metodo_ativacao,
            "suavizar_pontos": suavizar_pontos,
        }

        self.tempo_alvo = np.asarray(tempo_alvo, dtype=float)
        self.potencial_alvo = np.asarray(potencial_alvo, dtype=float)
        kw_alvo = {
            "janela_linha_base_ms": tuple(baseline_alvo_ms),
            "inicio_busca_pico_ms": float(inicio_pico_alvo_ms),
        }
        detalhes = calcular_apd90_detalhado(self.tempo_alvo, self.potencial_alvo, **kw_alvo, **self._opcoes_apd)
        if not detalhes["repolarizou"] or not detalhes["apd90"] > 0:
            raise ValueError(
                "Não foi possível medir o APD90 do alvo (o sinal não repolariza 90% na janela ou "
                "a janela de linha de base/busca do pico está mal ajustada)."
            )
        self.detalhes_alvo = detalhes
        self.apd90_alvo_ms = float(detalhes["apd90"])
        self.t_ativacao_alvo_ms = float(detalhes["t_ativacao"])
        # Para relatório: o mesmo alvo nas duas definições de APD90.
        self.apd90_alvo_por_referencia = {}
        for referencia in ("ativacao", "pico"):
            opcoes = dict(self._opcoes_apd, referencia=referencia)
            self.apd90_alvo_por_referencia[referencia] = float(
                calcular_apd90_detalhado(self.tempo_alvo, self.potencial_alvo, **kw_alvo, **opcoes)["apd90"]
            )

        self.potencial_alvo_norm, _, _ = normalizar_ap(
            self.tempo_alvo, self.potencial_alvo,
            janela_linha_base_ms=kw_alvo["janela_linha_base_ms"],
            inicio_busca_pico_ms=kw_alvo["inicio_busca_pico_ms"],
        )
        self.janela_forma_depois_ms = (
            None if janela_forma_depois_apd is None else float(janela_forma_depois_apd) * self.apd90_alvo_ms
        )

    # ------------------------------------------------------------------
    def parametros_do_candidato(self, x):
        """Dicionário completo de parâmetros: padrão do modelo + valores de ``x``."""
        x = np.asarray(x, dtype=float)
        if x.shape != (len(self.nomes_parametros),):
            raise ValueError(f"x deve ter {len(self.nomes_parametros)} valores (um por parâmetro).")
        parametros = dict(self.valores_iniciais)
        parametros.update({nome: float(valor) for nome, valor in zip(self.nomes_parametros, x)})
        return parametros

    def metricas_da_simulacao(self, simulacao):
        """Calcula os erros a partir de uma simulação já feita (levanta CandidatoInvalido se não houver AP)."""
        tempo = np.asarray(simulacao["tempo_ms"], dtype=float)
        potencial = np.asarray(simulacao["potencial_mV"], dtype=float)
        if not np.all(np.isfinite(potencial)):
            raise CandidatoInvalido("O potencial simulado contém valores não finitos.")

        detalhes = calcular_apd90_detalhado(
            tempo, potencial,
            janela_linha_base_ms=self.baseline_sim_ms,
            inicio_busca_pico_ms=self.inicio_pico_sim_ms,
            **self._opcoes_apd,
        )
        if not np.isfinite(detalhes["t_ativacao"]) or not detalhes["amplitude"] >= self.amplitude_minima_sim_mV:
            raise CandidatoInvalido(f"Sem potencial de ação (amplitude = {detalhes['amplitude']:.1f} mV).")

        repolarizou = bool(detalhes["repolarizou"])
        if repolarizou:
            apd90_sim = float(detalhes["apd90"])
            erro_apd90 = abs(apd90_sim - self.apd90_alvo_ms) / self.apd90_alvo_ms
        else:
            # APD90 desconhecido (> janela): usa o limite inferior observado e soma penalidade.
            apd90_sim = float(tempo[-1] - detalhes["t_referencia"])
            erro_apd90 = (
                abs(apd90_sim - self.apd90_alvo_ms) / self.apd90_alvo_ms + PENALIDADE_APD90_NAO_REPOLARIZOU
            )

        potencial_norm, linha_base, pico = normalizar_ap(
            tempo, potencial,
            janela_linha_base_ms=self.baseline_sim_ms,
            inicio_busca_pico_ms=self.inicio_pico_sim_ms,
        )
        erro_onda = erro_forma_onda_alinhada(
            tempo, potencial_norm, detalhes["t_ativacao"],
            self.tempo_alvo, self.potencial_alvo_norm, self.t_ativacao_alvo_ms,
            antes_ms=self.janela_forma_antes_ms,
            depois_ms=self.janela_forma_depois_ms,
        )
        return {
            "apd90_sim_ms": apd90_sim,
            "apd90_alvo_ms": self.apd90_alvo_ms,
            "apd90_repolarizou": repolarizou,
            "erro_apd90_relativo": float(erro_apd90),
            "erro_forma_onda": float(erro_onda),
            "t_ativacao_sim_ms": float(detalhes["t_ativacao"]),
            "amplitude_mV": float(detalhes["amplitude"]),
            "linha_base_mV": float(linha_base),
            "pico_mV": float(pico),
            "simulacao": simulacao,
            "potencial_sim_norm": potencial_norm,
        }

    def avaliar_candidato(self, x, **sobrescritas_simulacao):
        """Simula o candidato ``x`` e devolve as métricas.

        ``sobrescritas_simulacao`` substitui chaves da configuração de simulação
        (por exemplo ``num_pulsos=150`` para a validação em regime estacionário).
        """
        parametros = self.parametros_do_candidato(x)
        configuracao = {**self.configuracao_simulacao, **sobrescritas_simulacao}
        simulacao = simular_ultimo_ap(parametros, **configuracao)
        metricas = self.metricas_da_simulacao(simulacao)
        metricas["parametros"] = parametros
        return metricas

    def avaliar_regime_estacionario(self, x, *, num_pulsos, **sobrescritas_simulacao):
        """Avalia com ``num_pulsos`` batimentos e mede a deriva do APD90 no último batimento.

        ``deriva_apd90_ms`` = APD90(último batimento) - APD90(penúltimo). Valores próximos de zero
        indicam que o estado quase parou de variar (a [Na]i ainda pode derivar lentamente).
        """
        metricas = self.avaliar_candidato(x, num_pulsos=num_pulsos, **sobrescritas_simulacao)
        anterior = self.avaliar_candidato(x, num_pulsos=num_pulsos - 1, **sobrescritas_simulacao)
        metricas["num_pulsos"] = int(num_pulsos)
        metricas["deriva_apd90_ms"] = metricas["apd90_sim_ms"] - anterior["apd90_sim_ms"]
        return metricas
