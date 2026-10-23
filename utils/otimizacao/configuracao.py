"""Configuração central da otimização. Ajuste os valores deste arquivo.

Os valores podem ser sobrescritos na linha de comando (veja ``python main.py --help``).
"""

import os
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_DADOS = RAIZ_PROJETO / "data"
PASTA_RESULTADOS = RAIZ_PROJETO / "results"
ARQUIVO_AP_ALVO = PASTA_DADOS / "ap_mayra.txt"

# ---------------------------------------------------------------------------
# Protocolo de estimulação e janela analisada
# ---------------------------------------------------------------------------
# O oitavo estímulo ocorre em (8 - 1) * 450 = 3150 ms.
NUM_PULSOS = 8
BCL_MS = 450.0
ESTIMULO_AMPLITUDE = -15.0
ESTIMULO_DURACAO_MS = 3.0
ESTIMULO_OFFSET_MS = 0.0

# Janela de 300 ms: 50 ms antes do último estímulo e 250 ms depois.
JANELA_ANTES_ESTIMULO_MS = 50.0
JANELA_DEPOIS_ESTIMULO_MS = 250.0
DT_AP_MS = 0.25
# Nos primeiros ciclos só precisamos guardar pontos esparsos; a integração
# numérica ainda respeita max_step e os limites de cada estímulo.
DT_SAIDA_PREVIA_MS = 2.0

# Precisão do solver durante a BUSCA. Medido no modelo padrão: APD90 idêntico (±0,01 ms)
# ao de uma integração com rtol=1e-8, então estes valores são suficientes.
MAX_STEP_MS = 2.0
ATOL = 1e-6
RTOL = 1e-4

# ---------------------------------------------------------------------------
# Validação em regime (quase) estacionário, feita DEPOIS da busca
# ---------------------------------------------------------------------------
# Com 8 pulsos o modelo ainda não está em regime estacionário ([Na]i, carga do RS).
# No modelo padrão (BCL=450 ms) o APD90 vai de ~188 ms (8 pulsos) para ~192 ms (120 pulsos).
# As melhores soluções da busca são re-simuladas com este número de pulsos.
NUM_PULSOS_VALIDACAO = 10
TOP_K_VALIDACAO = 5
MAX_STEP_VALIDACAO_MS = 0.5
ATOL_VALIDACAO = 1e-8
RTOL_VALIDACAO = 1e-6

# ---------------------------------------------------------------------------
# Parâmetros otimizados e limites
# ---------------------------------------------------------------------------
PARAMETROS_OTIMIZAVEIS = ("gca", "gtof", "gtos", "gNaK", "gkr", "gks", "gkix", "gNaCa")

# Limites = fator * valor padrão do modelo (lido do próprio modelo, sem copiar números).
# [0,1x ; 1,9x] equivale a -90% / +90%. Obs.: gkix = 0,1x não repolariza (a simulação
# falha e o candidato é penalizado) e gca = 0,2x sozinho já leva o APD90 de ~188 ms para ~56 ms.
FATOR_MIN_PADRAO = 0.1
FATOR_MAX_PADRAO = 10
# Sobrescritas por parâmetro, no formato {"gNaK": (0.55, 1.15)}.
FATORES_LIMITES = {}

# Alternativa mais restrita, baseada em Gemmell et al. (2014, PLoS ONE 9:e90112):
# variação de +-30% em torno do padrão. Para usar: FATORES_LIMITES = FATORES_LITERATURA.
FATORES_LITERATURA = {nome: (0.7, 1.3) for nome in PARAMETROS_OTIMIZAVEIS}


def calcular_limites_absolutos(valores_padrao):
    """Converte os fatores acima em limites ABSOLUTOS {nome: (inferior, superior)}."""
    limites = {}
    for nome in PARAMETROS_OTIMIZAVEIS:
        fator_min, fator_max = FATORES_LIMITES.get(nome, (FATOR_MIN_PADRAO, FATOR_MAX_PADRAO))
        padrao = float(valores_padrao[nome])
        limites[nome] = (padrao * fator_min, padrao * fator_max)
    return limites


# ---------------------------------------------------------------------------
# Definição das métricas (precisa ser a MESMA para o alvo e para o simulado)
# ---------------------------------------------------------------------------
# O arquivo experimental tem tempo de 0 a 300 ms. Estimamos seu repouso no trecho inicial
# (0-25 ms); a busca do upstroke/pico começa em 25 ms. Se o upstroke do alvo começar antes
# de 25 ms ou a linha de base incluir parte dele, ajuste estes dois valores.
BASELINE_ALVO_MS = (0.0, 10.0)
INICIO_BUSCA_PICO_ALVO_MS = 10.0
# Simulado: os 50 ms que antecedem o pulso no trecho recortado.
BASELINE_SIM_MS = (0.0, JANELA_ANTES_ESTIMULO_MS)
INICIO_BUSCA_PICO_SIM_MS = JANELA_ANTES_ESTIMULO_MS

# "ativacao": APD90 do upstroke (dV/dt máx.) até 90% de repolarização (padrão da literatura);
# "pico": definição antiga do projeto (a partir do pico).
REFERENCIA_APD = "ativacao"
METODO_ATIVACAO = "dvdt_max"  # ou "meia_amplitude" (mais robusto a ruído no registro)
SUAVIZAR_PONTOS = 3

# Forma de onda comparada de -JANELA_FORMA_ANTES_MS até JANELA_FORMA_DEPOIS_APD * APD90 do alvo,
# ambos contados a partir da ativação. Evita diluir o erro com o trecho de repouso final.
# 0 começa na ativação e deixa de fora os ~3 ms do "degrau" do estímulo (pulso de 3 ms), que só
# existe no simulado e, se entrasse na janela, imporia um piso de ~0,02 ao RMSE.
JANELA_FORMA_ANTES_MS = 0.0
JANELA_FORMA_DEPOIS_APD = 2.0

# Um AP "de verdade" tem ~100-120 mV de amplitude no modelo; a resposta passiva ao estímulo
# (sem INa regenerativa) chega a ~41 mV. O limite antigo (10 mV) aceitaria essa resposta passiva.
AMPLITUDE_MINIMA_SIM_MV = 60.0
PENALIDADE = 1e3

# Escolha do melhor compromisso: entre as soluções com erro de APD90 <= tolerância,
# escolhe-se a de menor erro de forma de onda.
TOLERANCIA_APD90_RELATIVA = 0.05

# Refino local (Nelder-Mead) da melhor solução com NUM_PULSOS_VALIDACAO pulsos por avaliação.
# Corrige a diferença entre o APD90 de 8 pulsos e o de regime estacionário (medida: -6% a -10%
# para candidatos com APD90 curto). Cada avaliação simula ~150 batimentos.
REFINAR_REGIME_ESTACIONARIO = True
MAX_AVALIACOES_REFINO = 300
PESO_APD90_REFINO = 1.0

# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------
# Cada avaliação simula vários batimentos de um modelo com 26 EDOs. 128 x 200 = ~25 mil
# avaliações já é uma busca ampla para 8 variáveis e 2 objetivos; aumente se necessário.
POPULACAO = 1024
GERACOES = 200
SEMENTE = 1
NUM_PROCESSOS = os.cpu_count() or 1
# Critério de parada antecipada (pymoo DefaultMultiObjectiveTermination).
PARADA_FTOL = 0.005
PARADA_PERIODO = 30
# A cada N gerações a população é gravada em results/checkpoint_populacao.npz.
PERIODO_CHECKPOINT = 10
