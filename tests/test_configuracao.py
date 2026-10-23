"""Garante que os limites de busca contêm o valor padrão e seguem os fatores configurados."""

from utils.modelo_celular import get_default_params
from utils.otimizacao import configuracao as cfg


def test_limites_absolutos_contem_o_padrao_de_todos_os_parametros():
    padroes = get_default_params()
    limites = cfg.calcular_limites_absolutos(padroes)
    assert set(limites) == set(cfg.PARAMETROS_OTIMIZAVEIS)
    for nome, (inferior, superior) in limites.items():
        assert inferior < padroes[nome] < superior, f"{nome}: o padrão deve ficar estritamente dentro dos limites"


def test_limites_simetricos_em_fator_por_padrao():
    """Regressão: gtos e gNaK tinham limite superior igual ao valor padrão (faltava o fator)."""
    padroes = get_default_params()
    limites = cfg.calcular_limites_absolutos(padroes)
    for nome, (inferior, superior) in limites.items():
        fator_min, fator_max = cfg.FATORES_LIMITES.get(nome, (cfg.FATOR_MIN_PADRAO, cfg.FATOR_MAX_PADRAO))
        assert abs(inferior / padroes[nome] - fator_min) < 1e-12
        assert abs(superior / padroes[nome] - fator_max) < 1e-12


def test_janelas_de_referencia_do_simulado_acompanham_o_estimulo():
    assert cfg.BASELINE_SIM_MS == (0.0, cfg.JANELA_ANTES_ESTIMULO_MS)
    assert cfg.INICIO_BUSCA_PICO_SIM_MS == cfg.JANELA_ANTES_ESTIMULO_MS
