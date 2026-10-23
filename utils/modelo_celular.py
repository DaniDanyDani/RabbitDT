# Modelo eletrofisiológico celular: define estados, parâmetros, correntes iônicas e sua evolução temporal.
# As variáveis são armazenadas em listas/arrays indexados; os nomes e unidades aparecem em createLegends().
# Size of variable arrays:
sizeAlgebraic = 108
sizeStates = 26
sizeConstants = 80
from math import *
from numpy import *
import warnings
from difflib import get_close_matches
import numpy as np
from numba import njit  # Compila o núcleo numérico usado repetidamente pelo integrador.

# Cria os nomes descritivos das variáveis de estado, taxas, variáveis algébricas e constantes.
# Essa função serve principalmente para identificar as grandezas nos gráficos e nos resultados.
def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component Environment (ms)"
    legend_constants[0] = "R in component Environment (J_per_moleK)"
    legend_constants[1] = "T in component Environment (kelvin)"
    legend_constants[2] = "F in component Environment (coulomb_per_mmole)"
    legend_constants[3] = "K_o in component Environment (mM)"
    legend_constants[4] = "Ca_o in component Environment (mM)"
    legend_constants[5] = "Na_o in component Environment (mM)"
    legend_constants[75] = "FonRT in component Environment (per_mV)"
    legend_states[0] = "V in component cell (mV)"
    legend_algebraic[49] = "xik1 in component IK1 (nA_per_nF)"
    legend_algebraic[60] = "xito in component Ito (nA_per_nF)"
    legend_algebraic[65] = "xiNaK in component INaK (nA_per_nF)"
    legend_constants[6] = "wca in component cell (mV_per_uM)"
    legend_algebraic[102] = "xiNaCa in component INaCa (nA_per_nF)"
    legend_algebraic[93] = "xica in component ICaL (nA_per_nF)"
    legend_algebraic[107] = "Itotal in component cell (nA_per_nF)"
    legend_algebraic[106] = "xina in component INa (nA_per_nF)"
    legend_algebraic[51] = "xikr in component IKr (nA_per_nF)"
    legend_algebraic[104] = "xiks in component IKs (nA_per_nF)"
    legend_algebraic[10] = "i_Stim in component cell (nA_per_nF)"
    legend_constants[7] = "stim_offset in component cell (ms)"
    legend_constants[8] = "stim_period in component cell (ms)"
    legend_constants[9] = "stim_duration in component cell (ms)"
    legend_constants[10] = "stim_amplitude in component cell (nA_per_nF)"
    legend_algebraic[3] = "past in component cell (ms)"
    legend_algebraic[105] = "ena in component reversal_potentials (mV)"
    legend_states[1] = "xm in component INa (dimensionless)"
    legend_states[2] = "xh in component INa (dimensionless)"
    legend_states[3] = "xj in component INa (dimensionless)"
    legend_constants[11] = "gna in component INa (uS_per_nF)"
    legend_algebraic[0] = "am in component INa (per_ms)"
    legend_algebraic[7] = "bm in component INa (per_ms)"
    legend_algebraic[1] = "ah in component INa (per_ms)"
    legend_algebraic[8] = "bh in component INa (per_ms)"
    legend_algebraic[2] = "aj in component INa (per_ms)"
    legend_algebraic[9] = "bj in component INa (per_ms)"
    legend_states[4] = "Ca_dyad in component Ca (uM)"
    legend_algebraic[88] = "csm in component Ca (mM)"
    legend_states[5] = "c1 in component ICaL (dimensionless)"
    legend_states[6] = "c2 in component ICaL (dimensionless)"
    legend_states[7] = "xi1ca in component ICaL (dimensionless)"
    legend_states[8] = "xi1ba in component ICaL (dimensionless)"
    legend_states[9] = "xi2ca in component ICaL (dimensionless)"
    legend_states[10] = "xi2ba in component ICaL (dimensionless)"
    legend_constants[12] = "gca in component ICaL (mmole_per_coulomb_cm)"
    legend_constants[13] = "pca in component ICaL (cm_per_s)"
    legend_algebraic[14] = "za in component ICaL (dimensionless)"
    legend_algebraic[18] = "poinf in component ICaL (dimensionless)"
    legend_algebraic[26] = "fca in component ICaL (dimensionless)"
    legend_constants[14] = "vth in component ICaL (mV)"
    legend_constants[15] = "s6 in component ICaL (mV)"
    legend_constants[16] = "vx in component ICaL (mV)"
    legend_constants[17] = "sx in component ICaL (mV)"
    legend_constants[18] = "vy in component ICaL (mV)"
    legend_constants[19] = "sy in component ICaL (mV)"
    legend_constants[20] = "vyr in component ICaL (mV)"
    legend_constants[21] = "syr in component ICaL (mV)"
    legend_constants[22] = "cat in component ICaL (uM)"
    legend_constants[23] = "cpt in component ICaL (uM)"
    legend_algebraic[23] = "alpha in component ICaL (per_ms)"
    legend_algebraic[25] = "beta in component ICaL (per_ms)"
    legend_algebraic[28] = "k1 in component ICaL (per_ms)"
    legend_constants[24] = "k2 in component ICaL (per_ms)"
    legend_constants[25] = "k1t in component ICaL (per_ms)"
    legend_constants[26] = "k2t in component ICaL (per_ms)"
    legend_algebraic[31] = "k3 in component ICaL (per_ms)"
    legend_algebraic[32] = "k3t in component ICaL (per_ms)"
    legend_algebraic[39] = "k6 in component ICaL (per_ms)"
    legend_algebraic[40] = "k5 in component ICaL (per_ms)"
    legend_algebraic[41] = "k6t in component ICaL (per_ms)"
    legend_algebraic[42] = "k5t in component ICaL (per_ms)"
    legend_algebraic[43] = "k4 in component ICaL (per_ms)"
    legend_algebraic[44] = "k4t in component ICaL (per_ms)"
    legend_constants[27] = "r1 in component ICaL (per_ms)"
    legend_constants[28] = "r2 in component ICaL (per_ms)"
    legend_algebraic[27] = "s1 in component ICaL (per_ms)"
    legend_constants[29] = "s1t in component ICaL (per_ms)"
    legend_algebraic[29] = "s2 in component ICaL (per_ms)"
    legend_constants[76] = "s2t in component ICaL (per_ms)"
    legend_algebraic[34] = "recov in component ICaL (ms)"
    legend_constants[30] = "tca in component ICaL (ms)"
    legend_algebraic[35] = "tau_ca in component ICaL (ms)"
    legend_algebraic[36] = "tauca in component ICaL (ms)"
    legend_algebraic[37] = "tauba in component ICaL (ms)"
    legend_constants[31] = "taupo in component ICaL (ms)"
    legend_constants[32] = "tau3 in component ICaL (ms)"
    legend_algebraic[33] = "Pr in component ICaL (dimensionless)"
    legend_algebraic[38] = "Ps in component ICaL (dimensionless)"
    legend_algebraic[30] = "poi in component ICaL (dimensionless)"
    legend_algebraic[45] = "po in component ICaL (dimensionless)"
    legend_algebraic[89] = "rxa in component ICaL (mA_per_cm2)"
    legend_algebraic[90] = "jca in component ICaL (uM_per_ms)"
    legend_constants[79] = "ek in component reversal_potentials (mV)"
    legend_constants[33] = "gkix in component IK1 (uS_per_nF)"
    legend_algebraic[46] = "aki in component IK1 (per_ms)"
    legend_algebraic[47] = "bki in component IK1 (per_ms)"
    legend_algebraic[48] = "xkin in component IK1 (dimensionless)"
    legend_states[11] = "xr in component IKr (dimensionless)"
    legend_constants[34] = "gkr in component IKr (uS_per_nF)"
    legend_algebraic[4] = "xkrv1 in component IKr (per_ms)"
    legend_algebraic[11] = "xkrv2 in component IKr (per_ms)"
    legend_algebraic[15] = "taukr in component IKr (ms)"
    legend_algebraic[19] = "xkrinf in component IKr (dimensionless)"
    legend_algebraic[50] = "rg in component IKr (dimensionless)"
    legend_states[12] = "Ca_i in component Ca (uM)"
    legend_constants[35] = "gks in component IKs (uS_per_nF)"
    legend_states[13] = "xs1 in component IKs (dimensionless)"
    legend_states[14] = "xs2 in component IKs (dimensionless)"
    legend_algebraic[103] = "eks in component reversal_potentials (mV)"
    legend_algebraic[5] = "xs1ss in component IKs (dimensionless)"
    legend_algebraic[12] = "xs2ss in component IKs (dimensionless)"
    legend_algebraic[16] = "tauxs1 in component IKs (ms)"
    legend_algebraic[20] = "tauxs2 in component IKs (ms)"
    legend_algebraic[52] = "gksx in component IKs (dimensionless)"
    legend_algebraic[56] = "xitos in component Ito (nA_per_nF)"
    legend_algebraic[58] = "xitof in component Ito (nA_per_nF)"
    legend_states[15] = "xtos in component Ito (dimensionless)"
    legend_states[16] = "ytos in component Ito (dimensionless)"
    legend_states[17] = "xtof in component Ito (dimensionless)"
    legend_states[18] = "ytof in component Ito (dimensionless)"
    legend_constants[36] = "gtos in component Ito (uS_per_nF)"
    legend_constants[37] = "gtof in component Ito (uS_per_nF)"
    legend_algebraic[6] = "rt1 in component Ito (dimensionless)"
    legend_algebraic[53] = "rt2 in component Ito (dimensionless)"
    legend_algebraic[55] = "rt3 in component Ito (dimensionless)"
    legend_algebraic[13] = "rt4 in component Ito (dimensionless)"
    legend_algebraic[57] = "rt5 in component Ito (dimensionless)"
    legend_algebraic[17] = "xtos_inf in component Ito (dimensionless)"
    legend_algebraic[59] = "ytos_inf in component Ito (dimensionless)"
    legend_algebraic[21] = "xtof_inf in component Ito (dimensionless)"
    legend_algebraic[61] = "ytof_inf in component Ito (dimensionless)"
    legend_algebraic[54] = "rs_inf in component Ito (dimensionless)"
    legend_algebraic[22] = "txs in component Ito (ms)"
    legend_algebraic[62] = "tys in component Ito (ms)"
    legend_algebraic[24] = "txf in component Ito (ms)"
    legend_algebraic[64] = "tyf in component Ito (ms)"
    legend_states[19] = "Na_i in component Na (mM)"
    legend_constants[38] = "gNaK in component INaK (nA_per_nF)"
    legend_constants[39] = "xkmko in component INaK (mM)"
    legend_constants[40] = "xkmnai in component INaK (mM)"
    legend_constants[77] = "sigma in component INaK (dimensionless)"
    legend_algebraic[63] = "fNaK in component INaK (dimensionless)"
    legend_states[20] = "Ca_submem in component Ca (uM)"
    legend_constants[41] = "gNaCa in component INaCa (uM_per_ms)"
    legend_algebraic[67] = "aloss in component INaCa (dimensionless)"
    legend_algebraic[97] = "yz1 in component INaCa (mM4)"
    legend_algebraic[98] = "yz2 in component INaCa (mM4)"
    legend_algebraic[68] = "yz3 in component INaCa (mM4)"
    legend_algebraic[99] = "yz4 in component INaCa (mM4)"
    legend_algebraic[95] = "zw3 in component INaCa (mM4)"
    legend_algebraic[66] = "zw4 in component INaCa (dimensionless)"
    legend_algebraic[100] = "zw8 in component INaCa (mM4)"
    legend_algebraic[101] = "jNaCa in component INaCa (uM_per_ms)"
    legend_constants[42] = "xkdna in component INaCa (uM)"
    legend_constants[43] = "xmcao in component INaCa (mM)"
    legend_constants[44] = "xmnao in component INaCa (mM)"
    legend_constants[45] = "xmnai in component INaCa (mM)"
    legend_constants[46] = "xmcai in component INaCa (mM)"
    legend_states[21] = "Ca_NSR in component Ca (uM)"
    legend_algebraic[87] = "dCa_JSR in component Ca (uM_per_ms)"
    legend_constants[47] = "cstar in component Irel (uM)"
    legend_states[22] = "Ca_JSR in component Irel (uM)"
    legend_constants[48] = "gryr in component Irel (per_ms)"
    legend_constants[49] = "gbarsr in component Irel (dimensionless)"
    legend_constants[50] = "gdyad in component Irel (mmole_per_coulomb_cm)"
    legend_constants[51] = "ax in component Irel (per_mV)"
    legend_constants[52] = "ay in component Irel (per_mV)"
    legend_constants[53] = "av in component Irel (per_ms)"
    legend_constants[78] = "bv in component Irel (uM_per_ms)"
    legend_algebraic[69] = "Qr0 in component Irel (uM_per_ms)"
    legend_algebraic[70] = "Qr in component Irel (uM_per_ms)"
    legend_algebraic[71] = "sparkV in component Irel (dimensionless)"
    legend_algebraic[91] = "spark_rate in component Irel (per_ms)"
    legend_constants[54] = "taua in component Irel (ms)"
    legend_constants[55] = "taur in component Irel (ms)"
    legend_algebraic[92] = "xirp in component Irel (uM_per_ms)"
    legend_algebraic[94] = "xicap in component Irel (uM_per_ms)"
    legend_algebraic[96] = "xiryr in component Irel (uM_per_ms)"
    legend_states[23] = "xir in component Irel (uM_per_ms)"
    legend_algebraic[72] = "jup in component Ileak_Iup_Ixfer (uM_per_ms)"
    legend_algebraic[73] = "jleak in component Ileak_Iup_Ixfer (uM_per_ms)"
    legend_constants[56] = "cup in component Ileak_Iup_Ixfer (uM)"
    legend_constants[57] = "kj in component Ileak_Iup_Ixfer (uM)"
    legend_constants[58] = "vup in component Ileak_Iup_Ixfer (uM_per_ms)"
    legend_constants[59] = "gleak in component Ileak_Iup_Ixfer (per_ms)"
    legend_constants[60] = "bcal in component Ca (uM)"
    legend_constants[61] = "xkcal in component Ca (uM)"
    legend_constants[62] = "srmax in component Ca (uM)"
    legend_constants[63] = "srkd in component Ca (uM)"
    legend_constants[64] = "bmem in component Ca (uM)"
    legend_constants[65] = "kmem in component Ca (uM)"
    legend_constants[66] = "bsar in component Ca (uM)"
    legend_constants[67] = "ksar in component Ca (uM)"
    legend_algebraic[74] = "bpxs in component Ca (dimensionless)"
    legend_algebraic[75] = "spxs in component Ca (dimensionless)"
    legend_algebraic[76] = "mempxs in component Ca (dimensionless)"
    legend_algebraic[77] = "sarpxs in component Ca (dimensionless)"
    legend_algebraic[78] = "dcsib in component Ca (dimensionless)"
    legend_algebraic[79] = "bpxi in component Ca (dimensionless)"
    legend_algebraic[80] = "spxi in component Ca (dimensionless)"
    legend_algebraic[81] = "mempxi in component Ca (dimensionless)"
    legend_algebraic[82] = "sarpxi in component Ca (dimensionless)"
    legend_algebraic[83] = "dciib in component Ca (dimensionless)"
    legend_constants[68] = "xkon in component Ca (per_uM_per_ms)"
    legend_constants[69] = "xkoff in component Ca (per_ms)"
    legend_constants[70] = "btrop in component Ca (uM)"
    legend_algebraic[86] = "xbi in component Ca (uM_per_ms)"
    legend_algebraic[85] = "xbs in component Ca (uM_per_ms)"
    legend_states[24] = "tropi in component Ca (uM)"
    legend_states[25] = "trops in component Ca (uM)"
    legend_constants[71] = "taud in component Ca (ms)"
    legend_constants[72] = "taups in component Ca (ms)"
    legend_algebraic[84] = "jd in component Ca (uM_per_ms)"
    legend_constants[73] = "K_i in component reversal_potentials (mM)"
    legend_constants[74] = "prNaK in component reversal_potentials (dimensionless)"
    legend_rates[0] = "d/dt V in component cell (mV)"
    legend_rates[2] = "d/dt xh in component INa (dimensionless)"
    legend_rates[3] = "d/dt xj in component INa (dimensionless)"
    legend_rates[1] = "d/dt xm in component INa (dimensionless)"
    legend_rates[5] = "d/dt c1 in component ICaL (dimensionless)"
    legend_rates[6] = "d/dt c2 in component ICaL (dimensionless)"
    legend_rates[7] = "d/dt xi1ca in component ICaL (dimensionless)"
    legend_rates[8] = "d/dt xi1ba in component ICaL (dimensionless)"
    legend_rates[9] = "d/dt xi2ca in component ICaL (dimensionless)"
    legend_rates[10] = "d/dt xi2ba in component ICaL (dimensionless)"
    legend_rates[11] = "d/dt xr in component IKr (dimensionless)"
    legend_rates[13] = "d/dt xs1 in component IKs (dimensionless)"
    legend_rates[14] = "d/dt xs2 in component IKs (dimensionless)"
    legend_rates[15] = "d/dt xtos in component Ito (dimensionless)"
    legend_rates[16] = "d/dt ytos in component Ito (dimensionless)"
    legend_rates[17] = "d/dt xtof in component Ito (dimensionless)"
    legend_rates[18] = "d/dt ytof in component Ito (dimensionless)"
    legend_rates[22] = "d/dt Ca_JSR in component Irel (uM)"
    legend_rates[23] = "d/dt xir in component Irel (uM_per_ms)"
    legend_rates[19] = "d/dt Na_i in component Na (mM)"
    legend_rates[4] = "d/dt Ca_dyad in component Ca (uM)"
    legend_rates[20] = "d/dt Ca_submem in component Ca (uM)"
    legend_rates[12] = "d/dt Ca_i in component Ca (uM)"
    legend_rates[21] = "d/dt Ca_NSR in component Ca (uM)"
    legend_rates[24] = "d/dt tropi in component Ca (uM)"
    legend_rates[25] = "d/dt trops in component Ca (uM)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

# Mapas nome -> índice, construídos a partir das legendas (ex.: "gna" -> 11, "V" -> 0).
# Assim os parâmetros podem ser passados por nome, sem depender dos índices internos.
def _build_name_index():
    legend_states, _, _, legend_constants = createLegends()
    state_index = {leg.split(" in component")[0]: i for i, leg in enumerate(legend_states)}
    const_index = {leg.split(" in component")[0]: i for i, leg in enumerate(legend_constants)}
    return state_index, const_index

_STATE_INDEX, _CONSTANT_INDEX = _build_name_index()

# Constantes derivadas: calculadas a partir de outras em initConsts().
# São recalculadas automaticamente quando os parâmetros dos quais dependem são alterados.
_DERIVED_NAMES = ("FonRT", "s2t", "sigma", "bv", "ek")

# Substitui valores padrão por valores fornecidos pelo usuário (por nome); nomes inválidos geram erro.
def _apply_overrides(target, overrides, index, kind):
    unknown = [k for k in overrides if k not in index]
    if unknown:
        msgs = []
        for k in unknown:
            hint = ""
            if kind == "parameter" and k in _STATE_INDEX:
                hint = " (it is a state variable: use initial_states)"
            elif kind == "state" and k in _CONSTANT_INDEX:
                hint = " (it is a model parameter: use cell_params)"
            else:
                close = get_close_matches(str(k), list(index), n=3)
                if close:
                    hint = " (did you mean: " + ", ".join(close) + "?)"
            msgs.append("'%s'%s" % (k, hint))
        raise KeyError("Unknown %s name(s): %s" % (kind, "; ".join(msgs)))
    for name, value in overrides.items():
        target[index[name]] = float(value)

# Define os valores iniciais dos estados celulares e os parâmetros constantes do modelo.
# params: dict opcional {nome_do_parametro: valor}; init_states: dict opcional {nome_do_estado: valor}.
# O que não for informado mantém o valor padrão definido abaixo.
def initConsts(params=None, init_states=None):
    params = {} if params is None else dict(params)
    init_states = {} if init_states is None else dict(init_states)
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 8.314472
    constants[1] = 308
    constants[2] = 96.4853415
    constants[3] = 5.4
    constants[4] = 1.8
    constants[5] = 136
    states[0] = -87.169816169406
    constants[6] = 8
    constants[7] = 0
    constants[8] = 400
    constants[9] = 3
    constants[10] = -15
    states[1] = 0.001075453357
    states[2] = 0.990691306716
    states[3] = 0.993888937283
    constants[11] = 12
    states[4] = 1.716573130685
    states[5] = 0.000018211252
    states[6] = 0.979322592773
    states[7] = 0.001208153482
    states[8] = 0.000033616596
    states[9] = 0.004173008466
    states[10] = 0.015242594688
    constants[12] = 182
    constants[13] = 0.00054
    constants[14] = 0
    constants[15] = 8
    constants[16] = -40
    constants[17] = 3
    constants[18] = -40
    constants[19] = 4
    constants[20] = -40
    constants[21] = 11.32
    constants[22] = 3
    constants[23] = 6.09365
    constants[24] = 1.03615e-4
    constants[25] = 0.00413
    constants[26] = 0.00224
    constants[27] = 0.3
    constants[28] = 3
    constants[29] = 0.00195
    constants[30] = 78.0329
    constants[31] = 1
    constants[32] = 3
    constants[33] = 0.3
    states[11] = 0.007074239331
    constants[34] = 0.0125
    states[12] = 0.256752008084
    constants[35] = 0.1386
    states[13] = 0.048267587131
    states[14] = 0.105468807033
    states[15] = 0.00364776906
    states[16] = 0.174403618112
    states[17] = 0.003643592594
    states[18] = 0.993331326442
    constants[36] = 0.04
    constants[37] = 0.11
    states[19] = 11.441712311614
    constants[38] = 1.5
    constants[39] = 1.5
    constants[40] = 12
    states[20] = 0.226941113355
    constants[41] = 0.84
    constants[42] = 0.3
    constants[43] = 1.3
    constants[44] = 87.5
    constants[45] = 12.3
    constants[46] = 0.0036
    states[21] = 104.450004990523
    constants[47] = 90
    states[22] = 97.505463697266
    constants[48] = 2.58079
    constants[49] = 26841.8
    constants[50] = 9000
    constants[51] = 0.3576
    constants[52] = 0.05
    constants[53] = 11.3
    constants[54] = 100
    constants[55] = 30
    states[23] = 0.006679257264
    constants[56] = 0.5
    constants[57] = 50
    constants[58] = 0.4
    constants[59] = 0.00002069
    constants[60] = 24
    constants[61] = 7
    constants[62] = 47
    constants[63] = 0.6
    constants[64] = 15
    constants[65] = 0.3
    constants[66] = 42
    constants[67] = 13
    constants[68] = 0.0327
    constants[69] = 0.0196
    constants[70] = 70
    states[24] = 22.171689894953
    states[25] = 19.864701949854
    constants[71] = 4
    constants[72] = 0.5
    constants[73] = 140
    constants[74] = 0.01833
    # Aplica os valores fornecidos pelo usuário sobre os padrões acima.
    _apply_overrides(constants, params, _CONSTANT_INDEX, "parameter")
    _apply_overrides(states, init_states, _STATE_INDEX, "state")
    # Constantes derivadas: recalculadas a partir dos valores atuais,
    # a menos que o usuário tenha informado o valor delas explicitamente.
    if "FonRT" not in params:
        constants[75] = constants[2]/(constants[0]*constants[1])
    if "s2t" not in params:
        constants[76] = (((constants[29]*constants[27])/constants[28])*constants[26])/constants[25]
    if "sigma" not in params:
        constants[77] = (exp(constants[5]/67.3000)-1.00000)/7.00000
    if "bv" not in params:
        constants[78] = (1.00000-constants[53])*constants[47]-50.0000
    if "ek" not in params:
        constants[79] = (1.00000/constants[75])*log(constants[3]/constants[73])
    return (states, constants)

# Calcula as derivadas temporais dos estados (equações diferenciais) e as variáveis algébricas.
# A função é compilada com Numba porque o integrador a chama muitas vezes por simulação.
# voi é o instante atual; states e constants são arrays NumPy float64.
@njit(cache=True)
def _piecewise2(condition, value_true, value_false):
    """Versão escalar de uma função por partes, compatível com Numba."""
    if condition:
        return value_true
    return value_false


@njit(cache=True)
def _piecewise3(condition1, value1, condition2, value2, default_value):
    """Versão escalar de uma função por partes com dois ramos condicionais."""
    if condition1:
        return value1
    if condition2:
        return value2
    return default_value


@njit(cache=True)
def computeRates(voi, states, constants):
    rates = np.zeros(sizeStates, dtype=np.float64)
    algebraic = np.zeros(sizeAlgebraic, dtype=np.float64)
    # Evolução do cálcio no retículo sarcoplasmático juncional (Ca_JSR).
    rates[22] = (states[21]-states[22])/constants[54]
    # As expressões condicionais escalares usam _piecewise2/_piecewise3, em vez de
    # custom_piecewise/numpy.select, para manter compatibilidade com Numba nopython.
    # Cinética das variáveis de ativação/inativação dos canais rápidos de sódio (INa).
    algebraic[1] = _piecewise2(less(states[0] , -40.0000), 0.135000*exp((80.0000+states[0])/-6.80000), 0.00000)
    algebraic[8] = _piecewise2(less(states[0] , -40.0000), 3.56000*exp(0.0790000*states[0])+310000.*exp(0.350000*states[0]), 1.00000/(0.130000*(1.00000+exp((states[0]+10.6600)/-11.1000))))
    rates[2] = algebraic[1]*(1.00000-states[2])-algebraic[8]*states[2]
    algebraic[2] = _piecewise2(less(states[0] , -40.0000), ((-127140.*exp(0.244400*states[0])-3.47400e-05*exp(-0.0439100*states[0]))*1.00000*(states[0]+37.7800))/(1.00000+exp(0.311000*(states[0]+79.2300))), 0.00000)
    algebraic[9] = _piecewise2(less(states[0] , -40.0000), (0.121200*exp(-0.0105200*states[0]))/(1.00000+exp(-0.137800*(states[0]+40.1400))), (0.300000*exp(-2.53500e-07*states[0]))/(1.00000+exp(-0.100000*(states[0]+32.0000))))
    rates[3] = algebraic[2]*(1.00000-states[3])-algebraic[9]*states[3]
    algebraic[0] = _piecewise2(greater(fabs(states[0]+47.1300) , 0.00100000), (0.320000*1.00000*(states[0]+47.1300))/(1.00000-exp(-0.100000*(states[0]+47.1300))), 3.20000)
    algebraic[7] = 0.0800000*exp(-states[0]/11.0000)
    rates[1] = algebraic[0]*(1.00000-states[1])-algebraic[7]*states[1]
    # Estado estacionário e constante de tempo da ativação lenta de IKs.
    algebraic[5] = 1.00000/(1.00000+exp(-(states[0]-1.50000)/16.7000))
    algebraic[16] = _piecewise2(less(fabs(states[0]+30.0000) , 0.00100000/0.0687000), 1.00000/(7.19000e-05/0.148000+0.000131000/0.0687000), 1.00000/((7.19000e-05*(states[0]+30.0000))/(1.00000-exp(-0.148000*(states[0]+30.0000)))+(0.000131000*(states[0]+30.0000))/(exp(0.0687000*(states[0]+30.0000))-1.00000)))
    rates[13] = (algebraic[5]-states[13])/algebraic[16]
    # Cinética de ativação do canal de potássio rápido retificador tardio (IKr).
    algebraic[4] = _piecewise2(greater(fabs(states[0]+7.00000) , 0.00100000), (0.00138000*1.00000*(states[0]+7.00000))/(1.00000-exp(-0.123000*(states[0]+7.00000))), 0.00138000/0.123000)
    algebraic[11] = _piecewise2(greater(fabs(states[0]+10.0000) , 0.00100000), (0.000610000*1.00000*(states[0]+10.0000))/(exp(0.145000*(states[0]+10.0000))-1.00000), 0.000610000/0.145000)
    algebraic[15] = 1.00000/(algebraic[4]+algebraic[11])
    algebraic[19] = 1.00000/(1.00000+exp(-(states[0]+50.0000)/7.50000))
    rates[11] = (algebraic[19]-states[11])/algebraic[15]
    # Segunda variável de estado de IKs, com constante de tempo relacionada à primeira.
    algebraic[12] = algebraic[5]
    algebraic[20] = 4.00000*algebraic[16]
    rates[14] = (algebraic[12]-states[14])/algebraic[20]
    # Cinética das correntes transitórias de saída de potássio (Ito): componentes lenta e rápida.
    algebraic[6] = -(states[0]+3.00000)/15.0000
    algebraic[17] = 1.00000/(1.00000+exp(algebraic[6]))
    algebraic[22] = 9.00000/(1.00000+exp(-algebraic[6]))+0.500000
    rates[15] = (algebraic[17]-states[15])/algebraic[22]
    algebraic[21] = algebraic[17]
    algebraic[13] = ((-states[0]/30.0000)*states[0])/30.0000
    algebraic[24] = 3.50000*exp(algebraic[13])+1.50000
    rates[17] = (algebraic[21]-states[17])/algebraic[24]
    # Probabilidades e taxas do modelo de estados do canal de cálcio tipo L (ICaL).
    algebraic[18] = 1.00000/(1.00000+exp(-(states[0]-constants[14])/constants[15]))
    algebraic[23] = algebraic[18]/constants[31]
    algebraic[25] = (1.00000-algebraic[18])/constants[31]
    algebraic[26] = 1.00000/(1.00000+power(constants[22]/states[4], 3.00000))
    algebraic[34] = 10.0000+4954.00*exp(states[0]/15.6000)
    algebraic[35] = constants[30]/(1.00000+power(states[4]/constants[23], 4.00000))+0.100000
    algebraic[33] = 1.00000-1.00000/(1.00000+exp(-(states[0]-constants[18])/constants[19]))
    algebraic[36] = (algebraic[34]-algebraic[35])*algebraic[33]+algebraic[35]
    algebraic[38] = 1.00000/(1.00000+exp(-(states[0]-constants[20])/constants[21]))
    algebraic[39] = (algebraic[26]*algebraic[38])/algebraic[36]
    algebraic[40] = (1.00000-algebraic[38])/algebraic[36]
    algebraic[37] = (algebraic[34]-450.000)*algebraic[33]+450.000
    algebraic[41] = algebraic[38]/algebraic[37]
    algebraic[42] = (1.00000-algebraic[38])/algebraic[37]
    # Equações diferenciais dos estados do canal ICaL.
    rates[6] = (algebraic[25]*states[5]+algebraic[40]*states[9]+algebraic[42]*states[10])-(algebraic[39]+algebraic[41]+algebraic[23])*states[6]
    algebraic[30] = 1.00000/(1.00000+exp(-(states[0]-constants[16])/constants[17]))
    algebraic[31] = (1.00000-algebraic[30])/constants[32]
    algebraic[28] = 0.0241680*algebraic[26]
    algebraic[43] = (((((algebraic[31]*algebraic[23])/algebraic[25])*algebraic[28])/constants[24])*algebraic[40])/algebraic[39]
    rates[9] = (algebraic[31]*states[7]+algebraic[39]*states[6])-(algebraic[40]+algebraic[43])*states[9]
    algebraic[32] = algebraic[31]
    algebraic[44] = (((((algebraic[32]*algebraic[23])/algebraic[25])*constants[25])/constants[26])*algebraic[42])/algebraic[41]
    rates[10] = (algebraic[32]*states[8]+algebraic[41]*states[6])-(algebraic[42]+algebraic[44])*states[10]
    algebraic[45] = (((((1.00000-states[7])-states[9])-states[8])-states[10])-states[5])-states[6]
    rates[5] = (algebraic[23]*states[6]+constants[24]*states[7]+constants[26]*states[8]+constants[28]*algebraic[45])-(algebraic[25]+constants[27]+constants[25]+algebraic[28])*states[5]
    algebraic[27] = 0.0182688*algebraic[26]
    algebraic[29] = (((algebraic[27]*constants[27])/constants[28])*constants[24])/algebraic[28]
    rates[7] = (algebraic[28]*states[5]+algebraic[43]*states[9]+algebraic[27]*algebraic[45])-(algebraic[31]+constants[24]+algebraic[29])*states[7]
    rates[8] = (constants[25]*states[5]+algebraic[44]*states[10]+constants[29]*algebraic[45])-(algebraic[32]+constants[26]+constants[76])*states[8]
    # Cinética das variáveis de ativação/inativação de Ito.
    algebraic[53] = (states[0]+33.5000)/10.0000
    algebraic[59] = 1.00000/(1.00000+exp(algebraic[53]))
    algebraic[55] = (states[0]+60.0000)/10.0000
    algebraic[62] = 3000.00/(1.00000+exp(algebraic[55]))+30.0000
    rates[16] = (algebraic[59]-states[16])/algebraic[62]
    algebraic[61] = algebraic[59]
    algebraic[57] = (states[0]+33.5000)/10.0000
    algebraic[64] = 20.0000/(1.00000+exp(algebraic[57]))+20.0000
    rates[18] = (algebraic[61]-states[18])/algebraic[64]
    # Transporte de cálcio: captação pela SERCA e vazamento do retículo.
    algebraic[72] = (constants[58]*states[12]*states[12])/(states[12]*states[12]+constants[56]*constants[56])
    algebraic[73] = ((constants[59]*states[21]*states[21])/(states[21]*states[21]+constants[57]*constants[57]))*(states[21]*16.6670-states[12])
    # Frações de cálcio livre considerando o tamponamento por diferentes ligantes.
    algebraic[79] = (constants[60]*constants[61])/((constants[61]+states[12])*(constants[61]+states[12]))
    algebraic[80] = (constants[62]*constants[63])/((constants[63]+states[12])*(constants[63]+states[12]))
    algebraic[81] = (constants[64]*constants[65])/((constants[65]+states[12])*(constants[65]+states[12]))
    algebraic[82] = (constants[66]*constants[67])/((constants[67]+states[12])*(constants[67]+states[12]))
    algebraic[83] = 1.00000/(1.00000+algebraic[79]+algebraic[80]+algebraic[81]+algebraic[82])
    # Ligação do cálcio à troponina e balanço de cálcio no citosol.
    algebraic[86] = constants[68]*states[12]*(constants[70]-states[24])-constants[69]*states[24]
    algebraic[84] = (states[20]-states[12])/constants[71]
    rates[12] = algebraic[83]*(((algebraic[84]-algebraic[72])+algebraic[73])-algebraic[86])
    rates[24] = algebraic[86]
    algebraic[85] = constants[68]*states[20]*(constants[70]-states[25])-constants[69]*states[25]
    rates[25] = algebraic[85]
    # Balanço de cálcio no retículo sarcoplasmático não juncional (Ca_NSR).
    algebraic[87] = (-states[23]+algebraic[72])-algebraic[73]
    rates[21] = algebraic[87]
    # Liberação de cálcio do retículo (Irel) e dinâmica de abertura/liberação dos canais.
    algebraic[69] = _piecewise3(greater(states[22] , 50.0000) & less(states[22] , constants[47]), (states[22]-50.0000)/1.00000, greater_equal(states[22] , constants[47]), constants[53]*states[22]+constants[78], 0.00000)
    algebraic[70] = (states[21]*algebraic[69])/constants[47]
    # Corrente de cálcio tipo L e fluxo de cálcio associado.
    algebraic[88] = states[20]/1000.00
    algebraic[14] = states[0]*2.00000*constants[75]
    algebraic[89] = _piecewise2(less(fabs(algebraic[14]) , 0.00100000), (4.00000*constants[13]*constants[2]*constants[75]*(algebraic[88]*exp(algebraic[14])-0.341000*constants[4]))/(2.00000*constants[75]), (4.00000*constants[13]*states[0]*constants[2]*constants[75]*(algebraic[88]*exp(algebraic[14])-0.341000*constants[4]))/(exp(algebraic[14])-1.00000))
    algebraic[71] = exp(-constants[52]*(states[0]+30.0000))/(1.00000+exp(-constants[52]*(states[0]+30.0000)))
    algebraic[91] = (constants[48]/1.00000)*algebraic[45]*fabs(algebraic[89])*algebraic[71]
    rates[23] = algebraic[91]*algebraic[70]-(states[23]*(1.00000-(constants[55]*algebraic[87])/states[21]))/constants[55]
    algebraic[92] = (((algebraic[45]*algebraic[70]*fabs(algebraic[89])*constants[49])/1.00000)*exp(-constants[51]*(states[0]+30.0000)))/(1.00000+exp(-constants[51]*(states[0]+30.0000)))
    algebraic[94] = algebraic[45]*constants[50]*fabs(algebraic[89])
    algebraic[96] = algebraic[92]+algebraic[94]
    rates[4] = algebraic[96]-(states[4]-states[20])/constants[72]
    algebraic[90] = constants[12]*algebraic[45]*algebraic[89]
    # Cálculo das grandezas auxiliares da corrente trocadora sódio-cálcio (INaCa).
    algebraic[67] = 1.00000/(1.00000+power(constants[42]/states[20], 3.00000))
    algebraic[95] = (power(states[19], 3.00000))*constants[4]*exp(states[0]*0.350000*constants[75])-(power(constants[5], 3.00000))*algebraic[88]*exp(states[0]*(0.350000-1.00000)*constants[75])
    algebraic[66] = 1.00000+0.200000*exp(states[0]*(0.350000-1.00000)*constants[75])
    algebraic[97] = constants[43]*(power(states[19], 3.00000))+(power(constants[44], 3.00000))*algebraic[88]
    algebraic[98] = (power(constants[45], 3.00000))*constants[4]*(1.00000+algebraic[88]/constants[46])
    algebraic[68] = constants[46]*(power(constants[5], 3.00000))*(1.00000+power(states[19]/constants[45], 3.00000))
    algebraic[99] = (power(states[19], 3.00000))*constants[4]+(power(constants[5], 3.00000))*algebraic[88]
    algebraic[100] = algebraic[97]+algebraic[98]+algebraic[68]+algebraic[99]
    algebraic[101] = (constants[41]*algebraic[67]*algebraic[95])/(algebraic[66]*algebraic[100])
    # Tamponamento de cálcio no espaço submembranar e evolução de Ca_submem.
    algebraic[74] = (constants[60]*constants[61])/((constants[61]+states[20])*(constants[61]+states[20]))
    algebraic[75] = (constants[62]*constants[63])/((constants[63]+states[20])*(constants[63]+states[20]))
    algebraic[76] = (constants[64]*constants[65])/((constants[65]+states[20])*(constants[65]+states[20]))
    algebraic[77] = (constants[66]*constants[67])/((constants[67]+states[20])*(constants[67]+states[20]))
    algebraic[78] = 1.00000/(1.00000+algebraic[74]+algebraic[75]+algebraic[76]+algebraic[77])
    rates[20] = algebraic[78]*(50.0000*(((states[23]-algebraic[84])-algebraic[90])+algebraic[101])-algebraic[85])
    # Bomba Na+/K+ (INaK), potencial de reversão do sódio e corrente rápida de sódio (INa).
    algebraic[63] = 1.00000/(1.00000+0.124500*exp(-0.100000*states[0]*constants[75])+0.0365000*constants[77]*exp(-states[0]*constants[75]))
    algebraic[65] = (((constants[38]*algebraic[63]*states[19])/(states[19]+constants[40]))*constants[3])/(constants[3]+constants[39])
    algebraic[102] = constants[6]*algebraic[101]
    algebraic[105] = (1.00000/constants[75])*log(constants[5]/states[19])
    algebraic[106] = constants[11]*states[2]*states[3]*states[1]*states[1]*states[1]*(states[0]-algebraic[105])
    rates[19] = -(algebraic[106]+3.00000*algebraic[65]+3.00000*algebraic[102])/(constants[6]*1000.00)
    # Correntes de potássio: IK1, Ito, IKr e IKs, com seus potenciais de reversão.
    algebraic[46] = 1.02000/(1.00000+exp(0.238500*((states[0]-constants[79])-59.2150)))
    algebraic[47] = (0.491240*exp(0.0803200*((states[0]-constants[79])+5.47600))+1.00000*exp(0.0617500*((states[0]-constants[79])-594.310)))/(1.00000+exp(-0.514300*((states[0]-constants[79])+4.75300)))
    algebraic[48] = algebraic[46]/(algebraic[46]+algebraic[47])
    algebraic[49] = constants[33]*(power(constants[3]/5.40000, 1.0/2))*algebraic[48]*(states[0]-constants[79])
    algebraic[54] = 1.00000/(1.00000+exp(algebraic[53]))
    algebraic[56] = constants[36]*states[15]*(states[16]+0.500000*algebraic[54])*(states[0]-constants[79])
    algebraic[58] = constants[37]*states[17]*states[18]*(states[0]-constants[79])
    algebraic[60] = algebraic[56]+algebraic[58]
    algebraic[93] = 2.00000*constants[6]*algebraic[90]
    algebraic[50] = 1.00000/(1.00000+exp((states[0]+33.0000)/22.4000))
    algebraic[51] = constants[34]*(power(constants[3]/5.40000, 1.0/2))*states[11]*algebraic[50]*(states[0]-constants[79])
    algebraic[103] = (1.00000/constants[75])*log((constants[3]+constants[74]*constants[5])/(constants[73]+constants[74]*states[19]))
    algebraic[52] = 1.00000+0.800000/(1.00000+power(0.500000/states[12], 3.00000))
    algebraic[104] = constants[35]*algebraic[52]*states[13]*states[14]*(states[0]-algebraic[103])
    # Estímulo externo periódico: calcula o início do ciclo e verifica se o pulso está ativo.
    algebraic[3] = floor(voi/constants[8])*constants[8]
    algebraic[10] = _piecewise2(greater_equal(voi-algebraic[3] , constants[7]) & less_equal(voi-algebraic[3] , constants[7]+constants[9]), constants[10], 0.00000)
    # Soma das correntes de membrana; o sinal negativo converte a soma em dV/dt.
    algebraic[107] = -(algebraic[106]+algebraic[49]+algebraic[51]+algebraic[104]+algebraic[60]+algebraic[102]+algebraic[93]+algebraic[65]+algebraic[10])
    rates[0] = algebraic[107]
    return(rates)

# Recalcula as variáveis algébricas para todos os instantes já simulados.
# Diferentemente de computeRates(), aqui states e voi contêm séries temporais completas.
def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    # Taxas de abertura/fechamento dos canais de sódio.
    algebraic[1] = custom_piecewise([less(states[0] , -40.0000), 0.135000*exp((80.0000+states[0])/-6.80000) , True, 0.00000])
    algebraic[8] = custom_piecewise([less(states[0] , -40.0000), 3.56000*exp(0.0790000*states[0])+310000.*exp(0.350000*states[0]) , True, 1.00000/(0.130000*(1.00000+exp((states[0]+10.6600)/-11.1000)))])
    algebraic[2] = custom_piecewise([less(states[0] , -40.0000), ((-127140.*exp(0.244400*states[0])-3.47400e-05*exp(-0.0439100*states[0]))*1.00000*(states[0]+37.7800))/(1.00000+exp(0.311000*(states[0]+79.2300))) , True, 0.00000])
    algebraic[9] = custom_piecewise([less(states[0] , -40.0000), (0.121200*exp(-0.0105200*states[0]))/(1.00000+exp(-0.137800*(states[0]+40.1400))) , True, (0.300000*exp(-2.53500e-07*states[0]))/(1.00000+exp(-0.100000*(states[0]+32.0000)))])
    algebraic[0] = custom_piecewise([greater(fabs(states[0]+47.1300) , 0.00100000), (0.320000*1.00000*(states[0]+47.1300))/(1.00000-exp(-0.100000*(states[0]+47.1300))) , True, 3.20000])
    algebraic[7] = 0.0800000*exp(-states[0]/11.0000)
    # Variáveis de ativação e tempos característicos das correntes de potássio.
    algebraic[5] = 1.00000/(1.00000+exp(-(states[0]-1.50000)/16.7000))
    algebraic[16] = custom_piecewise([less(fabs(states[0]+30.0000) , 0.00100000/0.0687000), 1.00000/(7.19000e-05/0.148000+0.000131000/0.0687000) , True, 1.00000/((7.19000e-05*(states[0]+30.0000))/(1.00000-exp(-0.148000*(states[0]+30.0000)))+(0.000131000*(states[0]+30.0000))/(exp(0.0687000*(states[0]+30.0000))-1.00000))])
    algebraic[4] = custom_piecewise([greater(fabs(states[0]+7.00000) , 0.00100000), (0.00138000*1.00000*(states[0]+7.00000))/(1.00000-exp(-0.123000*(states[0]+7.00000))) , True, 0.00138000/0.123000])
    algebraic[11] = custom_piecewise([greater(fabs(states[0]+10.0000) , 0.00100000), (0.000610000*1.00000*(states[0]+10.0000))/(exp(0.145000*(states[0]+10.0000))-1.00000) , True, 0.000610000/0.145000])
    algebraic[15] = 1.00000/(algebraic[4]+algebraic[11])
    algebraic[19] = 1.00000/(1.00000+exp(-(states[0]+50.0000)/7.50000))
    algebraic[12] = algebraic[5]
    algebraic[20] = 4.00000*algebraic[16]
    algebraic[6] = -(states[0]+3.00000)/15.0000
    algebraic[17] = 1.00000/(1.00000+exp(algebraic[6]))
    algebraic[22] = 9.00000/(1.00000+exp(-algebraic[6]))+0.500000
    algebraic[21] = algebraic[17]
    algebraic[13] = ((-states[0]/30.0000)*states[0])/30.0000
    algebraic[24] = 3.50000*exp(algebraic[13])+1.50000
    # Variáveis algébricas do canal de cálcio tipo L.
    algebraic[18] = 1.00000/(1.00000+exp(-(states[0]-constants[14])/constants[15]))
    algebraic[23] = algebraic[18]/constants[31]
    algebraic[25] = (1.00000-algebraic[18])/constants[31]
    algebraic[26] = 1.00000/(1.00000+power(constants[22]/states[4], 3.00000))
    algebraic[34] = 10.0000+4954.00*exp(states[0]/15.6000)
    algebraic[35] = constants[30]/(1.00000+power(states[4]/constants[23], 4.00000))+0.100000
    algebraic[33] = 1.00000-1.00000/(1.00000+exp(-(states[0]-constants[18])/constants[19]))
    algebraic[36] = (algebraic[34]-algebraic[35])*algebraic[33]+algebraic[35]
    algebraic[38] = 1.00000/(1.00000+exp(-(states[0]-constants[20])/constants[21]))
    algebraic[39] = (algebraic[26]*algebraic[38])/algebraic[36]
    algebraic[40] = (1.00000-algebraic[38])/algebraic[36]
    algebraic[37] = (algebraic[34]-450.000)*algebraic[33]+450.000
    algebraic[41] = algebraic[38]/algebraic[37]
    algebraic[42] = (1.00000-algebraic[38])/algebraic[37]
    algebraic[30] = 1.00000/(1.00000+exp(-(states[0]-constants[16])/constants[17]))
    algebraic[31] = (1.00000-algebraic[30])/constants[32]
    algebraic[28] = 0.0241680*algebraic[26]
    algebraic[43] = (((((algebraic[31]*algebraic[23])/algebraic[25])*algebraic[28])/constants[24])*algebraic[40])/algebraic[39]
    algebraic[32] = algebraic[31]
    algebraic[44] = (((((algebraic[32]*algebraic[23])/algebraic[25])*constants[25])/constants[26])*algebraic[42])/algebraic[41]
    algebraic[45] = (((((1.00000-states[7])-states[9])-states[8])-states[10])-states[5])-states[6]
    algebraic[27] = 0.0182688*algebraic[26]
    algebraic[29] = (((algebraic[27]*constants[27])/constants[28])*constants[24])/algebraic[28]
    # Variáveis algébricas das correntes Ito.
    algebraic[53] = (states[0]+33.5000)/10.0000
    algebraic[59] = 1.00000/(1.00000+exp(algebraic[53]))
    algebraic[55] = (states[0]+60.0000)/10.0000
    algebraic[62] = 3000.00/(1.00000+exp(algebraic[55]))+30.0000
    algebraic[61] = algebraic[59]
    algebraic[57] = (states[0]+33.5000)/10.0000
    algebraic[64] = 20.0000/(1.00000+exp(algebraic[57]))+20.0000
    # Fluxos de cálcio e fatores de tamponamento.
    algebraic[72] = (constants[58]*states[12]*states[12])/(states[12]*states[12]+constants[56]*constants[56])
    algebraic[73] = ((constants[59]*states[21]*states[21])/(states[21]*states[21]+constants[57]*constants[57]))*(states[21]*16.6670-states[12])
    algebraic[79] = (constants[60]*constants[61])/((constants[61]+states[12])*(constants[61]+states[12]))
    algebraic[80] = (constants[62]*constants[63])/((constants[63]+states[12])*(constants[63]+states[12]))
    algebraic[81] = (constants[64]*constants[65])/((constants[65]+states[12])*(constants[65]+states[12]))
    algebraic[82] = (constants[66]*constants[67])/((constants[67]+states[12])*(constants[67]+states[12]))
    algebraic[83] = 1.00000/(1.00000+algebraic[79]+algebraic[80]+algebraic[81]+algebraic[82])
    algebraic[86] = constants[68]*states[12]*(constants[70]-states[24])-constants[69]*states[24]
    algebraic[84] = (states[20]-states[12])/constants[71]
    algebraic[85] = constants[68]*states[20]*(constants[70]-states[25])-constants[69]*states[25]
    algebraic[87] = (-states[23]+algebraic[72])-algebraic[73]
    # Liberação de cálcio do retículo e dependência dos estados do canal.
    algebraic[69] = custom_piecewise([greater(states[22] , 50.0000) & less(states[22] , constants[47]), (states[22]-50.0000)/1.00000 , greater_equal(states[22] , constants[47]), constants[53]*states[22]+constants[78] , True, 0.00000])
    algebraic[70] = (states[21]*algebraic[69])/constants[47]
    algebraic[88] = states[20]/1000.00
    algebraic[14] = states[0]*2.00000*constants[75]
    algebraic[89] = custom_piecewise([less(fabs(algebraic[14]) , 0.00100000), (4.00000*constants[13]*constants[2]*constants[75]*(algebraic[88]*exp(algebraic[14])-0.341000*constants[4]))/(2.00000*constants[75]) , True, (4.00000*constants[13]*states[0]*constants[2]*constants[75]*(algebraic[88]*exp(algebraic[14])-0.341000*constants[4]))/(exp(algebraic[14])-1.00000)])
    algebraic[71] = exp(-constants[52]*(states[0]+30.0000))/(1.00000+exp(-constants[52]*(states[0]+30.0000)))
    algebraic[91] = (constants[48]/1.00000)*algebraic[45]*fabs(algebraic[89])*algebraic[71]
    algebraic[92] = (((algebraic[45]*algebraic[70]*fabs(algebraic[89])*constants[49])/1.00000)*exp(-constants[51]*(states[0]+30.0000)))/(1.00000+exp(-constants[51]*(states[0]+30.0000)))
    algebraic[94] = algebraic[45]*constants[50]*fabs(algebraic[89])
    algebraic[96] = algebraic[92]+algebraic[94]
    algebraic[90] = constants[12]*algebraic[45]*algebraic[89]
    algebraic[67] = 1.00000/(1.00000+power(constants[42]/states[20], 3.00000))
    algebraic[95] = (power(states[19], 3.00000))*constants[4]*exp(states[0]*0.350000*constants[75])-(power(constants[5], 3.00000))*algebraic[88]*exp(states[0]*(0.350000-1.00000)*constants[75])
    algebraic[66] = 1.00000+0.200000*exp(states[0]*(0.350000-1.00000)*constants[75])
    algebraic[97] = constants[43]*(power(states[19], 3.00000))+(power(constants[44], 3.00000))*algebraic[88]
    algebraic[98] = (power(constants[45], 3.00000))*constants[4]*(1.00000+algebraic[88]/constants[46])
    algebraic[68] = constants[46]*(power(constants[5], 3.00000))*(1.00000+power(states[19]/constants[45], 3.00000))
    algebraic[99] = (power(states[19], 3.00000))*constants[4]+(power(constants[5], 3.00000))*algebraic[88]
    algebraic[100] = algebraic[97]+algebraic[98]+algebraic[68]+algebraic[99]
    algebraic[101] = (constants[41]*algebraic[67]*algebraic[95])/(algebraic[66]*algebraic[100])
    algebraic[74] = (constants[60]*constants[61])/((constants[61]+states[20])*(constants[61]+states[20]))
    algebraic[75] = (constants[62]*constants[63])/((constants[63]+states[20])*(constants[63]+states[20]))
    algebraic[76] = (constants[64]*constants[65])/((constants[65]+states[20])*(constants[65]+states[20]))
    algebraic[77] = (constants[66]*constants[67])/((constants[67]+states[20])*(constants[67]+states[20]))
    algebraic[78] = 1.00000/(1.00000+algebraic[74]+algebraic[75]+algebraic[76]+algebraic[77])
    # Correntes iônicas finais e estímulo, calculados para cada instante da simulação.
    algebraic[63] = 1.00000/(1.00000+0.124500*exp(-0.100000*states[0]*constants[75])+0.0365000*constants[77]*exp(-states[0]*constants[75]))
    algebraic[65] = (((constants[38]*algebraic[63]*states[19])/(states[19]+constants[40]))*constants[3])/(constants[3]+constants[39])
    algebraic[102] = constants[6]*algebraic[101]
    algebraic[105] = (1.00000/constants[75])*log(constants[5]/states[19])
    algebraic[106] = constants[11]*states[2]*states[3]*states[1]*states[1]*states[1]*(states[0]-algebraic[105])
    algebraic[46] = 1.02000/(1.00000+exp(0.238500*((states[0]-constants[79])-59.2150)))
    algebraic[47] = (0.491240*exp(0.0803200*((states[0]-constants[79])+5.47600))+1.00000*exp(0.0617500*((states[0]-constants[79])-594.310)))/(1.00000+exp(-0.514300*((states[0]-constants[79])+4.75300)))
    algebraic[48] = algebraic[46]/(algebraic[46]+algebraic[47])
    algebraic[49] = constants[33]*(power(constants[3]/5.40000, 1.0/2))*algebraic[48]*(states[0]-constants[79])
    algebraic[54] = 1.00000/(1.00000+exp(algebraic[53]))
    algebraic[56] = constants[36]*states[15]*(states[16]+0.500000*algebraic[54])*(states[0]-constants[79])
    algebraic[58] = constants[37]*states[17]*states[18]*(states[0]-constants[79])
    algebraic[60] = algebraic[56]+algebraic[58]
    algebraic[93] = 2.00000*constants[6]*algebraic[90]
    algebraic[50] = 1.00000/(1.00000+exp((states[0]+33.0000)/22.4000))
    algebraic[51] = constants[34]*(power(constants[3]/5.40000, 1.0/2))*states[11]*algebraic[50]*(states[0]-constants[79])
    algebraic[103] = (1.00000/constants[75])*log((constants[3]+constants[74]*constants[5])/(constants[73]+constants[74]*states[19]))
    algebraic[52] = 1.00000+0.800000/(1.00000+power(0.500000/states[12], 3.00000))
    algebraic[104] = constants[35]*algebraic[52]*states[13]*states[14]*(states[0]-algebraic[103])
    algebraic[3] = floor(voi/constants[8])*constants[8]
    algebraic[10] = custom_piecewise([greater_equal(voi-algebraic[3] , constants[7]) & less_equal(voi-algebraic[3] , constants[7]+constants[9]), constants[10] , True, 0.00000])
    algebraic[107] = -(algebraic[106]+algebraic[49]+algebraic[51]+algebraic[104]+algebraic[60]+algebraic[102]+algebraic[93]+algebraic[65]+algebraic[10])
    return algebraic

# Converte uma lista de condições e resultados em uma função por partes.
# Os elementos alternam entre condição e valor; select escolhe o valor cuja condição é verdadeira.
def custom_piecewise(cases):
    """Compute result of a piecewise function"""
    return select(cases[0::2],cases[1::2])

# Integra o sistema de EDOs nos instantes de voi e devolve (states, n_ok):
# states tem shape (sizeStates, len(voi)); n_ok é o número de instantes calculados com sucesso.
def _integrate(init_states, constants, voi, atol=1e-06, rtol=1e-06, max_step=1):
    """Integra o modelo com LSODA e interrompe a integração nas bordas dos estímulos.

    A integração é segmentada nos instantes de início/fim de cada pulso porque
    a corrente de estímulo é descontínua. `voi` determina somente os instantes
    de saída; o solver escolhe os passos internos de forma adaptativa.
    """
    from scipy.integrate import solve_ivp

    voi = np.asarray(voi, dtype=float)
    if voi.ndim != 1 or voi.size < 2 or np.any(np.diff(voi) <= 0):
        raise ValueError("voi deve ser um vetor crescente com pelo menos dois instantes.")
    if max_step <= 0:
        raise ValueError("max_step deve ser positivo.")

    t_start = float(voi[0])
    t_end = float(voi[-1])
    period = float(constants[8])
    stim_offset = float(constants[7])
    stim_duration = float(constants[9])

    # Força o solver a parar nas descontinuidades da corrente de estímulo.
    breakpoints = [t_start, t_end]
    if period > 0:
        k_min = max(0, int(np.floor((t_start - stim_offset) / period)) - 1)
        k_max = int(np.ceil((t_end - stim_offset) / period)) + 1
        for k in range(k_min, k_max + 1):
            pulse_start = k * period + stim_offset
            pulse_end = pulse_start + stim_duration
            if t_start < pulse_start < t_end:
                breakpoints.append(pulse_start)
            if t_start < pulse_end < t_end:
                breakpoints.append(pulse_end)
    breakpoints = np.unique(np.asarray(breakpoints, dtype=float))

    states_out = np.full((sizeStates, len(voi)), np.nan, dtype=float)
    y0 = np.asarray(init_states, dtype=np.float64)
    constants_array = np.asarray(constants, dtype=np.float64)

    def rhs(t, y):
        return computeRates(t, y, constants_array)

    for seg_start, seg_end in zip(breakpoints[:-1], breakpoints[1:]):
        solution = solve_ivp(
            rhs,
            (float(seg_start), float(seg_end)),
            y0,
            method="LSODA",
            atol=atol,
            rtol=rtol,
            max_step=max_step,
            dense_output=True,
        )

        if solution.sol is not None:
            if solution.success:
                in_segment = (voi >= seg_start - 1e-10) & (voi <= seg_end + 1e-10)
                indices = np.flatnonzero(in_segment)
                if indices.size:
                    states_out[:, indices] = solution.sol(voi[indices])
            else:
                # Mantém apenas as amostras alcançadas antes da falha.
                in_segment = (
                    (voi >= seg_start - 1e-10)
                    & (voi <= solution.t[-1] + 1e-10)
                )
                indices = np.flatnonzero(in_segment)
                if indices.size:
                    states_out[:, indices] = solution.sol(voi[indices])

        if not solution.success:
            break
        y0 = solution.y[:, -1]

    # Retorna o prefixo contínuo calculado. A função chamadora emite um aviso
    # caso o solver não tenha chegado a todos os instantes pedidos.
    valid = np.all(np.isfinite(states_out), axis=0)
    invalid_indices = np.flatnonzero(~valid)
    n_ok = int(invalid_indices[0]) if invalid_indices.size else len(voi)
    return states_out[:, :n_ok], n_ok

def solve_model():
    """Solve model with ODE solver"""
    # Obtém os estados iniciais e os parâmetros do modelo.
    # Initialise constants and state variables
    (init_states, constants) = initConsts()

    # Define os instantes de saída da simulação (tempo em ms).
    # Set timespan to solve over
    voi = linspace(0, 300, 500)

    # Integra o modelo.
    # Solve model
    (states, n_ok) = _integrate(init_states, constants, voi)

    # Calcula as variáveis algébricas a partir das trajetórias dos estados.
    # Compute algebraic variables
    algebraic = computeAlgebraic(constants, states, voi)
    return (voi, states, algebraic)

def simulate_cell(cell_params=None, *, t_span=(0.0, 10.0), n_points=500, t_eval=None,
                  stim_amplitude=None, stim_duration=None, stim_period=None, stim_offset=None,
                  initial_states=None, atol=1e-06, rtol=1e-06, max_step=1.0):
    """Simulate the cellular model and return the action potential and the states.

    Parameters
    ----------
    cell_params : dict, optional
        Model parameters by name, e.g. {"gna": 10, "gkr": 0.02, "K_o": 4.0}. It may contain
        only some of them: every parameter not given keeps the default value defined in this
        file. Valid names are the keys of get_default_params(). Unknown names raise KeyError.
        Derived constants (FonRT, s2t, sigma, bv, ek) are recomputed automatically from the
        parameters they depend on; pass them explicitly only to force a specific value.
    t_span : (t_start, t_end), optional
        Simulation interval in ms. Default (0, 10), as in the original script.
    n_points : int, optional
        Number of equally spaced output times in t_span. Default 500.
    t_eval : array-like, optional
        Explicit, strictly increasing output times (ms). Overrides t_span and n_points.
    stim_amplitude, stim_duration, stim_period, stim_offset : float, optional
        Periodic stimulus (nA/nF, ms, ms, ms). When given, they take precedence over the same
        names in cell_params. The pulse is active when
        (t mod stim_period) is between stim_offset and stim_offset + stim_duration.
        Use stim_amplitude=0 for no stimulus.
    initial_states : dict, optional
        Initial values of state variables by name, e.g. {"V": -80, "Ca_i": 0.3}.
        Unspecified states keep their default initial values.
    atol, rtol, max_step : float, optional
        Tolerances and maximum step (ms) of the LSODA solver. The integration is split at the
        start/end of every stimulus pulse, so a pulse shorter than max_step is not skipped.

    Returns
    -------
    t : ndarray, shape (n,)
        Output times in ms.
    V : ndarray, shape (n,)
        Membrane potential (action potential) in mV; same as states["V"].
    states : dict
        {state name: ndarray of shape (n,)} for the 26 state variables
        (V, xm, xh, xj, Ca_dyad, ..., Ca_i, Na_i, Ca_NSR, Ca_JSR, ...).
        Use get_default_states() to list the names.

    If the solver fails before the end, a warning is issued and the arrays are truncated
    to the instants computed successfully.

    Example
    -------
    >>> from modelo_celular_comentado import simulate_cell
    >>> t, V, states = simulate_cell({"gna": 10.0}, t_span=(0, 800), n_points=4000,
    ...                              stim_period=400, stim_amplitude=-15)
    >>> ca_i = states["Ca_i"]
    """
    # Valores do dicionário < argumentos explícitos de estímulo.
    overrides = {} if cell_params is None else dict(cell_params)
    stim = {"stim_amplitude": stim_amplitude, "stim_duration": stim_duration,
            "stim_period": stim_period, "stim_offset": stim_offset}
    overrides.update({k: v for k, v in stim.items() if v is not None})

    (init_states, constants) = initConsts(overrides, initial_states)

    # Instantes de saída.
    if t_eval is None:
        (t_start, t_end) = (float(t_span[0]), float(t_span[1]))
        if not t_end > t_start:
            raise ValueError("t_span must satisfy t_end > t_start.")
        if int(n_points) < 2:
            raise ValueError("n_points must be at least 2.")
        voi = np.linspace(t_start, t_end, int(n_points))
    else:
        voi = np.asarray(t_eval, dtype=float)
        if voi.ndim != 1 or voi.size < 2 or np.any(np.diff(voi) <= 0):
            raise ValueError("t_eval must be a 1-D, strictly increasing array with at least 2 points.")

    (states, n_ok) = _integrate(init_states, constants, voi, atol=atol, rtol=rtol, max_step=max_step)
    if n_ok < len(voi):
        warnings.warn("ODE solver failed at t = %.6g ms; returning only the %d points computed "
                      "before the failure." % (voi[n_ok], n_ok))
        voi = voi[:n_ok]
        states = states[:, :n_ok]

    states_dict = {name: states[i].copy() for (name, i) in _STATE_INDEX.items()}
    return voi, states_dict["V"], states_dict

def get_default_params(include_derived=False):
    """Return {name: default value} for the model parameters (constants).

    Derived constants (FonRT, s2t, sigma, bv, ek) are left out by default so that a copy
    of this dict can be edited and passed to simulate_cell() without freezing them."""
    (_, constants) = initConsts()
    return {name: constants[i] for (name, i) in _CONSTANT_INDEX.items()
            if include_derived or name not in _DERIVED_NAMES}

def get_default_states():
    """Return {name: default initial value} for the state variables."""
    (states, _) = initConsts()
    return {name: states[i] for (name, i) in _STATE_INDEX.items()}

# Plota todas as variáveis de estado e algébricas ao longo do tempo.
def plot_model(voi, states, algebraic):
    """Plot variables against variable of integration"""
    import pylab
    (legend_states, legend_algebraic, legend_voi, legend_constants) = createLegends()
    pylab.figure(1)
    pylab.plot(voi,vstack((states,algebraic)).T)
    pylab.xlabel(legend_voi)
    pylab.legend(legend_states + legend_algebraic, loc='best')
    pylab.show()

def plot_vm(voi, states, algebraic=None):
    """Plot only the membrane potential (Vm) against variable of integration"""
    import pylab

    legend_states, legend_algebraic, legend_voi, legend_constants = (
        createLegends()
    )

    pylab.figure(1)
    pylab.plot(voi, states[0], label=legend_states[0], color="b")
    pylab.xlabel(legend_voi)
    pylab.ylabel("V (mV)")
    pylab.title("Potencial de Ação")
    pylab.grid(True, linestyle="--", alpha=0.7)
    pylab.legend(loc="best")
    pylab.show()


# Ponto de entrada: executa a simulação e mostra os gráficos quando o arquivo é rodado diretamente.
if __name__ == "__main__":
    (voi, states, algebraic) = solve_model()
    # plot_model(voi, states, algebraic)
    plot_vm(voi, states, algebraic)