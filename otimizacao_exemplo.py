import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination.default import DefaultMultiObjectiveTermination
from scipy.integrate import solve_ivp
import pandas as pd
import time
from numba import njit, float64

# Constantes de conversão
scale = 122.6589
offset = -88.637

# Parâmetros iniciais desnormalizados
initial_parameters = {
    'u_o': -88.74687619325046,
    'u_u': 132.54546257306978,
    'theta_v': -45.70796132356614,
    'theta_w': -77.94089038677906,
    'theta_v_minus': -87.6145560705125,
    'theta_o': -109.07252255977295,
    'tau_v1_minus': 56.0,
    'tau_v2_minus': 1060.9375,
    'tau_v_plus': 1.451172,
    'tau_w1_minus': 91.894531,
    'tau_w2_minus': 4.648438,
    'k_w_minus': 35.546875,
    'u_w_minus': -58.9306,
    'tau_w_plus': 149.57098486557356,
    'tau_fi': 675.8477,
    'tau_o1': 47218.91,
    'tau_o2': 123.38,
    'tau_so1': 37.84000287685694,
    'tau_so2': 1.175781,
    'k_so': 0.0324,
    'u_so': 12.3395,
    'tau_s1': 1.567383,
    'tau_s2': 19.912109,
    'k_s': 0.001832,
    'u_s': 169.379452,
    'tau_si': 4.643555,
    'tau_w_inf': 8.2652,
    'w_inf_star': 1.064158046721859,
    'dt': 0.01,
    'pico': 0.325
}

# Definir os parâmetros a serem otimizados com base na análise de sensibilidade
parameters_to_optimize = [
    'u_o',        # Presente em ambas as listas
    'u_u',        # Presente em ambas as listas
    'theta_v',    # Presente na lista ST
    'theta_w',    # Presente em ambas as listas
    'theta_v_minus', # Presente na lista ST
    'theta_o',    # Presente em ambas as listas
    'tau_w_plus', # Presente em ambas as listas
    'tau_so1',    # Presente na lista S1
    'w_inf_star'  # Presente na lista S1
]

# Definir limites para cada parâmetro (valores personalizados ou ± 30%)
bounds = []
for param in parameters_to_optimize:
    # Valores personalizados para parâmetros específicos
    if param == 'theta_w':
        bounds.append((-94.0, -70.0))  # Range específico para theta_w
    elif param == 'u_o':
        bounds.append((-94.0, -88.0))  # Range específico para u_o
    elif param == 'tau_w_plus':
        bounds.append((30.0, 175.0))  # Range específico para tau_w_plus
    elif param == 'tau_v_plus':
        bounds.append((1.0, 7.0))  # Range específico para tau_v_plus
    elif param == 'tau_fi':
        bounds.append((475.0, 744.0))  # Range específico para tau_fi
    elif param == 'tau_v1_minus':
        bounds.append((125.0, 275.0))  # Range específico para tau_v1_minus
    elif param == 'u_u':
        bounds.append((112.0, 138.0))  # Range específico para u_u
    elif param == 'tau_so1':
        bounds.append((35.0, 42.0))  # Range específico para tau_so1
    elif param == 'tau_si':
        bounds.append((4.0, 11.0))  # Range específico para tau_si
    elif param == 'tau_w1_minus':
        bounds.append((40.0, 90.0))  # Range específico para tau_w1_minus
    elif param == 'u_s':
        bounds.append((100.0, 180.0))  # Range específico para u_s
    elif param == 'u_so':
        bounds.append((6.0, 25.0))  # Range específico para u_so
    elif param == 'tau_w_inf':
        bounds.append((143.0, 174.0))  # Range específico para tau_w_inf
    elif param == 'k_s':
        bounds.append((0.0009, 0.0024))  # Range específico para k_s
    elif param == 'k_so':
        bounds.append((0.0292, 0.0392))  # Range específico para k_so
    else:
        # Para os parâmetros não especificados, usar intervalo original (±30%)
        value = initial_parameters[param]
        if value < 0:
            # Para valores negativos, a ordem dos limites é invertida
            bounds.append((value * 1.3, value * 0.7))
        else:
            bounds.append((value * 0.7, value * 1.3))

# Funções do modelo cardíaco otimizadas com Numba
@njit
def H(x, y):
    """Função Heaviside otimizada com Numba"""
    if x > y:
        return 1.0
    elif x < y:
        return 0.0
    else:
        return 0.5

@njit
def v_inf(x, y):
    """Função v_inf otimizada com Numba"""
    if x < y:
        return 1.0
    else:
        return 0.0

# Versão numba-compilada do modelo cardíaco
@njit
def cardiac_model_odes_numba(t, y, 
                             u_o, u_u, theta_v, theta_w, theta_v_minus, theta_o,
                             tau_v1_minus, tau_v2_minus, tau_v_plus,
                             tau_w1_minus, tau_w2_minus, k_w_minus, u_w_minus, tau_w_plus,
                             tau_fi, tau_o1, tau_o2, tau_so1, tau_so2, k_so, u_so,
                             tau_s1, tau_s2, k_s, u_s, tau_si, tau_w_inf, w_inf_star, scale_factor):
    """Sistema de EDOs para o modelo cardíaco otimizado com Numba"""
    vm, v, w, s = y
    
    # Calcular constantes de tempo
    tauv = (1 - H(vm, theta_v_minus)) * tau_v1_minus + H(vm, theta_v_minus) * tau_v2_minus
    tauw = tau_w1_minus + (tau_w2_minus - tau_w1_minus) * (1 + np.tanh(k_w_minus * (vm - u_w_minus))) / 2.0
    taus0 = tau_so1 + (tau_so2 - tau_so1) * (1 + np.tanh(k_so * (vm - u_so))) / 2.0
    taus = (1 - H(vm, theta_w)) * tau_s1 + H(vm, theta_w) * tau_s2
    tau0 = (1 - H(vm, theta_o)) * tau_o1 + H(vm, theta_o) * tau_o2
    
    # Infinities
    vinf = v_inf(vm, theta_v_minus)
    winf = (1 - H(vm, theta_o)) * (1 - ((vm - u_o) / tau_w_inf)) + H(vm, theta_o) * w_inf_star
    
    # Correntes
    jfi = -v * H(vm, theta_v) * (vm - theta_v) * (u_u - (vm - u_o)) / tau_fi
    jso = ((vm - u_o)) * (1 - H(vm, theta_w)) / tau0 + H(vm, theta_w) / taus0
    jsi = -H(vm, theta_w) * w * s / tau_si
    
    # Derivadas - usando o fator de escala para dvm_dt
    dvm_dt = scale_factor * (-(jfi + jso + jsi))
    dv_dt = (1 - H(vm, theta_v)) * (vinf - v) / tauv - H(vm, theta_v) * v / tau_v_plus
    dw_dt = (1 - H(vm, theta_w)) * (winf - w) / tauw - H(vm, theta_w) * w / tau_w_plus
    ds_dt = ((1 + np.tanh(k_s * ((vm - u_o) - u_s))) / 2 - s) / taus
    
    return np.array([dvm_dt, dv_dt, dw_dt, ds_dt])

def cardiac_model_odes(t, y, params):
    """Função wrapper para compatibilidade com solve_ivp"""
    # Extrair parâmetros do dicionário
    u_o = params['u_o']
    u_u = params['u_u']
    theta_v = params['theta_v']
    theta_w = params['theta_w']
    theta_v_minus = params['theta_v_minus']
    theta_o = params['theta_o']
    tau_v1_minus = params['tau_v1_minus']
    tau_v2_minus = params['tau_v2_minus']
    tau_v_plus = params['tau_v_plus']
    tau_w1_minus = params['tau_w1_minus']
    tau_w2_minus = params['tau_w2_minus']
    k_w_minus = params['k_w_minus']
    u_w_minus = params['u_w_minus']
    tau_w_plus = params['tau_w_plus']
    tau_fi = params['tau_fi']
    tau_o1 = params['tau_o1']
    tau_o2 = params['tau_o2']
    tau_so1 = params['tau_so1']
    tau_so2 = params['tau_so2']
    k_so = params['k_so']
    u_so = params['u_so']
    tau_s1 = params['tau_s1']
    tau_s2 = params['tau_s2']
    k_s = params['k_s']
    u_s = params['u_s']
    tau_si = params['tau_si']
    tau_w_inf = params['tau_w_inf']
    w_inf_star = params['w_inf_star']
    
    # Chamar a versão otimizada com Numba
    return cardiac_model_odes_numba(
        t, y, u_o, u_u, theta_v, theta_w, theta_v_minus, theta_o,
        tau_v1_minus, tau_v2_minus, tau_v_plus,
        tau_w1_minus, tau_w2_minus, k_w_minus, u_w_minus, tau_w_plus,
        tau_fi, tau_o1, tau_o2, tau_so1, tau_so2, k_so, u_so,
        tau_s1, tau_s2, k_s, u_s, tau_si, tau_w_inf, w_inf_star, scale
    )

def run_simulation(params_dict, bcl=500, num_beats=16):
    """Executa a simulação do modelo cardíaco para um BCL específico"""
    dt = params_dict['dt']
    
    # Configuração inicial
    initial_state = [offset, 1.0, 1.0, 0.0]  # [vm, v, w, s]
    
    initial_state[0] += params_dict['pico'] * scale

    # Arrays para armazenar todas as soluções
    all_t = []
    all_vm = []
    all_v = []
    all_w = []
    all_s = []
    
    # Estado atual
    current_state = initial_state.copy()
    
    # Para cada batimento
    for beat in range(num_beats):
        # Tempos para este batimento
        t_start = beat * bcl
        t_end = (beat + 1) * bcl
        
        # Estimular no início do batimento (exceto o primeiro)
        if beat > 0:
            # Aplicar estímulo diretamente no potencial de membrana
            current_state[0] += params_dict['pico'] * scale
        
        # Resolver o sistema para este batimento
        sol = solve_ivp(
            lambda t, y: cardiac_model_odes(t, y, params_dict),
            [t_start, t_end],
            current_state,
            method='LSODA',
            rtol=1e-4,
            atol=1e-6,
            dense_output=True
        )
        
        # Pontos de tempo para saída uniforme
        t_uniform = np.arange(t_start, t_end, dt)
        
        # Obter solução interpolada em pontos uniformes
        sol_dense = sol.sol(t_uniform)
        
        # Armazenar resultados
        all_t.extend(t_uniform)
        all_vm.extend(sol_dense[0])
        all_v.extend(sol_dense[1])
        all_w.extend(sol_dense[2])
        all_s.extend(sol_dense[3])
        
        # Atualizar estado para o próximo batimento
        current_state = [sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1]]
    
    # Converter para arrays numpy
    t = np.array(all_t)
    vm = np.array(all_vm)
    v = np.array(all_v)
    w = np.array(all_w)
    s = np.array(all_s)
    
    # Analisar o último batimento para APD90
    last_beat_start = (num_beats - 1) * bcl
    idx_start = np.where(t >= last_beat_start)[0][0]
    
    # Encontrar o pico do potencial de ação no último batimento
    last_beat_vm = vm[idx_start:]
    
    # Extrair o último batimento
    last_beat_indices = np.where((t >= last_beat_start) & (t < num_beats * bcl))[0]
    last_beat_t = t[last_beat_indices] - last_beat_start
    last_beat_vm = vm[last_beat_indices]
    
    # Calcular APD90 e max dV/dt
    if len(last_beat_vm) > 0 and np.max(last_beat_vm) > offset + 10:  # Verificar se há PA
        peak_value = np.max(last_beat_vm)
        peak_idx = np.argmax(last_beat_vm)
        
        # Calcular derivada máxima durante upstroke
        dvdt = np.diff(last_beat_vm) / dt
        max_dvdt = np.max(dvdt[:peak_idx]) if peak_idx > 0 else 0
        
        # Calcular APD90
        threshold = offset + 0.1 * (peak_value - offset)  # 10% de repolarização (equivalente a APD90)
        
        # Encontrar quando o potencial cruza abaixo do threshold
        repol_idx = peak_idx
        for i in range(peak_idx, len(last_beat_vm)):
            if last_beat_vm[i] <= threshold:
                repol_idx = i
                break
        
        # Calcular APD
        apd = (repol_idx - peak_idx) * dt
    else:
        # Nenhum potencial de ação detectado
        apd = 0
        max_dvdt = 0
    
    return {
        't': last_beat_t,
        'vm': last_beat_vm,
        'apd': apd,
        'max_dvdt': max_dvdt,
        'bcl': bcl
    }

# NOVA FUNÇÃO: Simular múltiplos BCLs para gerar a curva de restituição
def run_restitution_curve(params_dict, bcl_values, num_beats=16):
    """
    Simula o modelo para múltiplos valores de BCL para gerar a curva de restituição
    """
    results = []
    
    for bcl in bcl_values:
        print(f"Simulando BCL = {bcl} ms...")
        result = run_simulation(params_dict, bcl=bcl, num_beats=num_beats)
        results.append(result)
        print(f"BCL = {bcl} ms: APD90 = {result['apd']:.2f} ms, max dV/dt = {result['max_dvdt']:.2f} mV/ms")
    
    return results

def load_restitution_data():
    """Carrega os dados da curva de restituição do arquivo"""
    try:
        restitution_data = pd.read_csv('torord_restitution_curve_results.txt', sep='\t')
        
        # Verificar se o arquivo tem o formato esperado
        expected_columns = ['BCL', 'APD90', 'Max_dVdt']
        for col in expected_columns:
            if col not in restitution_data.columns:
                print(f"Erro: Coluna {col} não encontrada no arquivo de dados de restituição")
                return None
        
        # Exibir estatísticas dos dados carregados
        bcl_values = sorted(restitution_data['BCL'].unique())
        print(f"Dados de restituição carregados para {len(bcl_values)} valores de BCL:")
        print(f"BCL (min-max): {min(bcl_values)}-{max(bcl_values)} ms")
        print(f"Total de pontos: {len(restitution_data)}")
        
        return restitution_data
    
    except Exception as e:
        print(f"Erro ao carregar dados da curva de restituição: {e}")
        return None

def load_ap_waveform():
    """Carrega a forma de onda do potencial de ação alvo para BCL=500"""
    try:
        # Tenta detectar automaticamente o separador (vírgula ou tabulação)
        try:
            # Primeiro tenta com tabulação
            ap_data = pd.read_csv('500_action_potential.txt', sep='\t', header=None)
            if ap_data.shape[1] == 1:  # Se tiver apenas uma coluna, provavelmente o separador está errado
                # Tenta com vírgula
                ap_data = pd.read_csv('500_action_potential.txt', sep=',', header=None)
        except:
            # Se falhar, tenta com vírgula
            ap_data = pd.read_csv('500_action_potential.txt', sep=',', header=None)
        
        # Verifica se o cabeçalho está sendo interpretado como dados
        first_row = ap_data.iloc[0, :]
        if any(isinstance(val, str) for val in first_row):
            print("Detectado cabeçalho no arquivo. Removendo primeira linha.")
            ap_data = ap_data.iloc[1:].reset_index(drop=True)
            # Converter para tipo numérico
            for col in ap_data.columns:
                ap_data[col] = pd.to_numeric(ap_data[col], errors='coerce')
        
        # Verifica se o arquivo tem formato adequado
        if ap_data.shape[1] >= 2:
            # Assumindo que a coluna 0 é o tempo e a coluna 1 é o potencial de membrana
            t = ap_data.iloc[:, 0].values
            vm = ap_data.iloc[:, 1].values
            
            print(f"Forma de onda do potencial de ação carregada: {len(t)} pontos")
            return t, vm
        else:
            print("Erro: Arquivo de potencial de ação não tem formato esperado (pelo menos 2 colunas)")
            return np.array([]), np.array([])
    
    except FileNotFoundError:
        print("Aviso: Arquivo '500_action_potential.txt' não encontrado. Otimização continuará sem considerar forma de onda.")
        return np.array([]), np.array([])
    
    except Exception as e:
        print(f"Erro ao carregar forma de onda do potencial de ação: {e}")
        return np.array([]), np.array([])

# Função de avaliação modificada para suportar múltiplos objetivos
def evaluate_objectives(x, restitution_data, target_ap_t, target_ap_vm):
    """
    Função de avaliação multi-objetivo para NSGA-II.
    Retorna três objetivos separados: erro APD, erro dV/dt e erro forma de onda
    """
    # Criar dicionário de parâmetros atualizado
    params_dict = initial_parameters.copy()
    for i, param in enumerate(parameters_to_optimize):
        params_dict[param] = x[i]
    
    # Verificar restrições dos parâmetros (para evitar combinações inválidas)
    if (params_dict['theta_v'] <= params_dict['theta_w'] or 
        params_dict['theta_w'] <= params_dict['theta_v_minus'] or 
        params_dict['theta_v_minus'] <= params_dict['theta_o']):
        return [1e10, 1e10, 1e10]  # Alta penalidade para todos os objetivos
    
    try:
        # Obter BCLs únicos do dataset de referência
        bcl_values = sorted(restitution_data['BCL'].unique())
        
        # Usar um subconjunto de BCLs para acelerar a otimização
        if len(bcl_values) > 10:
            # Selecionar BCLs representativos 
            step = max(1, len(bcl_values) // 6)  # Reduzir para 6 BCLs para agilizar
            bcl_subset = [bcl_values[0]]  # Menor BCL
            bcl_subset.extend(bcl_values[step::step])
            if bcl_values[-1] not in bcl_subset:
                bcl_subset.append(bcl_values[-1])  # Maior BCL
            
            # Garantir que BCL=500 esteja no conjunto para comparação da forma de onda
            if 500 not in bcl_subset and 500 in bcl_values:
                bcl_subset.append(500)
            
            bcl_subset.sort()
            print(f"Usando subconjunto de {len(bcl_subset)} BCLs para otimização: {bcl_subset}")
            bcl_values = bcl_subset
        
        # Simular para cada valor de BCL selecionado
        results = run_restitution_curve(params_dict, bcl_values, num_beats=16)
        
        # Calcular erro para cada ponto da curva de restituição
        apd_errors = []
        dvdt_errors = []
        
        for result in results:
            bcl = result['bcl']
            
            # Obter valores alvo para este BCL do conjunto de dados
            target_row = restitution_data[restitution_data['BCL'] == bcl]
            
            if not target_row.empty:
                target_apd = float(target_row['APD90'].values[0])
                target_dvdt = float(target_row['Max_dVdt'].values[0])
                
                # Verificar se a simulação produziu um potencial de ação válido
                if result['apd'] < 5 or result['max_dvdt'] < 5:
                    return [1e10, 1e10, 1e10]  # Alta penalidade para todos os objetivos
                
                # Erro no APD90 - normalizado
                apd_error = abs(result['apd'] - target_apd) / target_apd
                apd_errors.append(apd_error)
                
                # Erro na derivada máxima - normalizado
                dvdt_error = abs(result['max_dvdt'] - target_dvdt) / target_dvdt
                dvdt_errors.append(dvdt_error)
        
        # Erro médio em APD e dV/dt para todos os BCLs
        mean_apd_error = np.mean(apd_errors) if apd_errors else 1.0
        mean_dvdt_error = np.mean(dvdt_errors) if dvdt_errors else 1.0
        
        # Erro na forma de onda (apenas para BCL=500)
        waveform_error = 0.0
        bcl500_result = None
        
        # Encontrar resultado para BCL=500 (para comparação da forma de onda)
        for result in results:
            if result['bcl'] == 500:
                bcl500_result = result
                break
        
        # Calcular erro na forma de onda se tiver dados e resultado para BCL=500
        if (len(target_ap_t) > 0 and len(target_ap_vm) > 0 and bcl500_result is not None 
            and len(bcl500_result['t']) > 0 and len(bcl500_result['vm']) > 0):
            
            # Ajustar o intervalo de tempo do alvo para corresponder ao resultado
            target_t_max = min(target_ap_t[-1], bcl500_result['t'][-1])
            target_indices = target_ap_t <= target_t_max
            
            # Interpolar os valores do potencial simulado para os tempos alvo
            from scipy.interpolate import interp1d
            f = interp1d(bcl500_result['t'], bcl500_result['vm'], bounds_error=False, fill_value=bcl500_result['vm'][0])
            interpolated_vm = f(target_ap_t[target_indices])
            
            # Calcular erro da forma de onda (norma L2)
            waveform_error = np.sqrt(np.mean((interpolated_vm - target_ap_vm[target_indices])**2))
            # Normalizar pelo intervalo de voltagem do alvo
            voltage_range = np.max(target_ap_vm) - np.min(target_ap_vm)
            if voltage_range > 0:
                waveform_error = min(waveform_error / voltage_range, 1.0)
            else:
                waveform_error = 0.0
        else:
            # Se não tiver forma de onda, usar valor máximo como penalidade
            waveform_error = 1.0
        
        print(f"Erros: APD={mean_apd_error:.4f}, dV/dt={mean_dvdt_error:.4f}, Forma={waveform_error:.4f}")
        
        # Retornar os três objetivos separados para NSGA-II
        return [mean_apd_error, mean_dvdt_error, waveform_error]
    
    except Exception as e:
        print(f"Erro na simulação: {e}")
        return [1e10, 1e10, 1e10]  # Alta penalidade para todos os objetivos

# Define a classe de problema para pymoo - Multi-objetivo
class CardiacModelMultiObjectiveProblem(Problem):
    def __init__(self, restitution_data, target_ap_t, target_ap_vm):
        # n_var: número de variáveis (parâmetros a serem otimizados)
        # n_obj: número de objetivos (3 para otimização multi-objetivo)
        # n_constr: número de restrições (0 neste caso)
        # xl, xu: limites inferior e superior para cada variável
        n_var = len(parameters_to_optimize)
        n_obj = 3  # APD error, dV/dt error, waveform error
        xl = np.array([bound[0] for bound in bounds])
        xu = np.array([bound[1] for bound in bounds])
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
        
        self.restitution_data = restitution_data
        self.target_ap_t = target_ap_t
        self.target_ap_vm = target_ap_vm
    
    def _evaluate(self, x, out, *args, **kwargs):
        # Para múltiplos indivíduos (otimização baseada em população)
        n_individuals = len(x)
        F = np.zeros((n_individuals, self.n_obj))
        
        for i, ind in enumerate(x):
            F[i, :] = evaluate_objectives(ind, self.restitution_data, self.target_ap_t, self.target_ap_vm)
        
        out["F"] = F

def main():
    """Função principal para executar a otimização multi-objetivo com NSGA-II"""
    print("Iniciando otimização do modelo cardíaco usando NSGA-II...")
    
    # Compilar funções Numba (aquecimento)
    print("Compilando funções Numba...")
    dummy_t = 0.0
    dummy_y = np.array([offset, 1.0, 1.0, 0.0])
    cardiac_model_odes_numba(
        dummy_t, dummy_y, 
        initial_parameters['u_o'], 
        initial_parameters['u_u'], 
        initial_parameters['theta_v'], 
        initial_parameters['theta_w'], 
        initial_parameters['theta_v_minus'], 
        initial_parameters['theta_o'],
        initial_parameters['tau_v1_minus'],
        initial_parameters['tau_v2_minus'],
        initial_parameters['tau_v_plus'],
        initial_parameters['tau_w1_minus'],
        initial_parameters['tau_w2_minus'],
        initial_parameters['k_w_minus'],
        initial_parameters['u_w_minus'],
        initial_parameters['tau_w_plus'],
        initial_parameters['tau_fi'],
        initial_parameters['tau_o1'],
        initial_parameters['tau_o2'],
        initial_parameters['tau_so1'],
        initial_parameters['tau_so2'],
        initial_parameters['k_so'],
        initial_parameters['u_so'],
        initial_parameters['tau_s1'],
        initial_parameters['tau_s2'],
        initial_parameters['k_s'],
        initial_parameters['u_s'],
        initial_parameters['tau_si'],
        initial_parameters['tau_w_inf'],
        initial_parameters['w_inf_star'],
        scale
    )
    
    # Carregar dados da curva de restituição
    restitution_data = load_restitution_data()
    if restitution_data is None:
        print("Não foi possível carregar os dados da curva de restituição. Verifique o arquivo 'torord_restitution_curve_results.txt'.")
        return
    
    # Carregar forma de onda do potencial de ação (BCL=500)
    target_ap_t, target_ap_vm = load_ap_waveform()
    
    # Se não conseguiu carregar a forma de onda, verificar se deseja continuar
    if len(target_ap_t) == 0 or len(target_ap_vm) == 0:
        print("A otimização continuará sem considerar a forma de onda do potencial de ação.")
        print("Apenas APD90 e max dV/dt serão usados na função objetivo para diferentes BCLs.")
    
    # Configurar e executar a otimização com NSGA-II
    start_time = time.time()
    
    print("Iniciando otimização multi-objetivo com NSGA-II...")
    
    # Criar o problema para pymoo - Multi-objetivo
    problem = CardiacModelMultiObjectiveProblem(restitution_data, target_ap_t, target_ap_vm)
    
    # Configurar o algoritmo NSGA-II
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    
    algorithm = NSGA2(
        pop_size=50,                  # Tamanho da população - maior para cobertura do Pareto front
        n_offsprings=25,              # Número de descendentes por geração
        sampling=LHS(),               # Latin Hypercube Sampling
        crossover=SBX(prob=0.9, eta=15),  # Simulated Binary Crossover
        mutation=PM(eta=20),          # Polynomial Mutation
        eliminate_duplicates=True,    # Eliminar duplicatas para manter diversidade
    )
    
    # Configurar critérios de parada
    termination = DefaultMultiObjectiveTermination(
        xtol=1e-6,
        cvtol=1e-6,
        ftol=0.005,        # Tolerância na função objetivo
        n_max_gen=50,      # Número máximo de gerações
        n_max_evals=50*50  # Número máximo de avaliações
    )
    
    # Executar a otimização
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        verbose=True,
        save_history=True
    )
    
    end_time = time.time()
    print(f"Otimização concluída em {end_time - start_time:.2f} segundos")
    
    # Análise do Pareto front
    print("\nAnálise de Soluções Pareto-ótimas:")
    print(f"Número de soluções no Pareto front: {len(result.F)}")
    
    # Identificar a melhor solução "compromisso"
    # Normalizar os valores da função objetivo para dar peso igual a cada objetivo
    F_norm = result.F / np.max(result.F, axis=0)
    
    # Calcular a distância Euclidiana à origem (solução ideal seria [0,0,0])
    distances = np.sqrt(np.sum(F_norm**2, axis=1))
    best_idx = np.argmin(distances)
    
    print("\nMelhor solução de compromisso:")
    print(f"Índice: {best_idx}")
    print(f"Valores objetivos: APD Error = {result.F[best_idx, 0]:.4f}, " + 
          f"dV/dt Error = {result.F[best_idx, 1]:.4f}, " + 
          f"Waveform Error = {result.F[best_idx, 2]:.4f}")
    
    # Extrair os parâmetros da melhor solução
    best_params = initial_parameters.copy()
    for i, param in enumerate(parameters_to_optimize):
        best_params[param] = result.X[best_idx, i]
        print(f"{param}: {initial_parameters[param]} -> {result.X[best_idx, i]}")
    
    # Gerar e plotar a curva de restituição completa com os melhores parâmetros
    print("\nGerando curva de restituição completa com os melhores parâmetros...")
    
    # Obter todos os BCLs do dataset de referência
    all_bcl_values = sorted(restitution_data['BCL'].unique())
    
    # Simular a curva de restituição completa para a melhor solução
    best_results = run_restitution_curve(best_params, all_bcl_values, num_beats=16)
    
    # Extrair dados simulados para plotagem
    sim_bcl = [r['bcl'] for r in best_results]
    sim_apd = [r['apd'] for r in best_results]
    sim_dvdt = [r['max_dvdt'] for r in best_results]
    
    # Extrair dados de referência para plotagem
    ref_bcl = restitution_data['BCL'].values
    ref_apd = restitution_data['APD90'].values
    ref_dvdt = restitution_data['Max_dVdt'].values
    
    # Plotar curva de restituição de APD
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(ref_bcl, ref_apd, 'bo-', label='Referência')
    plt.plot(sim_bcl, sim_apd, 'ro-', label='Modelo Otimizado')
    plt.title('Curva de Restituição - APD90')
    plt.xlabel('BCL (ms)')
    plt.ylabel('APD90 (ms)')
    plt.grid(True)
    plt.legend()
    
    # Plotar curva de restituição de max dV/dt
    plt.subplot(2, 1, 2)
    plt.plot(ref_bcl, ref_dvdt, 'bo-', label='Referência')
    plt.plot(sim_bcl, sim_dvdt, 'ro-', label='Modelo Otimizado')
    plt.title('Curva de Restituição - Max dV/dt')
    plt.xlabel('BCL (ms)')
    plt.ylabel('Max dV/dt (mV/ms)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('curva_restituicao_otimizada_nsga2.png')
    
    # Plotar potencial de ação para BCL=500
    bcl500_result = None
    for r in best_results:
        if r['bcl'] == 500:
            bcl500_result = r
            break
    
    if bcl500_result is not None and len(target_ap_t) > 0 and len(target_ap_vm) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(target_ap_t, target_ap_vm, 'b-', label='Alvo')
        plt.plot(bcl500_result['t'], bcl500_result['vm'], 'r-', label='Otimizado')
        plt.title('Potencial de Ação (BCL=500ms)')
        plt.xlabel('Tempo (ms)')
        plt.ylabel('Potencial de Membrana (mV)')
        plt.legend()
        plt.grid(True)
        plt.savefig('potencial_acao_otimizado_nsga2.png')
    
    # Visualizar o front de Pareto
    plt.figure(figsize=(10, 8))
    
    # 3D Plot do front de Pareto
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotar todos os pontos do front de Pareto
    ax.scatter(result.F[:, 0], result.F[:, 1], result.F[:, 2], 
               c='blue', marker='o', alpha=0.5, label='Soluções Pareto')
    
    # Destacar a melhor solução de compromisso
    ax.scatter(result.F[best_idx, 0], result.F[best_idx, 1], result.F[best_idx, 2], 
               c='red', marker='*', s=100, label='Melhor compromisso')
    
    ax.set_xlabel('Erro APD')
    ax.set_ylabel('Erro dV/dt')
    ax.set_zlabel('Erro Forma de Onda')
    ax.set_title('Front de Pareto da Otimização NSGA-II')
    ax.legend()
    plt.savefig('pareto_front_nsga2.png')
    
    # Salvar todas as soluções do front de Pareto
    pareto_solutions = []
    for i in range(len(result.F)):
        solution = {
            'solution_id': i,
            'objectives': {
                'apd_error': result.F[i, 0],
                'dvdt_error': result.F[i, 1],
                'waveform_error': result.F[i, 2]
            },
            'is_best_compromise': (i == best_idx),
            'parameters': {}
        }
        
        # Adicionar parâmetros otimizados
        for j, param in enumerate(parameters_to_optimize):
            solution['parameters'][param] = result.X[i, j]
        
        pareto_solutions.append(solution)
    
    # Converter para DataFrame e salvar
    pareto_df = pd.DataFrame(pareto_solutions)
    pareto_df.to_excel('pareto_solutions_nsga2.xlsx', index=False)
    
    # Salvar parâmetros e resultados do melhor compromisso
    with open('parametros_otimizados_nsga2.txt', 'w') as f:
        f.write("Parâmetros Otimizados (Melhor Compromisso):\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        
        f.write("\nResultados da Curva de Restituição:\n")
        f.write("BCL\tAPD90_Sim\tAPD90_Ref\tErro_APD(%)\tMax_dVdt_Sim\tMax_dVdt_Ref\tErro_dVdt(%)\n")
        
        for i, bcl in enumerate(sim_bcl):
            ref_row = restitution_data[restitution_data['BCL'] == bcl]
            if not ref_row.empty:
                ref_apd_val = float(ref_row['APD90'].values[0])
                ref_dvdt_val = float(ref_row['Max_dVdt'].values[0])
                
                apd_error = abs(sim_apd[i] - ref_apd_val) / ref_apd_val * 100
                dvdt_error = abs(sim_dvdt[i] - ref_dvdt_val) / ref_dvdt_val * 100
                
                f.write(f"{bcl}\t{sim_apd[i]:.2f}\t{ref_apd_val:.2f}\t{apd_error:.2f}\t")
                f.write(f"{sim_dvdt[i]:.2f}\t{ref_dvdt_val:.2f}\t{dvdt_error:.2f}\n")
    
    # Salvar resultados em formato Excel para análise posterior
    results_df = pd.DataFrame({
        'BCL': sim_bcl,
        'APD90_Sim': sim_apd,
        'Max_dVdt_Sim': sim_dvdt
    })
    
    # Mesclar com dados de referência
    merged_results = pd.merge(
        results_df,
        restitution_data[['BCL', 'APD90', 'Max_dVdt']],
        on='BCL',
        suffixes=('', '_Ref')
    )
    
    # Calcular erros percentuais
    merged_results['APD90_Error(%)'] = 100 * abs(merged_results['APD90_Sim'] - merged_results['APD90']) / merged_results['APD90']
    merged_results['dVdt_Error(%)'] = 100 * abs(merged_results['Max_dVdt_Sim'] - merged_results['Max_dVdt']) / merged_results['Max_dVdt']
    
    # Salvar em Excel
    merged_results.to_excel('resultados_otimizacao_nsga2.xlsx', index=False)
    
    print("\nResultados salvos em:")
    print("- 'parametros_otimizados_nsga2.txt' (parâmetros e valores)")
    print("- 'resultados_otimizacao_nsga2.xlsx' (comparação das curvas de restituição)")
    print("- 'pareto_solutions_nsga2.xlsx' (todas as soluções do front de Pareto)")
    print("- 'curva_restituicao_otimizada_nsga2.png' (gráfico das curvas)")
    print("- 'potencial_acao_otimizado_nsga2.png' (gráfico do potencial de ação)")
    print("- 'pareto_front_nsga2.png' (visualização 3D do front de Pareto)")

if __name__ == "__main__":
    main()