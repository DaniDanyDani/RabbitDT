import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# Configurações de Estilo IEEE (Padrões Científicos)
# ---------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,             # Padrão IEEE: 8-10 pt para figuras
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'savefig.bbox': 'tight',
    'figure.dpi': 300            # Padrão IEEE: alta resolução (300+ DPI)
})

def normalize_by_peak(source_data, target_data):
    """Mantém a linha de base em zero e iguala apenas a amplitude de pico absoluta."""
    s_max_abs = np.max(np.abs(source_data))
    t_max_abs = np.max(np.abs(target_data))
    
    if s_max_abs == 0:
        return source_data
        
    return source_data * (t_max_abs / s_max_abs)

def calc_metrics(target_aligned, source_norm):
    """Calcula o NRMSE (%) e o Coeficiente de Correlação de Pearson."""
    rmse = np.sqrt(np.mean((target_aligned - source_norm) ** 2))
    t_min, t_max = np.min(target_aligned), np.max(target_aligned)
    t_range = t_max - t_min
    
    nrmse_pct = (rmse / t_range) * 100 if t_range != 0 else 0.0
    
    if np.std(target_aligned) == 0 or np.std(source_norm) == 0:
        pearson_r = 0.0
    else:
        pearson_r = np.corrcoef(target_aligned, source_norm)[0, 1]
        
    return nrmse_pct, pearson_r

def CompareAllECGs(InferencePath, ClinicalPath, SimulatedPath, OutputImagePath='comparacao_ecg.pdf'):
    # Carregamento de dados
    try:
        df_Inference = pd.read_csv(InferencePath)
        df_Clinical = pd.read_csv(ClinicalPath, header=None)
        df_Simulated = pd.read_csv(SimulatedPath, sep=r'\s+', header=None) # Arquivo txt do Monoalg
    except FileNotFoundError as e:
        print(f"Erro ao carregar arquivos: {e}")
        return
    
    # --- FILTRAGEM DOS DADOS SIMULADOS ---
    df_Simulated_filtered = df_Simulated[(df_Simulated[0] >= 3600) & (df_Simulated[0] <= 3600+300)]
    
    # Simulação possui 5 eletrodos ativos (Col 2 a 6. Col 0 = tempo, Col 1 = Ref)
    num_sim_leads = df_Simulated_filtered.shape[1] - 2 

    num_leads = min(
        df_Inference.shape[1], 
        df_Clinical.shape[0], 
        num_sim_leads
    )
    
    inf_lead_names = df_Inference.columns
    
    # Configuração de Layout para IEEE: 
    # Largura de 3.5 polegadas por gráfico garante boa proporção em colunas de artigos
    fig, axes = plt.subplots(1, num_leads, figsize=(3.5 * num_leads, 3.0))
    if num_leads == 1: axes = [axes]

    for i in range(num_leads):
        inf_raw = df_Inference.iloc[:, i].dropna().values
        clin_raw = df_Clinical.iloc[i, :].dropna().values
        
        # Subtração do eletrodo simulado com downsampling [::5]
        sim_ref = df_Simulated_filtered.iloc[::5, 1].dropna().values
        sim_electrode = df_Simulated_filtered.iloc[::5, i + 2].dropna().values
        
        min_len_sim = min(len(sim_electrode), len(sim_ref))
        sim_raw = sim_electrode[:min_len_sim] - sim_ref[:min_len_sim]

        # Alinhamento pelo mínimo global
        idx_min_clin = np.argmin(clin_raw)
        idx_min_sim = np.argmin(sim_raw)
        shift = idx_min_clin - idx_min_sim

        if shift > 0:
            sim_shifted = np.pad(sim_raw, (shift, 0), mode='constant', constant_values=0)[:len(sim_raw)]
        else:
            sim_shifted = np.pad(sim_raw, (0, abs(shift)), mode='constant', constant_values=0)[abs(shift):]

        # Recorte pelo menor array
        min_len = min(len(inf_raw), len(clin_raw), len(sim_shifted))
        
        inf_aligned = inf_raw[:min_len]
        clin_aligned = clin_raw[:min_len]
        sim_aligned = sim_shifted[:min_len] - sim_shifted[0]

        # Normalização com base no pico do dado clínico (alvo)
        inf_norm = normalize_by_peak(inf_aligned, clin_aligned)
        sim_norm = normalize_by_peak(sim_aligned, clin_aligned)

        # Cálculo de métricas
        nrmse_inf, r_inf = calc_metrics(clin_aligned, inf_norm)
        nrmse_sim, r_sim = calc_metrics(clin_aligned, sim_norm)

        # -----------------
        # Configuração Gráfica (IEEE Plotting Setup)
        # -----------------
        ax = axes[i]
        time_axis = np.linspace(0, min_len - 1, min_len)
        
        # 1. Plotagem das formas de onda
        linha_clinica, = ax.plot(time_axis, clin_aligned, label='Experimental (Ref.)', color='black', linewidth=1.5)
        linha_inferencia, = ax.plot(time_axis, inf_norm, label='Inferência', color='#d62728', linestyle='--', linewidth=1.5)
        linha_mono, = ax.plot(time_axis, sim_norm, label='Monodomínio', color='#1f77b4', linestyle=':', linewidth=1.5)
        
        # 2. Títulos e eixos
        ax.set_title(f'Eletrodo: {inf_lead_names[i]}', pad=8, fontweight='bold')
        ax.set_xlabel('Tempo (ms)', labelpad=4)
        
        if i == 0:
            ax.set_ylabel('Potencial (mV)', labelpad=4)
        
        # 3. Métricas estatísticas compactas
        # Abreviei os nomes para caberem em uma linha perfeitamente no centro
        metric_text = (
            f"Inf.: NRMSE {nrmse_inf:.1f}% | $r$: {r_inf:.2f}\n"
            f"Mono.: NRMSE {nrmse_sim:.1f}% | $r$: {r_sim:.2f}"
        )
        
        # OTIMIZAÇÃO DE ESPAÇO: Pegar limites reais das ondas do ECG
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        # Expande 10% do teto (para o pico R não bater) 
        # Expande 30% o fundo, criando uma área em branco artificial para o texto
        ax.set_ylim(y_min - y_range * 0.30, y_max + y_range * 0.10)
        
        # Coloca a caixa de texto bem no centro desse espaço vazio inferior
        ax.text(0.5, 0.02, metric_text, transform=ax.transAxes, fontsize=7.5,
                horizontalalignment='center', verticalalignment='bottom', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=0.5))
        
        # 4. Estética limpa e grade
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.5)

    # 5. OTIMIZAÇÃO DE LEGENDA: Legenda ÚNICA no topo da figura (fora dos gráficos)
    handles = [linha_clinica, linha_inferencia, linha_mono]
    labels = ['Experimental (Ref.)', 'Inferência', 'Monodomínio']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), 
               ncol=3, frameon=False, fontsize=9, columnspacing=1.5)

    # Ajusta o espaçamento entre os subplots para não ficarem grudados
    plt.tight_layout()
    
    # Gerenciamento dinâmico de extensão de arquivo (.pdf e .png)
    if OutputImagePath.endswith('.pdf'):
        pdf_path = OutputImagePath
        png_path = OutputImagePath.replace('.pdf', '.png')
    else:
        png_path = OutputImagePath
        pdf_path = OutputImagePath.replace('.png', '.pdf')
        
    #bbox_inches='tight' garante que a legenda externa não seja cortada ao salvar
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print("Figuras padrão IEEE salvas com sucesso:")
    print(f" - Formato Raster (PNG): {png_path}")
    print(f" - Formato Vetorial (PDF) - Recomendado: {pdf_path}")

if __name__ == "__main__":
    CompareAllECGs(
        InferencePath='results/INFERENCIA/ecg/last_ecg2.csv', 
        ClinicalPath='results/EXPERIMENTAL/ecg/rabbit_clinical_qrs_ecg.csv', 
        SimulatedPath='results/MONOALG/ecg/ecg2.txt'
    )