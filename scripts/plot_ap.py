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
    'legend.fontsize': 8,
    'savefig.bbox': 'tight',
    'figure.dpi': 300            # Padrão IEEE: alta resolução (300+ DPI)
})

def plot_epicardial_action_potential(data_path, output_path='potencial_epicardico.pdf'):
    """
    Lê um arquivo txt e plota o Sinal do Potencial de Ação Epicárdica
    nos padrões IEEE.
    """
    # 1. Carregamento dos dados
    # Assume-se que o TXT não possui cabeçalho e é separado por espaços/tabs
    try:
        df = pd.read_csv(data_path, sep=r'\s+', header=None)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{data_path}' não foi encontrado.")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return

    # Extraindo as colunas
    tempo_ms = df.iloc[:, 0].dropna().values
    potencial = df.iloc[:, 1].dropna().values

    # 2. Configuração de Layout para IEEE
    # Uma coluna padrão IEEE tem cerca de 3.5 polegadas (8.89 cm) de largura
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # 3. Plotagem da forma de onda
    ax.plot(tempo_ms, potencial, label='Sinal Epicárdico', color='black', linewidth=1.5)

    # 4. Títulos e eixos (em Português)
    ax.set_title('Potencial de Ação da Superfície Epicárdica', pad=8, fontweight='bold')
    ax.set_xlabel('Tempo (ms)', labelpad=4)
    ax.set_ylabel('Amplitude (mV)', labelpad=4) # Altere 'mV' se a unidade for diferente (ex: V, u.a.)
    
    # 5. Estética limpa e grade (Padrão IEEE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Grade suave para facilitar a leitura dos valores sem poluir o gráfico
    ax.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
    
    # Legenda limpa e sem bordas pesadas
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='none', loc='upper right')

    plt.tight_layout()
    
    # 6. Gerenciamento dinâmico de extensão de arquivo (.pdf e .png)
    if output_path.endswith('.pdf'):
        pdf_path = output_path
        png_path = output_path.replace('.pdf', '.png')
    else:
        png_path = output_path
        pdf_path = output_path.replace('.png', '.pdf')
        
    # Salvando a figura
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, dpi=300)
    
    print("Figuras padrão IEEE salvas com sucesso:")
    print(f" - Formato Raster (PNG): {png_path}")
    print(f" - Formato Vetorial (PDF) - Recomendado: {pdf_path}")

if __name__ == "__main__":
    # Exemplo de uso: substitua o 'caminho_para_o_arquivo.txt' pelo nome real do seu arquivo
    caminho_txt = '/home/daniel.leme/RabbitDT/results/EXPERIMENTAL/ap_mayra.txt'
    
    # Criando um arquivo TXT de exemplo automaticamente (apenas para teste inicial)
    if not os.path.exists(caminho_txt):
        print("Criando arquivo de exemplo para demonstração...")
        t = np.linspace(0, 300, 1000)
        # Geração de um pulso simples parecendo um potencial de ação genérico
        sinal = -85 + 115 * np.exp(-((t - 20) / 10)**2) - 10 * np.exp(-((t - 150) / 50)**2)
        pd.DataFrame({0: t, 1: sinal}).to_csv(caminho_txt, sep='\t', header=False, index=False)

    # Chamada da função
    plot_epicardial_action_potential(
        data_path=caminho_txt, 
        output_path='figura_potencial_epicardico.pdf'
    )