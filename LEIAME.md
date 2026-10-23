# Otimização do modelo celular de Mahajan contra um AP experimental

Encontra parâmetros do modelo (condutâncias) que reproduzem o **APD90** e a **morfologia**
de um potencial de ação experimental.

## Uso
```bash
pip install -r requirements.txt
# coloque o registro em data/ap_mayra.txt (2 colunas: tempo em ms, potencial)
python -m pytest                      # testes (o teste do arquivo experimental é pulado se ele não existir)
python main.py --rapido               # teste rápido do pipeline completo
mkdir -p results && nohup python -u main.py > results/resultado.txt 2>&1 &   # execução completa
python main.py --help                 # população, gerações, processos, pulsos, --retomar, --sem-refino...
```
Valores padrão: `utils/otimizacao/configuracao.py`.

## Pipeline (`utils/otimizacao/executar.py`)
1. **NSGA-II** com poucos pulsos (`NUM_PULSOS`), 2 objetivos: erro relativo do APD90 e RMSE da forma de
   onda normalizada **e alinhada no upstroke**. Avaliação em paralelo (`NUM_PROCESSOS`), população inicial por
   hipercubo latino com o modelo padrão semeado, parada antecipada e checkpoint em `results/`.
2. **Validação em regime estacionário** das `TOP_K_VALIDACAO` melhores soluções (`NUM_PULSOS_VALIDACAO`
   pulsos, tolerâncias mais rígidas) com a deriva do APD90 do último batimento.
3. **Refino** (Nelder-Mead) da melhor solução com pacing longo, pois o APD90 de poucos pulsos difere do de regime.
4. Saídas em `results/`: `melhores_parametros_metricas.json`, `solucoes_pareto.csv`, `AP_simulado_otimizado*.txt`,
   `comparacao_ap.png/pdf`, `front_pareto.png`, `checkpoint_populacao.npz`.

## Módulos
| arquivo | papel |
|---|---|
| `utils/ap.py` | leitura, ativação (dV/dt máx.), APD90, normalização, erro de forma alinhado |
| `utils/simulacao.py` | simula N pulsos e devolve a janela do último AP (`FalhaDeSimulacao` para falhas esperadas) |
| `utils/otimizacao/avaliador.py` | objetivo (sem pymoo, testável) |
| `utils/otimizacao/problema.py` | adaptador fino para o pymoo |
| `utils/otimizacao/selecao.py` | escolha do compromisso na frente de Pareto |
| `utils/otimizacao/refino.py` | refino local em regime estacionário |

## Limitações conhecidas
* O estado inicial é o padrão do modelo; mesmo com 150 pulsos a [Na]i ainda deriva lentamente (o APD90 muda
  ~0,03 ms/batimento entre o 96º e o 120º). A `deriva_apd90_ms` do JSON mostra o quanto.
* O problema é mal condicionado (várias combinações de condutâncias dão APs parecidos; ver Gemmell et al., 2014).
  Confira `parametros_no_limite` e os fatores em relação ao padrão antes de interpretar os valores.
* O registro é comparado normalizado (0–1): amplitude e potencial de repouso em mV não entram no ajuste.
