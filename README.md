# Otimização do potencial de ação celular (Mahajan)

Projeto para ajustar parâmetros de condutâncias do modelo celular Mahajan ao
potencial de ação experimental, minimizando dois objetivos com NSGA-II:

1. erro relativo do APD90;
2. RMSE da morfologia normalizada do potencial de ação.

A implementação mantém os parâmetros não otimizados nos valores padrão do
modelo. `fator_gto` escala conjuntamente `gtof` e `gtos`, as componentes rápida
e lenta da corrente transitória de saída `Ito`.

## Ambiente

Recomenda-se Python 3.11 em ambiente virtual ou Conda.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Com Conda, ative o ambiente desejado antes de instalar os requisitos.

## Dados de entrada

O ZIP revisado não contém `data/ap_mayra.txt`. Antes de executar a otimização,
coloque o traçado experimental nesse caminho. O arquivo deve conter tempo (ms)
e potencial/amplitude em duas colunas numéricas, com o tempo estritamente
crescente. Veja `data/README.md`.

**Alinhamento temporal:** o AP simulado é recortado entre 50 ms antes do último
pulso e 250 ms depois dele; portanto, o estímulo aparece em `t=50 ms` no recorte.
A comparação da forma de onda é feita no mesmo eixo temporal e não corrige
translações automaticamente. Confirme que o traçado experimental está alinhado
com esse protocolo; ajuste `BASELINE_ALVO_MS`, `INICIO_BUSCA_PICO_ALVO_MS` e/ou
`JANELA_ANTES_ESTIMULO_MS` se os tempos do experimento forem diferentes.

## Executar

Na raiz do projeto:

```bash
python main.py
```

Para executar os testes:

```bash
python -m pytest -q
```

## Resultados

A pasta `results/` é criada automaticamente. A execução salva:

- `AP_simulado_otimizado.txt`: tempo local (ms) e potencial em mV;
- `AP_simulado_otimizado_normalizado.txt`: curva normalizada para comparar morfologia;
- `comparacao_ap.png` e `comparacao_ap.pdf`: sobreposição do experimento e do modelo;
- `front_pareto.png` e `solucoes_pareto.csv`: soluções não dominadas e objetivos;
- `melhores_parametros_metricas.json`: variáveis de decisão, parâmetros físicos, objetivos e métricas de validação.

A busca parte de oito pulsos e o compromisso selecionado é reavaliado com 64
pulsos. Em cada simulação, o projeto também guarda janelas densas dos últimos
quatro potenciais e verifica as diferenças entre pares consecutivos e entre o
primeiro e o último desses potenciais. São avaliados APD90, diferença máxima da
forma de onda normalizada, potencial de repouso e pico.

Os limites padrão de convergência são: variação relativa do APD90 <= 2%,
diferença máxima da forma normalizada <= 0,05 (5% da amplitude), variação de
repouso <= 1 mV e variação do pico <= 2 mV. São critérios numéricos ajustáveis
em `utils/otimizacao/configuracao.py`, não limites fisiológicos universais. A
checagem aparece no terminal e é gravada no JSON de métricas. Se a otimização
curta não convergir, seus objetivos ainda representam o último AP transitório:
aumente `NUM_PULSOS` e repita a busca antes de interpretar os parâmetros como
calibrados. A validação longa também precisa convergir; se nem 64 pulsos forem
suficientes, aumente `NUM_PULSOS_VALIDACAO` e confira se existe adaptação lenta
ou alternância de batimentos.

## Premissas importantes

- Os limites iniciais são ±30% ao redor dos padrões, inspirados no estudo de
  sensibilidade anexado. Eles são limites de busca, não prova de plausibilidade
  nem garantia de que o alvo será alcançável.
- A normalização remove diferenças de repouso e amplitude absoluta, mas preserva
  undershoot e overshoot. Portanto, o objetivo morfológico não calibra o nível de
  repouso nem a amplitude absoluta em mV.
- O modelo é um sistema de EDOs celular com múltiplos estados e dinâmica de cálcio;
  um único APD90 não identifica de modo único as condutâncias. Analise o front de
  Pareto e os valores físicos antes de aceitar uma solução.
- Confira unidades, protocolo de estimulação, tipo celular e definição de APD90
  antes de comparar o alvo com valores publicados. Não amplie limites apenas para
  forçar o ajuste se o candidato perder plausibilidade fisiológica.
