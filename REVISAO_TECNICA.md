# Revisão técnica do projeto de otimização celular

## Resumo executivo

A estrutura de módulos (`modelo_celular.py`, `simulacao.py`, `ap.py` e `otimizacao/`) é uma boa base para separar modelo, protocolo, métricas e algoritmo. Os maiores problemas encontrados estavam na consistência científica da métrica e dos parâmetros, no orçamento de busca e na ausência do dado experimental necessário para executar o fluxo completo.

## Erros e riscos encontrados

| Prioridade | Problema no ZIP original | Efeito potencial | Ajuste feito |
|---|---|---|---|
| Crítica | Não havia `data/ap_mayra.txt`, embora a otimização e o teste dependessem dele. | O teste falhava e `main.py` não podia fazer a calibração. | Documentei o formato esperado; a execução agora apresenta uma mensagem de erro orientando onde colocar o dado. Não inventei um substituto sintético. |
| Alta | O comentário indicava otimização de `gca` e `gtof`, mas a lista real variava oito parâmetros. | O experimento computacional não correspondia ao que os comentários descreviam; a busca tinha dimensionalidade e custo maiores. | A primeira versão agora tem seis variáveis de decisão alinhadas às seis famílias de condutância exploradas por Gemmell et al. (2014). |
| Alta | `uf=1.9` e `df=0.1` implementavam 10%–190% do valor nominal, embora o comentário dissesse ±30%. Havia ainda limites especiais incoerentes com esse comentário. | A região explorada era diferente da documentada e poderia levar a candidatos difíceis de justificar fisiologicamente. | Os limites iniciais são explicitamente ±30%. São limites de partida, não garantia de que o alvo será alcançável. |
| Alta | `gtof` e `gtos` eram genes de decisão independentes, e `gNaCa` também era otimizado. | O espaço de busca não correspondia à abordagem de seis condutâncias do estudo de sensibilidade anexado. | `fator_gto` escala `gtof` e `gtos` juntos; `gNaCa` permanece no padrão. Se a hipótese experimental exigir estimar `gNaCa`, deve-se adicioná-lo de forma explícita e justificar a dimensão extra. |
| Alta | O APD90 era contado do ápice do potencial até o cruzamento de 90% de repolarização. | O valor não seguia a definição do estudo anexado, que mede do instante de `dVm/dtmax` até 90% de repolarização. | A função `calcular_apd90` agora usa `dV/dt` máximo como início e interpolação linear no cruzamento descendente. Valores antigos de APD90 podem mudar ligeiramente e precisam ser recalculados. |
| Alta | A normalização aplicava `clip(..., 0, 1)`. | Hiperpolarização abaixo da linha de base e valores acima do pico eram removidos, ocultando erros de morfologia. | A normalização mantém esses valores; apenas desloca a linha de base para zero e escala o pico para um. |
| Alta | O protocolo utilizava oito pulsos sem medir convergência entre batimentos. | O último AP pode depender da história de estimulação; uma diferença grande entre 8 e 64 pulsos indica que o objetivo da otimização não está estável. | A simulação guarda os últimos quatro APs, compara pares consecutivos e deriva acumulada, e compara diretamente o AP final do protocolo curto e longo. O JSON salva todos os diagnósticos. |
| Alta | O orçamento configurado era 1024 indivíduos × 500 gerações. | Pode exigir centenas de milhares de simulações caras. | O padrão passou para 64 indivíduos × 40 gerações; população, gerações e semente continuam configuráveis. |
| Média | O retorno do `pymoo` era indexado diretamente, assumindo sempre arrays 2-D com mais de uma solução. | Um front com uma única solução podia gerar indexação incorreta em fluxos específicos. | Normalização explícita de `X` e `F` para arrays 2-D antes de selecionar e salvar soluções. |
| Média | A classe de problema penalizava candidatos inválidos sem contabilizar a razão. | Ficava difícil descobrir se limites/protocolo faziam o modelo falhar. | Contagem de avaliações inválidas e dos motivos mais frequentes, gravadas no JSON. |
| Média | Não havia instruções de instalação, formato do alvo ou alinhamento temporal documentado. | Execução e interpretação dos resultados dependiam de conhecimento implícito. | Adicionei README, arquivo de requisitos, instruções para dados, saídas esperadas e teste automatizado. |
| Baixa | O ZIP incluía bytecode/cache do Python e arquivos de cache do Numba. | Artefatos locais podiam poluir o projeto e ser incompatíveis com outra máquina/versão. | Os caches foram removidos do pacote e adicionados ao `.gitignore`. |

## Relação com os arquivos de referência

### `otimizacao_exemplo.py`

O script original usa um modelo fenomenológico de quatro estados e parâmetros de limiar/constantes de tempo. O projeto revisado usa o modelo detalhado Mahajan (`modelo_celular.py`) com dezenas de estados e condutâncias iônicas. Portanto, o padrão de arquitetura multiobjetivo é reutilizável, mas os parâmetros, equações, limites e custo de avaliação do script original não devem ser copiados diretamente para o modelo Mahajan.

### `artigo_original_mahajan.pdf`

Os valores padrão codificados para as condutâncias relevantes são compatíveis com a tabela de condutâncias do artigo (por exemplo, `gca=182`, `gtof=0.11`, `gtos=0.04`, `gkix=0.3`, `gkr=0.0125`, `gks=0.1386` e `gNaK=1.5`, nas unidades do modelo CellML). Esta revisão preserva os valores padrão da implementação e só altera as variáveis escolhidas pelo algoritmo.

### `artigo_parametros_mahajan.pdf`

Gemmell et al. (2014), nas seções de métodos e Tabela 1, descrevem seis condutâncias (`gto`, `gKr`, `gKs`, `gK1`, `gCa,L` e `gNaK`) e um estudo com variações de 0%, ±15% e ±30%. O mesmo artigo define APD90 entre `dVm/dtmax` e a repolarização de 90%, e verifica estado estacionário comparando APs consecutivos ponto a ponto; os autores continuaram simulações quando necessário, chegando a usar simulações de até 1000 s. Isso motivou o vínculo das duas componentes `gtof`/`gtos`, o limite inicial ±30% e a correção do início do APD90.

A Tabela 1 desse artigo reporta, para APs epicárdicos de coelho, APD90 de 142–185 ms em CL=400 ms e 160–220 ms em CL=600 ms. O teste do ZIP original esperava aproximadamente 66,79 ms do arquivo experimental, mas o arquivo real não veio no ZIP e esse número não pôde ser confirmado nesta revisão. Se ~66,79 ms for de fato o APD90 do experimento, ele é bem menor que essas faixas publicadas; antes de ampliar limites ou aceitar parâmetros extremos, vale confirmar espécie/tipo celular, BCL/protocolo, unidades, alinhamento e a definição de APD usada.

## Medições de sanidade realizadas

- `python -m compileall -q .`: passou.
- `python -m pytest -q`: **9 passaram; 2 foram ignorados** neste ambiente. Os ignorados são o teste que depende de `pymoo` (pacote não instalado neste runtime) e o teste opcional que depende do arquivo experimental ausente.
- Simulação do modelo padrão: com o protocolo da configuração, a janela retornou 1201 pontos de 0 a 300 ms e um AP válido.
- Comparação exploratória do modelo padrão, usando esta implementação da métrica: aproximadamente 188,90 ms após 8 pulsos e 191,10 ms após 64 pulsos (diferença de ~2,21 ms). Isso demonstra sensibilidade ao histórico no caso padrão; não é uma validação biológica nem uma prova de convergência.
- A avaliação de candidato foi exercitada isoladamente com uma classe-base simulada porque `pymoo` não está instalado no runtime. A execução integral do NSGA-II não foi possível neste ambiente.

## Próximos passos antes de interpretar a calibração

1. Adicionar o arquivo real `data/ap_mayra.txt` e conferir visualmente se a despolarização experimental está alinhada com o estímulo simulado em t=50 ms do recorte.
2. Instalar `requirements.txt` no ambiente de trabalho e executar uma otimização de orçamento pequeno primeiro.
3. Analisar o front de Pareto, não apenas o compromisso único; comparar os parâmetros com os padrões e registrar se o candidato encosta nos limites.
4. Executar a checagem implementada. Se os protocolos de 8 e 64 pulsos não forem consistentes, aumentar `NUM_PULSOS` e repetir a otimização; em casos de adaptação lenta, testar mais pontos de controle ou inspecionar os estados iônicos.
5. Se o alvo continuar distante usando ±30%, realizar uma análise de sensibilidade e ampliar os limites de forma controlada. Não interpretar a redução do objetivo, por si só, como evidência de plausibilidade fisiológica.

## Atualização: diagnóstico de convergência por batimento

A versão atual coleta janelas densas dos últimos quatro potenciais da simulação e compara cada par consecutivo, além de medir a deriva entre o primeiro e o último dos quatro. As métricas incluem APD90, RMSE e diferença máxima da forma de onda normalizada, linha de base e pico.

Também há comparação entre o AP final da otimização curta e o AP final da validação longa. Essa comparação é importante porque uma sequência pode parecer localmente estável nos últimos batimentos e ainda apresentar adaptação lenta ao longo de dezenas de pulsos. Portanto, o estado final só é marcado como `convergencia_verificada` quando as verificações locais da otimização e validação e a comparação curta-versus-longa passam.

Limites padrão em `utils/otimizacao/configuracao.py`: variação relativa de APD90 <= 2%; diferença máxima normalizada <= 0,05; linha de base <= 1 mV; pico <= 2 mV. Os limites são heurísticas numéricas configuráveis. O critério de 5% na forma ponto a ponto foi inspirado no método de verificação de estado estacionário do artigo de sensibilidade anexado; os outros limites são escolhas do projeto, não regras universais.

Quando a verificação falha, o código imprime o motivo e grava as comparações em `results/melhores_parametros_metricas.json`. A otimização ainda não rejeita automaticamente candidatos não convergidos; o relatório classifica a calibração como não comprovada e recomenda aumentar `NUM_PULSOS` e repetir a busca. Isso evita interromper a busca silenciosamente, mas exige que o usuário respeite o aviso antes de aceitar o resultado.

Validação da implementação: `pytest -q` resultou em 9 testes aprovados e 2 ignorados neste ambiente. A simulação exploratória com parâmetros padrão produziu APD90 de 188,898 ms com 8 pulsos e 191,104 ms com 64 pulsos; a diferença relativa ao protocolo curto foi de 1,17%, dentro dos limites configurados. Esse ensaio valida o fluxo de checagem, não substitui a verificação do conjunto de parâmetros que gerou o resultado de 86,424 ms versus 118,385 ms nem valida o arquivo experimental ausente.

