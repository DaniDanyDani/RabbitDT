# Revisão do projeto_inicial.zip

Tudo abaixo foi verificado no código original e, quando há número, medido por simulação (modelo padrão,
BCL = 450 ms). Veja "O que não foi verificado" no fim.

## 1. Erros de implementação (por gravidade)

**1. Formas de onda comparadas sem alinhamento temporal (crítico)** — `problema.py`, `configuracao.py`
- O simulado tem o estímulo em 50 ms (upstroke ≈ 53 ms). O alvo usa linha de base 0–25 ms e busca de pico
  a partir de 25 ms, o que indica upstroke perto de 25–30 ms. `erro_forma_onda` compara as duas curvas no mesmo
  eixo absoluto, então ~25 ms de defasagem entram no RMSE.
- Teste com alvo sintético de parâmetros conhecidos (upstroke em 28 ms): RMSE antigo = **0,213 mesmo nos parâmetros
  verdadeiros** (novo: 0,000). O RMSE antigo foi *menor* num candidato com APD mais curto (gca ×0,1: 0,203) do que no
  ponto verdadeiro: a defasagem empurra a solução para APD menor, contra o objetivo de APD90.
- Correção: `instante_ativacao` (dV/dt máx.) nos dois sinais e `erro_forma_onda_alinhada`, na janela
  [0, 2×APD90 do alvo] a partir da ativação. O código avisa se a ativação detectada cola no início da busca.

**2. 8 pulsos não são regime estacionário (alto)** — `configuracao.py`
- Modelo padrão: APD90 188,4 ms (8 pulsos) → 192,4 ms (120 pulsos), ainda subindo ~0,03 ms/batimento; [Na]i 11,41 → 10,92 mM.
- Candidatos com APD90 curto (perto do alvo): gca ×0,2: 55,9 → 52,4 ms (−6,3%); gca ×0,3 + gtof/gkr/gks ×1,5:
  47,2 → 42,9 ms (−9,2%); gNaCa ×0,1 + gNaK ×0,3: 133,2 → 119,5 ms (−10,3%).
- Efeito: um APD90 "dentro de 5%" com 8 pulsos sai 6–10% curto em regime.
- Correção: validação das `TOP_K_VALIDACAO` melhores com `NUM_PULSOS_VALIDACAO` pulsos (deriva do último
  batimento no JSON) e refino Nelder-Mead com pacing longo.

**3. Limites de busca inconsistentes (alto)** — `configuracao.py`
- `gtos` e `gNaK` têm limite superior igual ao valor padrão (`0.04`, `1.5`; faltou `*uf`); os outros 6 vão de 0,1× a 1,9×.
  O comentário diz "±30%" e outro diz "apenas gca e gtof", mas são 8 parâmetros.
- Correção: limites calculados de fatores sobre o padrão do próprio modelo (`FATOR_MIN_PADRAO`, `FATOR_MAX_PADRAO`,
  `FATORES_LIMITES`); teste de regressão; preset `FATORES_LITERATURA` (±30%, Gemmell et al., 2014).

**4. Critério de AP aceita resposta passiva (médio)** — `problema.py`
- `amplitude_minima_sim_mV = 10`. Sem INa (gna = 0) a resposta passiva ao estímulo tem 41 mV de amplitude e seria
  tratada como AP. Novo mínimo: 60 mV (um AP real tem ~100–120 mV).

**5. Penalidades e tratamento de exceções (médio)** — `problema.py`
- APD90 = NaN (não repolariza em 250 ms) virava `ValueError` e penalidade fixa 1e3, sem gradiente. Agora o APD90 é
  tratado como limite inferior (censurado) e soma 1,0 ao erro.
- `except (ValueError, RuntimeError, …)` genérico transforma qualquer erro de programação que levante essas exceções
  em penalidade silenciosa (uma otimização inteira de 1e3 sem aviso). Agora só `FalhaDeSimulacao` e
  `CandidatoInvalido` viram penalidade.

**6. Escolha do compromisso (médio)** — `executar.py::_escolher_compromisso`
- Cada objetivo é reescalado pelo mínimo e pela amplitude do front observado: um front inteiro ruim (ex.: todos com
  80% de erro) produz um "ótimo" aparente, e diferenças minúsculas são ampliadas. Agora: tolerância absoluta em
  APD90 (`TOLERANCIA_APD90_RELATIVA`, 5%) e, dentro dela, menor RMSE de forma.

**7. Custo e execução (médio)** — `configuracao.py`, `executar.py`
- 1024 × 500 = 512 mil avaliações (o comentário diz "conservador"), sem paralelismo, sem parada antecipada (o exemplo
  usava `DefaultMultiObjectiveTermination`), sem checkpoint, população inicial sem o modelo padrão.
- Agora: 128 × 200 por padrão, `multiprocessing`, parada por `ftol`, checkpoint e `--retomar`, LHS com o padrão semeado.

**8. Infraestrutura (baixo)**
- `data/ap_mayra.txt` não veio no zip: o teste falhava com `FileNotFoundError`, e o valor 66,79 ms é fixo. Agora o
  teste do arquivo real é pulado se ele não existir; as demais métricas têm testes com sinais sintéticos.
- `pytest` simples não encontrava `utils` (sem `pythonpath`/`conftest`). Corrigido.
- `nohup ... > results/resultado.txt` falha se `results/` não existir (ela só era criada dentro do Python).
- Docstrings: "VODE/BDF" (o solver é LSODA) e "reduza max_step para não pular o pulso" (a integração já para nas
  bordas do estímulo).

**9. APD90 medido a partir do pico (baixo)** — `ap.py`
- Gemmell et al. medem a partir do dV/dt máximo. No modelo a diferença é ~0,5 ms (188,4 × 188,9 ms), mas depende do
  tempo de subida do registro. Agora configurável (`REFERENCIA_APD`, padrão `"ativacao"`); o log mostra as duas.

## 2. O que já estava correto
- Tolerâncias da busca (rtol = 1e-4, max_step = 2 ms): APD90 188,396 ms contra 188,400 ms com rtol = 1e-8 (0,004 ms).
- Os padrões dos 8 parâmetros conferem com as Tabelas 3, 4 e 6 do artigo de Mahajan et al.
- A integração é segmentada nas bordas dos pulsos de estímulo.

## 3. Pontos de melhoria
**Implementados:** alinhamento no upstroke; APD90 pela ativação; validação e refino em regime; limites por fatores;
paralelismo, parada antecipada e checkpoint; seleção por tolerância absoluta; CLI (`--rapido`, `--retomar`...);
avisos sobre janelas do alvo e parâmetros no limite; testes.

**Sugeridos (não implementados):**
- *Sensibilidade antes de otimizar 8 parâmetros.* Efeito isolado sobre o APD90 (8 pulsos; padrão 188 ms), em 0,1× e 1,9×:
  gca 51–211 ms; gtof 209–87; gNaCa 102–200; gNaK 215–164; gkr 206–174; gks 240–170; gkix (0,1× não repolariza)–175;
  gtos 194–182 (quase sem efeito: pouco identificável).
- *Identificabilidade.* Recuperação com alvo sintético e 2 parâmetros livres (verdade: gca ×0,40, gtof ×1,30): o ajuste
  chegou a APD90 exato e RMSE de forma 0,0013 com gca ×0,50 e gtof ×1,46. AP praticamente igual, parâmetros diferentes
  (efeito discutido por Gemmell et al.). Reporte faixas (soluções quase ótimas) em vez de um único vetor, e considere
  fixar parâmetros pouco sensíveis ou penalizar a distância ao padrão.
- *Alvo versus modelo.* O APD90 do alvo (66,8 ms, a partir do pico) é ~2,8× menor que o do modelo padrão em
  BCL 450 ms e está bem abaixo da faixa de epicárdio de coelho do artigo (142–185 ms em CL 400, Tabela 1 de Gemmell).
  gca ×0,2 sozinho já leva o APD90 a ~56 ms; o alvo exige reduções muito grandes de ICaL ou combinações extremas.
  Vale confirmar espécie, temperatura e condição do registro.
- *Mais de um BCL.* O exemplo original usava restituição (vários BCL); Gemmell usa 400/600/1000 ms. Restringe muito mais.
- *Registro ruidoso:* `METODO_ATIVACAO = "meia_amplitude"`.
- *Modelagem:* `gca` escala só a corrente/fluxo sarcolemal; o gatilho do RyR (`gryr`, `gdyad`) usa Po·iCa e não
  acompanha. Se a hipótese for "menos canais LTCC", `pca` escalaria os dois.

## 4. O que não foi verificado
- O ambiente de teste não tinha `numba`, `pymoo` nem rede. O NSGA-II real **não foi executado**: a parte do pymoo segue
  a API 0.6, e o pipeline completo (seleção, validação, refino, arquivos, gráficos, `Pool` com 2 processos) foi
  testado com um pymoo falso, só para o encadeamento. Rode `python -m pytest` e `python main.py --rapido`.
- Sem o `ap_mayra.txt`, toda a validação usou um alvo sintético gerado pelo próprio modelo.
- Sem numba, um batimento leva ~0,4 s; não medi o tempo com numba.
