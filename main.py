"""Ponto de entrada: configura e executa a otimização celular.

Execução longa, em segundo plano:
    mkdir -p results && nohup python -u main.py > results/resultado.txt 2>&1 &

Teste rápido do pipeline completo (poucos minutos):
    python main.py --rapido

Os valores padrão estão em utils/otimizacao/configuracao.py; as opções abaixo os sobrescrevem.
"""

import argparse
import logging

from utils.otimizacao.executar import executar_otimizacao


def _ler_argumentos():
    parser = argparse.ArgumentParser(description="Otimiza parâmetros do modelo de Mahajan contra um AP experimental.")
    parser.add_argument("--populacao", type=int, help="tamanho da população do NSGA-II")
    parser.add_argument("--geracoes", type=int, help="número máximo de gerações")
    parser.add_argument("--processos", type=int, help="processos paralelos (1 = sem paralelismo)")
    parser.add_argument("--semente", type=int, help="semente aleatória")
    parser.add_argument("--pulsos", type=int, help="pulsos simulados por candidato na busca")
    parser.add_argument("--pulsos-validacao", type=int, help="pulsos na validação/refino em regime estacionário")
    parser.add_argument("--top-k", type=int, help="quantas soluções do front validar em regime estacionário")
    parser.add_argument("--sem-refino", action="store_true", help="não refinar a solução final em regime estacionário")
    parser.add_argument("--retomar", action="store_true", help="começar da população salva em results/checkpoint_populacao.npz")
    parser.add_argument("--alvo", help="arquivo do AP experimental (padrão: data/ap_mayra.txt)")
    parser.add_argument("--saida", help="pasta de resultados (padrão: results/)")
    parser.add_argument("--rapido", action="store_true", help="teste rápido: população 16, 3 gerações, validação com 20 pulsos")
    return parser.parse_args()


if __name__ == "__main__":
    args = _ler_argumentos()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    opcoes = {
        "populacao": args.populacao,
        "geracoes": args.geracoes,
        "processos": args.processos,
        "semente": args.semente,
        "num_pulsos": args.pulsos,
        "num_pulsos_validacao": args.pulsos_validacao,
        "top_k": args.top_k,
        "refinar": False if args.sem_refino else None,
        "retomar": args.retomar,
        "arquivo_alvo": args.alvo,
        "pasta_resultados": args.saida,
    }
    if args.rapido:
        opcoes.update(populacao=16, geracoes=3, num_pulsos_validacao=20, top_k=2, max_avaliacoes_refino=10)
    executar_otimizacao(**opcoes)
