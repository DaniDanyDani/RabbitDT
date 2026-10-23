"""Escolha da solução de compromisso na frente de Pareto (somente numpy)."""

import numpy as np


def mascara_validos(F, penalidade=1e3):
    """Linhas de F sem valores não finitos e sem penalidade."""
    F = np.atleast_2d(np.asarray(F, dtype=float))
    return np.all(np.isfinite(F), axis=1) & np.all(F < penalidade, axis=1)


def chave_compromisso(f, tolerancia_apd90):
    """Chave de ordenação (menor é melhor) para um par (erro_apd90, erro_forma).

    O APD90 é o objetivo principal: soluções com erro de APD90 <= tolerância vêm primeiro,
    ordenadas pelo erro da forma de onda. As demais são ordenadas pela soma dos dois erros.
    Diferente de escalar pelo mínimo/amplitude do front observado, a regra usa erros
    absolutos: um front inteiro ruim não é escolhido como se fosse bom.
    """
    erro_apd90, erro_onda = float(f[0]), float(f[1])
    if erro_apd90 <= tolerancia_apd90:
        return (0, erro_onda, erro_apd90)
    return (1, erro_apd90 + erro_onda, erro_apd90)


def ordenar_por_compromisso(F, *, tolerancia_apd90, penalidade=1e3):
    """Índices das soluções válidas, da melhor para a pior."""
    F = np.atleast_2d(np.asarray(F, dtype=float))
    validos = np.flatnonzero(mascara_validos(F, penalidade))
    ordenados = sorted(validos, key=lambda i: chave_compromisso(F[i], tolerancia_apd90))
    return np.asarray(ordenados, dtype=int)


def escolher_compromisso(F, *, tolerancia_apd90, penalidade=1e3):
    """Índice da melhor solução de compromisso (levanta RuntimeError se todas falharam)."""
    ordem = ordenar_por_compromisso(F, tolerancia_apd90=tolerancia_apd90, penalidade=penalidade)
    if ordem.size == 0:
        raise RuntimeError(
            "Todas as soluções receberam penalidade. Verifique se o modelo produz AP com o "
            "protocolo de estímulo e se os limites dos parâmetros são adequados."
        )
    return int(ordem[0])


def parametros_no_limite(valores, limites, margem=0.02):
    """Lista de (nome, 'inferior'|'superior') para parâmetros a menos de ``margem`` da faixa de um limite.

    Parâmetros colados no limite indicam que a faixa pode estar restringindo o ajuste.
    """
    encontrados = []
    for nome, (inferior, superior) in limites.items():
        faixa = superior - inferior
        if faixa <= 0:
            continue
        if valores[nome] <= inferior + margem * faixa:
            encontrados.append((nome, "inferior"))
        elif valores[nome] >= superior - margem * faixa:
            encontrados.append((nome, "superior"))
    return encontrados
