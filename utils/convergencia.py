"""Diagnóstico de convergência entre potenciais de ação consecutivos."""

import numpy as np

from .ap import calcular_apd90, normalizar_ap


def diagnosticar_convergencia_batimentos(
    batimentos,
    *,
    janela_linha_base_ms=(0.0, 50.0),
    inicio_busca_pico_ms=50.0,
    tolerancia_apd_relativa=0.02,
    tolerancia_max_delta_forma=0.05,
    tolerancia_linha_base_mV=1.0,
    tolerancia_pico_mV=2.0,
):
    """Compara os últimos batimentos e retorna métricas e estado de convergência.

    A forma normalizada é comparada ponto a ponto. O limite de 0,05 equivale
    a 5% da amplitude de referência, próximo do critério de comparação entre
    os dois últimos potenciais usado no artigo de sensibilidade anexado.
    Os limites de APD, repouso e pico são critérios numéricos configuráveis do
    projeto, e não limiares universais de fisiologia.
    """
    criterios = {
        "tolerancia_apd_relativa": float(tolerancia_apd_relativa),
        "tolerancia_max_delta_forma_normalizada": float(tolerancia_max_delta_forma),
        "tolerancia_delta_linha_base_mV": float(tolerancia_linha_base_mV),
        "tolerancia_delta_pico_mV": float(tolerancia_pico_mV),
    }
    if len(batimentos) < 2:
        return {
            "convergiu": False,
            "motivo": "São necessários ao menos dois batimentos detalhados.",
            "criterios": criterios,
            "num_batimentos_avaliados": len(batimentos),
            "comparacoes_consecutivas": [],
            "comparacao_primeiro_ultimo": None,
        }

    medidas = []
    for batimento in batimentos:
        t = np.asarray(batimento["tempo_ms"], dtype=float)
        v = np.asarray(batimento["potencial_mV"], dtype=float)
        baseline_mask = (t >= janela_linha_base_ms[0]) & (t < janela_linha_base_ms[1])
        pico_mask = t >= inicio_busca_pico_ms
        if not np.any(baseline_mask) or not np.any(pico_mask):
            raise ValueError("A janela de convergência não cobre repouso e pico do potencial.")

        baseline = float(np.median(v[baseline_mask]))
        pico = float(np.max(v[pico_mask]))
        apd90 = float(calcular_apd90(
            t, v,
            janela_linha_base_ms=janela_linha_base_ms,
            inicio_busca_pico_ms=inicio_busca_pico_ms,
        ))
        if not np.isfinite(apd90) or apd90 <= 0:
            ap_norm = np.full_like(v, np.nan, dtype=float)
        else:
            ap_norm, _, _ = normalizar_ap(
                t, v,
                janela_linha_base_ms=janela_linha_base_ms,
                inicio_busca_pico_ms=inicio_busca_pico_ms,
            )
        medidas.append({
            "numero_pulso": int(batimento.get("numero_pulso", len(medidas) + 1)),
            "tempo_ms": t,
            "potencial_mV": v,
            "potencial_normalizado": ap_norm,
            "apd90_ms": apd90,
            "linha_base_mV": baseline,
            "pico_mV": pico,
        })

    def comparar(a, b):
        apd_a, apd_b = a["apd90_ms"], b["apd90_ms"]
        apds_validos = bool(np.isfinite(apd_a) and np.isfinite(apd_b))
        if apds_validos:
            delta_apd = abs(apd_b - apd_a)
            denom_apd = max(abs(apd_a), abs(apd_b), 1e-12)
            delta_apd_rel = delta_apd / denom_apd
        else:
            delta_apd = None
            delta_apd_rel = None

        ta = a["tempo_ms"]
        tb = b["tempo_ms"]
        va = a["potencial_normalizado"]
        vb = b["potencial_normalizado"]
        if not np.all(np.isfinite(va)) or not np.all(np.isfinite(vb)):
            rmse_forma = None
            max_delta_forma = None
        else:
            inicio = max(float(ta[0]), float(tb[0]))
            fim = min(float(ta[-1]), float(tb[-1]))
            mascara = (ta >= inicio) & (ta <= fim)
            if np.count_nonzero(mascara) < 3:
                rmse_forma = None
                max_delta_forma = None
            else:
                delta = va[mascara] - np.interp(ta[mascara], tb, vb)
                rmse_forma = float(np.sqrt(np.mean(delta ** 2)))
                max_delta_forma = float(np.max(np.abs(delta)))

        delta_baseline = abs(b["linha_base_mV"] - a["linha_base_mV"])
        delta_pico = abs(b["pico_mV"] - a["pico_mV"])
        checks = {
            "apd90": bool(apds_validos and delta_apd_rel <= tolerancia_apd_relativa),
            "forma": bool(max_delta_forma is not None and max_delta_forma <= tolerancia_max_delta_forma),
            "linha_base": bool(delta_baseline <= tolerancia_linha_base_mV),
            "pico": bool(delta_pico <= tolerancia_pico_mV),
        }
        return {
            "pulso_anterior": a["numero_pulso"],
            "pulso_atual": b["numero_pulso"],
            "apd90_anterior_ms": float(apd_a) if np.isfinite(apd_a) else None,
            "apd90_atual_ms": float(apd_b) if np.isfinite(apd_b) else None,
            "delta_apd90_ms": float(delta_apd) if delta_apd is not None else None,
            "delta_apd90_relativo": float(delta_apd_rel) if delta_apd_rel is not None else None,
            "rmse_forma_normalizada": float(rmse_forma) if rmse_forma is not None else None,
            "delta_maximo_forma_normalizada": float(max_delta_forma) if max_delta_forma is not None else None,
            "delta_linha_base_mV": float(delta_baseline),
            "delta_pico_mV": float(delta_pico),
            "criterios_atendidos": checks,
            "convergiu": bool(all(checks.values())),
        }

    comparacoes = [comparar(a, b) for a, b in zip(medidas[:-1], medidas[1:])]
    acumulada = comparar(medidas[0], medidas[-1])
    convergiu = bool(all(c["convergiu"] for c in comparacoes) and acumulada["convergiu"])
    motivos = []
    nomes = {
        "apd90": "variação relativa do APD90 acima do limite",
        "forma": "diferença ponto a ponto da forma acima do limite",
        "linha_base": "variação da linha de base acima do limite",
        "pico": "variação do pico acima do limite",
    }
    for comp in comparacoes + [acumulada]:
        for nome, atendido in comp["criterios_atendidos"].items():
            if not atendido and nomes[nome] not in motivos:
                motivos.append(nomes[nome])

    # Só valores escalares serializáveis são expostos. As curvas internas são
    # removidas para que o resultado possa ser salvo diretamente em JSON.
    return {
        "convergiu": convergiu,
        "motivo": "Convergência detectada." if convergiu else "; ".join(motivos),
        "criterios": criterios,
        "num_batimentos_avaliados": len(medidas),
        "apd90s_ms": [float(m["apd90_ms"]) if np.isfinite(m["apd90_ms"]) else None for m in medidas],
        "pulsos_avaliados": [int(m["numero_pulso"]) for m in medidas],
        "comparacoes_consecutivas": comparacoes,
        "comparacao_primeiro_ultimo": acumulada,
    }


def comparar_protocolos(
    dados_curtos,
    dados_longos,
    *,
    janela_linha_base_ms=(0.0, 50.0),
    inicio_busca_pico_ms=50.0,
    tolerancia_apd_relativa=0.02,
    tolerancia_max_delta_forma=0.05,
    tolerancia_linha_base_mV=1.0,
    tolerancia_pico_mV=2.0,
):
    """Compara os APs finais de protocolos com números de pulsos diferentes.

    Esta comparação complementar detecta deriva lenta que pode passar numa
    checagem apenas entre os últimos batimentos de cada execução.
    ``dados_*`` precisam conter tempo_ms, potencial_mV e apd90_ms.
    """
    t_curto = np.asarray(dados_curtos["tempo_ms"], dtype=float)
    v_curto = np.asarray(dados_curtos["potencial_mV"], dtype=float)
    t_longo = np.asarray(dados_longos["tempo_ms"], dtype=float)
    v_longo = np.asarray(dados_longos["potencial_mV"], dtype=float)
    apd_curto = float(dados_curtos["apd90_ms"])
    apd_longo = float(dados_longos["apd90_ms"])

    delta_apd = abs(apd_longo - apd_curto) if np.isfinite(apd_curto) and np.isfinite(apd_longo) else None
    delta_apd_rel = delta_apd / max(abs(apd_curto), 1e-12) if delta_apd is not None else None

    norm_curto, base_curto, pico_curto = normalizar_ap(
        t_curto, v_curto,
        janela_linha_base_ms=janela_linha_base_ms,
        inicio_busca_pico_ms=inicio_busca_pico_ms,
    )
    norm_longo, base_longo, pico_longo = normalizar_ap(
        t_longo, v_longo,
        janela_linha_base_ms=janela_linha_base_ms,
        inicio_busca_pico_ms=inicio_busca_pico_ms,
    )
    inicio = max(float(t_curto[0]), float(t_longo[0]))
    fim = min(float(t_curto[-1]), float(t_longo[-1]))
    mascara = (t_curto >= inicio) & (t_curto <= fim)
    if np.count_nonzero(mascara) < 3:
        rmse_forma = None
        max_delta_forma = None
    else:
        delta = norm_curto[mascara] - np.interp(t_curto[mascara], t_longo, norm_longo)
        rmse_forma = float(np.sqrt(np.mean(delta ** 2)))
        max_delta_forma = float(np.max(np.abs(delta)))

    delta_base = abs(float(base_longo) - float(base_curto))
    delta_pico = abs(float(pico_longo) - float(pico_curto))
    checks = {
        "apd90": bool(delta_apd_rel is not None and delta_apd_rel <= tolerancia_apd_relativa),
        "forma": bool(max_delta_forma is not None and max_delta_forma <= tolerancia_max_delta_forma),
        "linha_base": bool(delta_base <= tolerancia_linha_base_mV),
        "pico": bool(delta_pico <= tolerancia_pico_mV),
    }
    motivos = []
    nomes = {
        "apd90": "APD90 mudou entre os protocolos",
        "forma": "a morfologia mudou entre os protocolos",
        "linha_base": "o potencial de repouso mudou entre os protocolos",
        "pico": "o pico mudou entre os protocolos",
    }
    for nome, atendido in checks.items():
        if not atendido:
            motivos.append(nomes[nome])

    return {
        "convergiu_entre_protocolos": bool(all(checks.values())),
        "apd90_protocolo_curto_ms": apd_curto if np.isfinite(apd_curto) else None,
        "apd90_protocolo_longo_ms": apd_longo if np.isfinite(apd_longo) else None,
        "delta_apd90_ms": float(delta_apd) if delta_apd is not None else None,
        "delta_apd90_relativo_ao_curto": float(delta_apd_rel) if delta_apd_rel is not None else None,
        "rmse_forma_normalizada": rmse_forma,
        "delta_maximo_forma_normalizada": max_delta_forma,
        "delta_linha_base_mV": float(delta_base),
        "delta_pico_mV": float(delta_pico),
        "criterios_atendidos": checks,
        "motivo": "Protocolos consistentes." if not motivos else "; ".join(motivos),
    }
