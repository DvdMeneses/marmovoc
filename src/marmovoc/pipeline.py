"""
Pipeline completo de um WAV: segmentar → salvar blocos → classificar.

Equivale ao corpo do loop de `extract_and_classify_vocalizations`
(marmoset_analysis/classification_script.py), recebendo caminhos e
parâmetros em vez de usar constantes, `input()` e diálogo do Tk.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import soundfile as sf

from marmovoc.classificacao import Classificador, classificar_bloco
from marmovoc.segmentacao import ParametrosSegmentacao, nome_arquivo_bloco, segmentar

log = logging.getLogger(__name__)

# Blocos com confiança abaixo disso não entram na tabela (mesmo corte do original).
CONFIANCA_MINIMA_PERCENT = 10.0

# MAX_PATH: sem "caminhos longos" habilitado no Windows, nada acima disso abre.
LIMITE_CAMINHO_WINDOWS = 260

COLUNAS = [
    "original_file",
    "block_file",
    "block_index",
    "start_time_seconds",
    "end_time_seconds",
    "duration_seconds",
    "offset_in_block_seconds",
    "predicted_label",
    "confidence_percent",
    "energy_max",
    "start_time_formatted",
    "end_time_formatted",
]


def processar_arquivo(
    caminho_wav: str,
    pasta_saida: str,
    classificador: Optional[Classificador] = None,
    params: ParametrosSegmentacao = ParametrosSegmentacao(),
    confianca_minima: float = CONFIANCA_MINIMA_PERCENT,
) -> List[Dict[str, Any]]:
    """Segmenta um WAV, salva cada bloco em `pasta_saida/<nome do wav>/` e
    devolve uma linha por bloco (colunas em `COLUNAS`).

    Sem `classificador`, todos os blocos entram na tabela com rótulo e
    confiança vazios — útil para só recortar, sem precisar de torch.
    """

    audio_data, sample_rate = sf.read(caminho_wav)
    audio_filtrado, blocos = segmentar(audio_data, sample_rate, params)

    audio_basename = os.path.splitext(os.path.basename(caminho_wav))[0]
    pasta_blocos = os.path.join(pasta_saida, audio_basename)
    os.makedirs(pasta_blocos, exist_ok=True)

    log.info("%s: %d Hz, %d blocos", os.path.basename(caminho_wav), sample_rate, len(blocos))

    linhas: List[Dict[str, Any]] = []
    for bloco in blocos:
        if bloco["curto"]:
            continue

        inicio = int(bloco["start_time"] * sample_rate)
        fim = int(bloco["end_time"] * sample_rate)
        nome_bloco = nome_arquivo_bloco(audio_basename, bloco)
        caminho_bloco = os.path.abspath(os.path.join(pasta_blocos, nome_bloco))
        if os.name == "nt" and len(caminho_bloco) >= LIMITE_CAMINHO_WINDOWS:
            # libsndfile só devolve "System error" nesse caso — melhor dizer o porquê.
            raise OSError(
                f"Caminho com {len(caminho_bloco)} caracteres (limite do Windows: "
                f"{LIMITE_CAMINHO_WINDOWS - 1}). Escolha uma pasta de saída mais curta: {caminho_bloco}"
            )
        sf.write(caminho_bloco, audio_filtrado[inicio:fim], sample_rate)

        if classificador is not None:
            resultado = classificar_bloco(classificador, audio_filtrado, bloco)
            if resultado is None or resultado["confidence_percent"] < confianca_minima:
                continue
        else:
            resultado = {"predicted_label": "", "confidence_percent": None, "energy_max": None}

        energia = resultado["energy_max"]
        linhas.append(
            {
                "original_file": os.path.basename(caminho_wav),
                "block_file": nome_bloco,
                "block_index": bloco["index"],
                "start_time_seconds": round(bloco["start_time"], 3),
                "end_time_seconds": round(bloco["end_time"], 3),
                "duration_seconds": round(bloco["end_time"] - bloco["start_time"], 3),
                "offset_in_block_seconds": 0.0,
                "predicted_label": resultado["predicted_label"],
                "confidence_percent": resultado["confidence_percent"],
                "energy_max": round(energia, 3) if energia is not None else None,
                "start_time_formatted": f"{int(bloco['start_time'] // 60)}m {int(bloco['start_time'] % 60)}s",
                "end_time_formatted": f"{int(bloco['end_time'] // 60)}m {int(bloco['end_time'] % 60)}s",
            }
        )

    return linhas


def processar_arquivos(
    caminhos_wav: Iterable[str],
    pasta_saida: str,
    classificador: Optional[Classificador] = None,
    params: ParametrosSegmentacao = ParametrosSegmentacao(),
    csv_saida: Optional[str] = None,
):
    """Roda `processar_arquivo` em vários WAVs (um erro não interrompe os
    outros) e devolve um DataFrame. Com `csv_saida`, grava também o CSV
    (utf-8-sig, como o original)."""

    import pandas as pd

    todas: List[Dict[str, Any]] = []
    for caminho in caminhos_wav:
        try:
            todas.extend(processar_arquivo(caminho, pasta_saida, classificador, params))
        except Exception as e:
            log.error("Erro ao processar %s: %s", caminho, e)

    df = pd.DataFrame(todas, columns=COLUNAS)
    if csv_saida:
        df.to_csv(csv_saida, index=False, encoding="utf-8-sig")
    return df
