"""
Duração total dos blocos de vocalização por pasta.

Porta de marmoset_analysis/duration.py. Usa `soundfile` em vez de `pydub`
(mesmo resultado para WAV, sem precisar de ffmpeg).
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import soundfile as sf

log = logging.getLogger(__name__)


def formatar_hhmmss(segundos: float) -> str:
    s = int(segundos)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def duracao_pasta(pasta: str) -> Dict[str, float]:
    """Soma a duração de todos os `.wav` de uma pasta."""

    wavs = [f for f in os.listdir(pasta) if f.lower().endswith(".wav")]
    total = 0.0
    for nome in wavs:
        try:
            total += sf.info(os.path.join(pasta, nome)).duration
        except Exception as e:
            log.warning("Erro ao ler %s: %s", nome, e)
    return {"arquivos": len(wavs), "total_segundos": total}


def escrever_durations_txt(pasta: str) -> str:
    """Grava `durations.txt` dentro da pasta, no mesmo formato do script original."""

    d = duracao_pasta(pasta)
    caminho = os.path.join(pasta, "durations.txt")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(f"Total de arquivos .wav: {d['arquivos']}\n")
        f.write(f"Total em segundos: {d['total_segundos']:.2f}\n")
        f.write(f"Total em minutos: {d['total_segundos'] / 60:.2f}\n")
        f.write(f"Formato HH:MM:SS: {formatar_hhmmss(d['total_segundos'])}\n")
    return caminho
