"""
Segmentação de vocalizações num WAV contínuo.

Porta direta de `highpass_filter`, `detect_vocalizations` e
`merge_close_vocalizations` (marmoset_analysis/classification_script.py),
sem mudar o algoritmo nem os valores padrão. A diferença é que aqui nada
lê/escreve disco nem depende de torch — quem orquestra é `pipeline.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class ParametrosSegmentacao:
    """Valores usados pelo pipeline original (`extract_and_classify_vocalizations`)."""

    cutoff_hz: float = 6000.0
    ordem_filtro: int = 5
    threshold: float = 0.02
    min_duration: float = 0.05
    max_duration: float = 5.0  # permite Phees contínuos e longos
    silence_duration: float = 0.05
    merge_threshold: float = 1.0  # silêncio máximo dentro de um mesmo canto
    duracao_minima_bloco: float = 0.3  # blocos mais curtos são descartados


def highpass_filter(audio_data, sample_rate, cutoff=6000, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return signal.filtfilt(b, a, audio_data)


def detect_vocalizations(
    audio_data,
    sample_rate,
    threshold=0.05,
    min_duration=0.05,
    max_duration=5.0,
    silence_duration=0.1,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    energy = np.abs(audio_data)
    window_size = int(0.02 * sample_rate)
    energy_smooth = np.convolve(energy, np.ones(window_size) / window_size, mode="same")
    energy_normalized = energy_smooth / np.max(energy_smooth)

    adaptive_threshold = np.percentile(energy_normalized, 95) * 0.8
    actual_threshold = max(threshold, adaptive_threshold)

    vocal_regions = (energy_normalized > actual_threshold).astype(int)
    diff = np.diff(vocal_regions)
    start_indices = np.where(diff == 1)[0] + 1
    end_indices = np.where(diff == -1)[0] + 1

    if vocal_regions[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
    if vocal_regions[-1] == 1:
        end_indices = np.append(end_indices, len(vocal_regions) - 1)

    valid_vocalizations = []
    for start, end in zip(start_indices, end_indices):
        duration = (end - start) / sample_rate
        if min_duration <= duration <= max_duration:
            start_margin = max(0, start - int(silence_duration * sample_rate))
            end_margin = min(len(audio_data), end + int(silence_duration * sample_rate))
            valid_vocalizations.append(
                {
                    "start_sample": start_margin,
                    "end_sample": end_margin,
                    "start_time": start_margin / sample_rate,
                    "end_time": end_margin / sample_rate,
                    "duration": (end_margin - start_margin) / sample_rate,
                    "energy_max": np.max(energy_normalized[start:end]),
                }
            )

    return valid_vocalizations, energy_normalized


def merge_close_vocalizations(vocalizations, merge_threshold=1.0) -> List[Dict[str, Any]]:
    if not vocalizations:
        return []

    merged_blocks = []
    current_block = [vocalizations[0]]

    for v in vocalizations[1:]:
        prev_end = current_block[-1]["end_time"]
        if v["start_time"] - prev_end <= merge_threshold:
            current_block.append(v)
        else:
            merged_blocks.append(
                {
                    "start_time": current_block[0]["start_time"],
                    "end_time": current_block[-1]["end_time"],
                    "vocalizations": current_block,
                }
            )
            current_block = [v]

    merged_blocks.append(
        {
            "start_time": current_block[0]["start_time"],
            "end_time": current_block[-1]["end_time"],
            "vocalizations": current_block,
        }
    )

    return merged_blocks


def segmentar(
    audio_data: np.ndarray, sample_rate: int, params: ParametrosSegmentacao = ParametrosSegmentacao()
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Filtra, detecta e agrupa. Devolve (áudio filtrado, blocos).

    Cada bloco ganha `"index"` (1-based, na ordem dos blocos agrupados — a
    mesma numeração dos nomes de arquivo do script original) e `"curto"`
    (True se mais curto que `duracao_minima_bloco`; o original pulava esses).
    """

    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    audio_filtrado = highpass_filter(audio_data, sample_rate, cutoff=params.cutoff_hz, order=params.ordem_filtro)

    vocalizacoes, _ = detect_vocalizations(
        audio_filtrado,
        sample_rate,
        threshold=params.threshold,
        min_duration=params.min_duration,
        max_duration=params.max_duration,
        silence_duration=params.silence_duration,
    )

    blocos = merge_close_vocalizations(vocalizacoes, merge_threshold=params.merge_threshold)
    for i, bloco in enumerate(blocos, start=1):
        bloco["index"] = i
        bloco["curto"] = (bloco["end_time"] - bloco["start_time"]) < params.duracao_minima_bloco

    return audio_filtrado, blocos


def _mmss(segundos: float) -> str:
    return f"{int(segundos // 60)}m{int(segundos % 60)}s"


def nome_arquivo_bloco(audio_basename: str, bloco: Dict[str, Any]) -> str:
    """Mesmo formato do original: `<wav>_block_001_0m9s-0m11s.wav` — é dele
    que analise_audio_eventos.py lê os tempos de volta."""

    return (
        f"{audio_basename}_block_{bloco['index']:03d}_"
        f"{_mmss(bloco['start_time'])}-{_mmss(bloco['end_time'])}.wav"
    )
