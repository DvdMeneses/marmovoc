"""
Medidas de qualidade de cada bloco detectado.

- Planura espectral (spectral flatness): 0 = sinal tonal (energia em poucas
  frequências, como uma vocalização); 1 = banda larga (energia espalhada,
  como um clique, impacto ou chiado). É a base do filtro de tonalidade.
- SNR (dB): pico do envelope do bloco em relação ao ruído de fundo da
  própria gravação (mediana do envelope).

Valores calculados sobre o áudio já filtrado pelo passa-alta, só na banda
acima do corte — a mesma que a detecção usa.

O limite padrão de planura (`PLANURA_MAXIMA`) veio da inspeção de 590
blocos (25 gravações de 5 animais + 22 vocalizações anotadas da validação):
vocalizações ficaram em até 0,043; cliques/impactos, a partir de 0,071.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

PLANURA_MAXIMA = 0.06

_JANELA_ENVELOPE_S = 0.02  # mesma janela da detecção (segmentacao.detect_vocalizations)


def envelope(audio_filtrado: np.ndarray, sample_rate: int) -> np.ndarray:
    janela = max(1, int(_JANELA_ENVELOPE_S * sample_rate))
    return np.convolve(np.abs(audio_filtrado), np.ones(janela) / janela, mode="same")


def piso_de_ruido_db(env: np.ndarray) -> float:
    """Ruído de fundo da gravação: mediana do envelope, em dB."""

    return float(20 * np.log10(np.median(env) + 1e-12))


def snr_db(env: np.ndarray, inicio: int, fim: int, piso_db: float) -> float:
    return float(20 * np.log10(env[inicio:fim].max() + 1e-12) - piso_db)


def planura_espectral(segmento: np.ndarray, sample_rate: int, f_min_hz: float = 6000.0) -> float:
    """Mediana da planura espectral nos quadros mais energéticos do bloco
    (metade superior), considerando só frequências >= `f_min_hz`."""

    nperseg = min(512, len(segmento))
    f, _, pxx = signal.spectrogram(segmento, fs=sample_rate, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    banda = pxx[f >= f_min_hz] + 1e-20
    if banda.size == 0:
        return float("nan")
    energia = banda.sum(axis=0)
    ativos = energia >= np.percentile(energia, 50)
    quadros = banda[:, ativos]
    planura = np.exp(np.log(quadros).mean(axis=0)) / quadros.mean(axis=0)
    return float(np.median(planura))
