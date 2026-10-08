"""
Medidas de qualidade de cada bloco detectado e o filtro de tonalidade.

- Planura espectral (spectral flatness): 0 = sinal tonal (energia em poucas
  frequências, como uma vocalização); 1 = banda larga (energia espalhada,
  como um clique, impacto ou chiado).
- Centroide espectral (kHz): "centro de massa" do espectro acima do corte.
  Impactos concentram energia logo acima do corte; chamadas como o Tsik têm
  uma faixa forte em ~10–12 kHz.
- SNR (dB): pico do envelope do bloco em relação ao ruído de fundo da
  própria gravação (mediana do envelope). Só informativo.

Planura e centroide são calculados sobre o áudio já filtrado pelo
passa-alta, só na banda acima do corte — a mesma que a detecção usa.

Regra do filtro (`e_ruido_banda_larga`), definida com 564 blocos (11
impactos, 22 vocalizações anotadas e 531 blocos rotulados como vocalização
em sessões de 7 animais):

- planura > 0,06  → ruído (banda larga evidente; impactos do Zico 0,071–0,100);
- planura > 0,035 e centroide < 9,4 kHz → ruído (impactos mais "tonais",
  como os do Café, 0,041–0,047 com centroide ≤ 9,2 kHz).

Vocalizações típicas ficaram com planura ≤ 0,013; a exceção foi uma sessão
atípica com longas séries de Tsik (planura até 0,043), mantida pelo
centroide alto (≥ 9,6 kHz).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal

PLANURA_MAXIMA = 0.06
PLANURA_SUSPEITA = 0.035
CENTROIDE_MINIMO_KHZ = 9.4

_JANELA_ENVELOPE_S = 0.02  # mesma janela da detecção (segmentacao.detect_vocalizations)


def envelope(audio_filtrado: np.ndarray, sample_rate: int) -> np.ndarray:
    janela = max(1, int(_JANELA_ENVELOPE_S * sample_rate))
    return np.convolve(np.abs(audio_filtrado), np.ones(janela) / janela, mode="same")


def piso_de_ruido_db(env: np.ndarray) -> float:
    """Ruído de fundo da gravação: mediana do envelope, em dB."""

    return float(20 * np.log10(np.median(env) + 1e-12))


def snr_db(env: np.ndarray, inicio: int, fim: int, piso_db: float) -> float:
    return float(20 * np.log10(env[inicio:fim].max() + 1e-12) - piso_db)


def _quadros_ativos(segmento: np.ndarray, sample_rate: int, f_min_hz: float):
    """Espectrograma da banda >= f_min_hz, só na metade mais energética dos quadros."""

    nperseg = min(512, len(segmento))
    f, _, pxx = signal.spectrogram(segmento, fs=sample_rate, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    sel = f >= f_min_hz
    banda = pxx[sel] + 1e-20
    if banda.size == 0:
        return None, None
    energia = banda.sum(axis=0)
    ativos = energia >= np.percentile(energia, 50)
    return f[sel], banda[:, ativos]


def planura_espectral(segmento: np.ndarray, sample_rate: int, f_min_hz: float = 6000.0) -> float:
    """Mediana da planura espectral nos quadros mais energéticos do bloco."""

    _, quadros = _quadros_ativos(segmento, sample_rate, f_min_hz)
    if quadros is None:
        return float("nan")
    planura = np.exp(np.log(quadros).mean(axis=0)) / quadros.mean(axis=0)
    return float(np.median(planura))


def centroide_espectral_khz(segmento: np.ndarray, sample_rate: int, f_min_hz: float = 6000.0) -> float:
    """Mediana do centroide espectral (kHz) nos quadros mais energéticos do bloco."""

    freqs, quadros = _quadros_ativos(segmento, sample_rate, f_min_hz)
    if quadros is None:
        return float("nan")
    centroide = (quadros * freqs[:, None]).sum(axis=0) / quadros.sum(axis=0)
    return float(np.median(centroide) / 1000.0)


def e_ruido_banda_larga(
    planura: float,
    centroide_khz: float,
    planura_maxima: Optional[float] = PLANURA_MAXIMA,
    planura_suspeita: Optional[float] = PLANURA_SUSPEITA,
    centroide_minimo_khz: float = CENTROIDE_MINIMO_KHZ,
) -> bool:
    """True se o bloco deve ser descartado como ruído (ver regra no topo do módulo).
    `planura_maxima=None` desliga o filtro inteiro; `planura_suspeita=None`
    desliga só a segunda regra (planura + centroide)."""

    if planura_maxima is None:
        return False
    if planura > planura_maxima:
        return True
    return planura_suspeita is not None and planura > planura_suspeita and centroide_khz < centroide_minimo_khz
