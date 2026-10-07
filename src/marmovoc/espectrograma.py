"""
Espectrogramas para inspeção visual dos blocos detectados.

Duas figuras:

- `salvar_espectrograma_bloco`: um bloco com contexto antes/depois, as
  bordas do bloco e o corte do passa-alta — para julgar se o que foi
  detectado é vocalização ou ruído.
- `salvar_visao_geral`: a gravação inteira com os blocos marcados — para
  ver se o ruído ocupa a mesma faixa de frequência das vocalizações.

Ambas usam o áudio **bruto** (sem o passa-alta): o objetivo é ver o que o
microfone captou, inclusive abaixo do corte. Usa `matplotlib.figure.Figure`
direto (sem `pyplot`), o que permite gerar figuras fora da thread do Tk.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy import signal

# Limita a largura da visão geral: uma sessão de 12 min tem ~15 mil janelas
# de STFT; mais colunas que pixels só gasta memória.
MAX_COLUNAS_VISAO_GERAL = 3000


def calcular(audio: np.ndarray, sample_rate: int, nperseg: int = 1024, noverlap: Optional[int] = None):
    """Devolve (frequências em kHz, tempos em s, potência em dB)."""

    if audio.ndim > 1:
        audio = audio[:, 0]
    nperseg = min(nperseg, len(audio))
    f, t, sxx = signal.spectrogram(
        audio.astype(np.float32), fs=sample_rate, window="hann", nperseg=nperseg, noverlap=noverlap, mode="psd"
    )
    return f / 1000.0, t, 10 * np.log10(sxx + 1e-12)


def _limites_cor(db: np.ndarray):
    """Contraste robusto: ignora os extremos para o ruído de fundo não dominar."""

    return float(np.percentile(db, 5)), float(np.percentile(db, 99.7))


def _figura(largura: float, altura: float):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(largura, altura), dpi=110)
    FigureCanvasAgg(fig)
    return fig


def _formatar_eixo(ax, cutoff_hz: Optional[float], f_max_khz: float) -> None:
    ax.set_ylabel("Frequência (kHz)")
    ax.set_ylim(0, f_max_khz)
    if cutoff_hz:
        ax.axhline(cutoff_hz / 1000.0, color="white", linestyle=":", linewidth=1, alpha=0.8)
        ax.text(
            0.995, cutoff_hz / 1000.0, f" corte {cutoff_hz / 1000:g} kHz",
            transform=ax.get_yaxis_transform(), color="white", fontsize=7, ha="right", va="bottom",
        )


def salvar_espectrograma_bloco(
    audio: np.ndarray,
    sample_rate: int,
    inicio_s: float,
    fim_s: float,
    caminho_png: str,
    titulo: str = "",
    cutoff_hz: Optional[float] = 6000.0,
    contexto_s: float = 0.5,
) -> str:
    """Espectrograma de [inicio_s - contexto, fim_s + contexto] da gravação
    completa `audio`, com as bordas do bloco tracejadas."""

    if audio.ndim > 1:
        audio = audio[:, 0]
    ini = max(0, int((inicio_s - contexto_s) * sample_rate))
    fim = min(len(audio), int((fim_s + contexto_s) * sample_rate))
    trecho = audio[ini:fim]
    deslocamento = ini / sample_rate

    f, t, db = calcular(trecho, sample_rate, nperseg=512, noverlap=384)
    vmin, vmax = _limites_cor(db)

    fig = _figura(8, 3.6)
    ax = fig.add_subplot(1, 1, 1)
    malha = ax.pcolormesh(t + deslocamento, f, db, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    for borda in (inicio_s, fim_s):
        ax.axvline(borda, color="#00e5e5", linestyle="--", linewidth=1.2)
    _formatar_eixo(ax, cutoff_hz, f[-1])
    ax.set_xlabel("Tempo na gravação (s)")
    ax.set_title(titulo, fontsize=9)
    fig.colorbar(malha, ax=ax, label="dB", pad=0.01)
    fig.tight_layout()
    fig.savefig(caminho_png)
    return caminho_png


def salvar_visao_geral(
    audio: np.ndarray,
    sample_rate: int,
    blocos: Sequence[Dict[str, Any]],
    caminho_png: str,
    titulo: str = "",
    cutoff_hz: Optional[float] = 6000.0,
) -> str:
    """Gravação inteira com cada bloco sombreado e identificado.

    Cada item de `blocos` precisa de `start_time` e `end_time` (s); `rotulo`
    é opcional e aparece acima do bloco."""

    f, t, db = calcular(audio, sample_rate, nperseg=2048, noverlap=0)

    # Reduz a largura agrupando colunas pelo máximo — preserva eventos curtos,
    # que uma média apagaria.
    if db.shape[1] > MAX_COLUNAS_VISAO_GERAL:
        grupo = int(np.ceil(db.shape[1] / MAX_COLUNAS_VISAO_GERAL))
        n = db.shape[1] // grupo * grupo
        db = db[:, :n].reshape(db.shape[0], -1, grupo).max(axis=2)
        t = t[:n].reshape(-1, grupo).mean(axis=1)

    vmin, vmax = _limites_cor(db)

    fig = _figura(14, 4.2)
    ax = fig.add_subplot(1, 1, 1)
    malha = ax.pcolormesh(t, f, db, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    for bloco in blocos:
        ax.axvspan(bloco["start_time"], bloco["end_time"], color="#00e5e5", alpha=0.25)
        rotulo = bloco.get("rotulo")
        if rotulo:
            ax.text(
                (bloco["start_time"] + bloco["end_time"]) / 2, f[-1] * 0.97, rotulo,
                color="#00e5e5", fontsize=7, ha="center", va="top", rotation=90,
            )
    _formatar_eixo(ax, cutoff_hz, f[-1])
    duracao = len(audio) / sample_rate
    ax.set_xlim(0, duracao)
    ax.set_xlabel("Tempo (s)")
    ax.set_title(titulo or f"{len(blocos)} bloco(s) em {duracao:.0f} s", fontsize=9)
    fig.colorbar(malha, ax=ax, label="dB", pad=0.01)
    fig.tight_layout()
    fig.savefig(caminho_png)
    return caminho_png
