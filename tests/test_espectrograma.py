"""
Espectrogramas: geração dos PNGs (sem verificar o desenho — só que saem,
nos lugares certos e com a gravação inteira processável).
"""

import numpy as np
import soundfile as sf

from marmovoc.espectrograma import calcular, salvar_espectrograma_bloco, salvar_visao_geral
from marmovoc.pipeline import processar_arquivo

SR = 48000


def _sinal(chamadas, duracao=20.0, freq=8000.0):
    rng = np.random.default_rng(0)
    t = np.arange(int(duracao * SR)) / SR
    audio = rng.normal(0, 0.001, t.size)
    for ini, fim in chamadas:
        trecho = (t >= ini) & (t < fim)
        audio[trecho] += 0.5 * np.sin(2 * np.pi * freq * t[trecho])
    return audio


def test_calcular_acha_a_frequencia_da_chamada():
    f, _, db = calcular(_sinal([(2.0, 3.0)], duracao=5.0), SR)
    pico = f[np.argmax(db.max(axis=1))]
    assert abs(pico - 8.0) < 0.1  # kHz


def test_figuras_sao_salvas(tmp_path):
    audio = _sinal([(2.0, 3.0)])
    bloco = salvar_espectrograma_bloco(audio, SR, 1.95, 3.05, str(tmp_path / "bloco.png"), titulo="teste")
    geral = salvar_visao_geral(
        audio, SR, [{"start_time": 1.95, "end_time": 3.05, "rotulo": "#1"}], str(tmp_path / "geral.png")
    )
    for caminho in (bloco, geral):
        assert open(caminho, "rb").read(8) == b"\x89PNG\r\n\x1a\n"


def test_bloco_no_inicio_da_gravacao_nao_quebra(tmp_path):
    """Contexto antes do bloco fica fora do áudio — tem que cortar em 0."""

    audio = _sinal([(0.0, 0.8)], duracao=3.0)
    salvar_espectrograma_bloco(audio, SR, 0.0, 0.85, str(tmp_path / "b.png"))
    assert (tmp_path / "b.png").exists()


def test_pipeline_com_espectrogramas(tmp_path):
    wav = tmp_path / "Sessao_SOM.wav"
    sf.write(wav, _sinal([(2.0, 3.0), (10.0, 11.5)]), SR)

    linhas = processar_arquivo(str(wav), str(tmp_path / "saida"), espectrogramas=True)

    pasta = tmp_path / "saida" / "Sessao_SOM"
    pngs = sorted(p.name for p in pasta.glob("*.png"))
    assert "Sessao_SOM_espectrograma.png" in pngs
    for linha in linhas:
        assert linha["block_file"].replace(".wav", ".png") in pngs


def test_sem_espectrogramas_nao_gera_png(tmp_path):
    wav = tmp_path / "Sessao_SOM.wav"
    sf.write(wav, _sinal([(2.0, 3.0)]), SR)

    processar_arquivo(str(wav), str(tmp_path / "saida"))

    assert not list((tmp_path / "saida").rglob("*.png"))
