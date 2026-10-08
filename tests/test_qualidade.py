"""
Planura espectral, SNR e o filtro de tonalidade no pipeline — com sinais
sintéticos: tom (vocalização) x rajada de ruído branco (clique/impacto).
"""

import numpy as np
import soundfile as sf

from marmovoc.pipeline import COLUNAS, processar_arquivo
from marmovoc.qualidade import (
    PLANURA_MAXIMA,
    centroide_espectral_khz,
    e_ruido_banda_larga,
    envelope,
    piso_de_ruido_db,
    planura_espectral,
    snr_db,
)
from marmovoc.segmentacao import highpass_filter

SR = 48000


def _gravacao(tons=(), rajadas=(), duracao=20.0):
    """Ruído de fundo baixo + tons de 8 kHz + rajadas de ruído branco."""

    rng = np.random.default_rng(0)
    t = np.arange(int(duracao * SR)) / SR
    audio = rng.normal(0, 0.001, t.size)
    for ini, fim in tons:
        trecho = (t >= ini) & (t < fim)
        audio[trecho] += 0.5 * np.sin(2 * np.pi * 8000 * t[trecho])
    for ini, fim in rajadas:
        trecho = (t >= ini) & (t < fim)
        audio[trecho] += rng.normal(0, 0.3, int(trecho.sum()))
    return audio


def test_planura_separa_tom_de_ruido():
    filtrado = highpass_filter(_gravacao(tons=[(1, 2)], rajadas=[(3, 3.5)], duracao=4), SR)
    tom = planura_espectral(filtrado[int(1.1 * SR):int(1.9 * SR)], SR)
    ruido = planura_espectral(filtrado[int(3.05 * SR):int(3.45 * SR)], SR)
    assert tom < 0.01
    assert ruido > PLANURA_MAXIMA


def test_snr_mede_acima_do_fundo():
    filtrado = highpass_filter(_gravacao(tons=[(5, 6)]), SR)
    env = envelope(filtrado, SR)
    piso = piso_de_ruido_db(env)
    assert snr_db(env, int(5.2 * SR), int(5.8 * SR), piso) > 30
    assert abs(snr_db(env, int(10 * SR), int(11 * SR), piso)) < 10


def test_pipeline_descarta_banda_larga(tmp_path):
    wav = tmp_path / "S_SOM.wav"
    sf.write(wav, _gravacao(tons=[(2, 3)], rajadas=[(10, 10.6)]), SR)

    linhas = processar_arquivo(str(wav), str(tmp_path / "saida"))

    assert len(linhas) == 1
    assert abs(linhas[0]["start_time_seconds"] - 2.0) < 0.1
    assert linhas[0]["spectral_flatness"] < 0.01
    assert linhas[0]["snr_db"] > 30
    assert list(linhas[0].keys()) == COLUNAS
    # o .wav do bloco descartado continua salvo
    assert len(list((tmp_path / "saida" / "S_SOM").glob("*.wav"))) == 2


def test_filtro_desligado_mantem_tudo(tmp_path):
    wav = tmp_path / "S_SOM.wav"
    sf.write(wav, _gravacao(tons=[(2, 3)], rajadas=[(10, 10.6)]), SR)

    linhas = processar_arquivo(str(wav), str(tmp_path / "saida"), planura_maxima=None)

    assert len(linhas) == 2


def test_centroide_fica_na_frequencia_do_tom():
    filtrado = highpass_filter(_gravacao(tons=[(1, 2)], duracao=3), SR)
    assert abs(centroide_espectral_khz(filtrado[int(1.1 * SR):int(1.9 * SR)], SR) - 8.0) < 0.3


def test_regra_de_banda_larga():
    # banda larga evidente: sai, qualquer que seja o centroide
    assert e_ruido_banda_larga(0.08, 15.0)
    # planura intermediária + energia logo acima do corte (impactos do Café)
    assert e_ruido_banda_larga(0.041, 7.96)
    # planura intermediária + faixa alta (séries de Tsik): fica
    assert not e_ruido_banda_larga(0.043, 10.46)
    # vocalização típica
    assert not e_ruido_banda_larga(0.0002, 7.6)
    # segunda regra desligada: só o limite de 0,06
    assert not e_ruido_banda_larga(0.041, 7.96, planura_suspeita=None)
    # filtro desligado
    assert not e_ruido_banda_larga(0.5, 8.0, planura_maxima=None)
