"""
Testes da segmentação com sinal sintético — sem modelo, sem torch.
"""

import numpy as np
import soundfile as sf

from marmovoc.pipeline import COLUNAS, processar_arquivo
from marmovoc.segmentacao import (
    ParametrosSegmentacao,
    merge_close_vocalizations,
    nome_arquivo_bloco,
    segmentar,
)

SR = 48000


def _sinal_com_chamadas(chamadas, duracao_total=20.0, freq=8000.0):
    """Ruído baixo + senoides de `freq` Hz nos intervalos (início, fim)."""

    rng = np.random.default_rng(0)
    t = np.arange(int(duracao_total * SR)) / SR
    audio = rng.normal(0, 0.001, t.size)
    for ini, fim in chamadas:
        trecho = (t >= ini) & (t < fim)
        audio[trecho] += 0.5 * np.sin(2 * np.pi * freq * t[trecho])
    return audio


def test_segmentar_encontra_chamadas_separadas():
    audio = _sinal_com_chamadas([(2.0, 3.0), (10.0, 11.5)])

    _, blocos = segmentar(audio, SR)

    assert len(blocos) == 2
    assert abs(blocos[0]["start_time"] - 2.0) < 0.1
    assert abs(blocos[1]["end_time"] - 11.5) < 0.1
    assert [b["index"] for b in blocos] == [1, 2]
    assert not any(b["curto"] for b in blocos)


def test_chamadas_proximas_viram_um_bloco():
    # 0,5 s de silêncio entre as duas — abaixo do merge_threshold (1 s)
    audio = _sinal_com_chamadas([(5.0, 5.6), (6.1, 6.8)])

    _, blocos = segmentar(audio, SR)

    assert len(blocos) == 1
    assert len(blocos[0]["vocalizations"]) == 2


def test_passa_alta_remove_som_grave():
    # 1 kHz fica abaixo do corte de 6 kHz: não deve sobrar bloco nenhum
    # que não seja o trecho agudo.
    audio = _sinal_com_chamadas([(2.0, 3.0)], freq=1000.0) + _sinal_com_chamadas([(10.0, 11.0)])

    _, blocos = segmentar(audio, SR)

    assert len(blocos) == 1
    assert abs(blocos[0]["start_time"] - 10.0) < 0.1


def test_merge_respeita_limite():
    v = lambda ini, fim: {"start_time": ini, "end_time": fim}
    blocos = merge_close_vocalizations([v(0, 1), v(1.5, 2), v(4, 5)], merge_threshold=1.0)
    assert [(b["start_time"], b["end_time"]) for b in blocos] == [(0, 2), (4, 5)]
    assert merge_close_vocalizations([]) == []


def test_nome_arquivo_bloco_mesmo_formato_do_original():
    bloco = {"index": 1, "start_time": 9.423, "end_time": 71.127}
    assert nome_arquivo_bloco("Teste_SOM", bloco) == "Teste_SOM_block_001_0m9s-1m11s.wav"


def test_processar_arquivo_sem_classificador(tmp_path):
    wav = tmp_path / "Sujeito_Habituacao_SOM.wav"
    sf.write(wav, _sinal_com_chamadas([(2.0, 3.0), (10.0, 11.5)]), SR)

    linhas = processar_arquivo(str(wav), str(tmp_path / "saida"))

    assert len(linhas) == 2
    assert list(linhas[0].keys()) == COLUNAS
    assert linhas[0]["predicted_label"] == ""

    pasta_blocos = tmp_path / "saida" / "Sujeito_Habituacao_SOM"
    arquivos = sorted(p.name for p in pasta_blocos.iterdir())
    assert arquivos == [linhas[0]["block_file"], linhas[1]["block_file"]]


def test_bloco_curto_e_descartado(tmp_path):
    wav = tmp_path / "curto_SOM.wav"
    # 0,15 s de chamada: passa no min_duration (0,05 s), mas o bloco fica < 0,3 s
    sf.write(wav, _sinal_com_chamadas([(5.0, 5.15)]), SR)

    linhas = processar_arquivo(str(wav), str(tmp_path / "saida"), params=ParametrosSegmentacao())

    assert linhas == []
