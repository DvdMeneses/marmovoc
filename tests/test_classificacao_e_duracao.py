"""
Votação do bloco (com classificador fake — sem torch/modelo) e duração.
"""

import numpy as np
import soundfile as sf

from marmovoc.classificacao import classificar_bloco
from marmovoc.duracao import duracao_pasta, escrever_durations_txt, formatar_hhmmss


class FakeClassificador:
    """Devolve, em ordem, os (rótulo, confiança) pré-definidos."""

    def __init__(self, respostas):
        self._respostas = list(respostas)
        self.chamadas = 0

    def classificar(self, segmento):
        resposta = self._respostas[self.chamadas]
        self.chamadas += 1
        if isinstance(resposta, Exception):
            raise resposta
        return resposta


def _voc(ini, fim, energia):
    return {"start_sample": ini, "end_sample": fim, "energy_max": energia}


def test_bloco_fica_com_a_vocalizacao_de_maior_confianca():
    audio = np.full(300, 0.5)
    bloco = {"vocalizations": [_voc(0, 100, 0.3), _voc(100, 200, 0.9), _voc(200, 300, 0.5)]}
    clf = FakeClassificador([("Twitter", 40.0), ("Phee", 95.0), ("Trill", 60.0)])

    melhor = classificar_bloco(clf, audio, bloco)

    assert melhor == {"predicted_label": "Phee", "confidence_percent": 95.0, "energy_max": 0.9}


def test_vocalizacao_fraca_ou_com_erro_e_ignorada():
    audio = np.concatenate([np.full(100, 0.001), np.full(100, 0.5)])
    bloco = {"vocalizations": [_voc(0, 100, 0.1), _voc(100, 200, 0.8), _voc(100, 200, 0.8)]}
    # 1ª vocalização é fraca demais e nem chega ao classificador
    clf = FakeClassificador([RuntimeError("falhou"), ("Phee", 70.0)])

    melhor = classificar_bloco(clf, audio, bloco)

    assert clf.chamadas == 2
    assert melhor["predicted_label"] == "Phee"


def test_bloco_sem_classificacao_devolve_none():
    audio = np.zeros(100)
    assert classificar_bloco(FakeClassificador([]), audio, {"vocalizations": [_voc(0, 100, 0.0)]}) is None


def test_duracao_pasta(tmp_path):
    sf.write(tmp_path / "a.wav", np.zeros(48000), 48000)  # 1 s
    sf.write(tmp_path / "b.wav", np.zeros(24000), 48000)  # 0,5 s
    (tmp_path / "notas.txt").write_text("ignorado")

    d = duracao_pasta(str(tmp_path))
    assert d["arquivos"] == 2
    assert abs(d["total_segundos"] - 1.5) < 1e-9

    conteudo = open(escrever_durations_txt(str(tmp_path)), encoding="utf-8").read()
    assert "Total de arquivos .wav: 2" in conteudo
    assert "Formato HH:MM:SS: 00:00:01" in conteudo


def test_formatar_hhmmss():
    assert formatar_hhmmss(3725.9) == "01:02:05"
