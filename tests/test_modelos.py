"""
Localização/download do classificador — sem rede: o "Zenodo" aqui é um
zip local servido por URL file://.
"""

import pathlib
import zipfile

import pytest

from marmovoc import modelos


def _zip_falso(tmp_path, dentro="Code_Usage/Code_Usage/"):
    caminho = tmp_path / "Code_Usage.zip"
    with zipfile.ZipFile(caminho, "w") as z:
        z.writestr(dentro + modelos.NOMES_MODELO[0], b"pesos")
        z.writestr(dentro + modelos.NOMES_ROTULOS[0], "label\nPhee\n")
        z.writestr(dentro + "paper_usage_note_code_p1.py", "# não deve ser extraído")
    return pathlib.Path(caminho).as_uri()


def test_localizar_aceita_nome_do_lab_e_do_zenodo(tmp_path):
    assert modelos.localizar(str(tmp_path)) is None

    (tmp_path / modelos.NOMES_MODELO[1]).write_bytes(b"x")  # CLF_... (pasta do lab)
    (tmp_path / modelos.NOMES_ROTULOS[0]).write_text("label\n")  # nome do Zenodo
    modelo, rotulos = modelos.localizar(str(tmp_path))

    assert modelo.endswith(modelos.NOMES_MODELO[1])
    assert rotulos.endswith(modelos.NOMES_ROTULOS[0])


def test_baixar_extrai_so_o_classificador(tmp_path):
    url = _zip_falso(tmp_path)
    destino = tmp_path / "cache"

    modelo, rotulos = modelos.baixar_modelos(str(destino), url=url)

    assert open(modelo, "rb").read() == b"pesos"
    assert sorted(p.name for p in destino.iterdir()) == sorted([modelos.NOMES_MODELO[0], modelos.NOMES_ROTULOS[0]])


def test_baixar_nao_baixa_de_novo(tmp_path):
    destino = tmp_path / "cache"
    destino.mkdir()
    (destino / modelos.NOMES_MODELO[0]).write_bytes(b"ja estava")
    (destino / modelos.NOMES_ROTULOS[0]).write_text("label\n")

    # URL inválida: se tentasse baixar, falharia.
    modelo, _ = modelos.baixar_modelos(str(destino), url="file:///nao/existe.zip")
    assert open(modelo, "rb").read() == b"ja estava"


def test_zip_sem_classificador_da_erro(tmp_path):
    caminho = tmp_path / "vazio.zip"
    with zipfile.ZipFile(caminho, "w") as z:
        z.writestr("outro.txt", "nada")

    with pytest.raises(RuntimeError):
        modelos.baixar_modelos(str(tmp_path / "cache"), url=caminho.as_uri())


def test_pasta_cache_respeita_variavel(monkeypatch, tmp_path):
    monkeypatch.setenv(modelos.ENV_CACHE, str(tmp_path))
    assert modelos.pasta_cache() == str(tmp_path)
