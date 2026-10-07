"""
Localização e download do classificador do MarmAudio.

O classificador não vem no repositório: é baixado da fonte oficial, o
registro Zenodo do dataset MarmAudio (Lamothe et al., 2025 — CC BY 4.0,
https://doi.org/10.5281/zenodo.15017207), arquivo `Code_Usage.zip`.
Só o `.stdc` e o `.tsv` são extraídos, uma vez, para uma pasta de cache.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from typing import Optional, Sequence, Tuple

log = logging.getLogger(__name__)

URL_CODE_USAGE = "https://zenodo.org/api/records/15017207/files/Code_Usage.zip/content"

# Nomes dentro do Code_Usage.zip e o nome usado na pasta Models\Models do
# lab (mesmo arquivo, renomeado com prefixo CLF_). Aceita qualquer um.
NOMES_MODELO: Sequence[str] = (
    "resnet50_logMel128_trainsplit_wo_brown_wo_vocs_epoch1.stdc",
    "CLF_resnet50_logMel128_trainsplit_wo_brown_wo_vocs.stdc",
)
NOMES_ROTULOS: Sequence[str] = (
    "resnet50_logMel128_trainsplit_wo_brown_wo_vocs_labels_used.tsv",
    "CLF_resnet50_logMel128_trainsplit_wo_brown_wo_vocs_labels_used.tsv",
)

ENV_CACHE = "MARMOVOC_CACHE"


def pasta_cache() -> str:
    """`MARMOVOC_CACHE`, ou `%LOCALAPPDATA%\\marmovoc` (Windows) / `~/.cache/marmovoc`."""

    if os.environ.get(ENV_CACHE):
        return os.environ[ENV_CACHE]
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "marmovoc")


def localizar(pasta: str) -> Optional[Tuple[str, str]]:
    """(modelo, rótulos) dentro de `pasta`, com qualquer um dos nomes aceitos."""

    def primeiro(nomes):
        for nome in nomes:
            caminho = os.path.join(pasta, nome)
            if os.path.exists(caminho):
                return caminho
        return None

    modelo, rotulos = primeiro(NOMES_MODELO), primeiro(NOMES_ROTULOS)
    return (modelo, rotulos) if modelo and rotulos else None


def baixar_modelos(destino: Optional[str] = None, url: str = URL_CODE_USAGE) -> Tuple[str, str]:
    """Garante o classificador em `destino` (padrão: `pasta_cache()`),
    baixando o `Code_Usage.zip` (~89 MB) só se ainda não estiver lá."""

    destino = destino or pasta_cache()
    existentes = localizar(destino)
    if existentes:
        return existentes

    os.makedirs(destino, exist_ok=True)
    log.warning("Baixando o classificador do Zenodo (~89 MB, só na primeira vez): %s", url)

    with tempfile.TemporaryDirectory() as tmp:
        caminho_zip = os.path.join(tmp, "Code_Usage.zip")
        with urllib.request.urlopen(url) as resposta, open(caminho_zip, "wb") as f:
            shutil.copyfileobj(resposta, f, length=1024 * 1024)

        with zipfile.ZipFile(caminho_zip) as z:
            for nomes in (NOMES_MODELO, NOMES_ROTULOS):
                membro = next(
                    (m for m in z.namelist() if os.path.basename(m) in nomes), None
                )
                if membro is None:
                    raise RuntimeError(f"{nomes[0]} não encontrado em {url}")

                # Grava com .part e renomeia: um download interrompido não
                # deixa um modelo pela metade na pasta de cache.
                final = os.path.join(destino, os.path.basename(membro))
                with z.open(membro) as origem, open(final + ".part", "wb") as saida:
                    shutil.copyfileobj(origem, saida, length=1024 * 1024)
                os.replace(final + ".part", final)

    encontrados = localizar(destino)
    if not encontrados:
        raise RuntimeError(f"Download concluído, mas o classificador não está em {destino}")
    log.warning("Classificador salvo em %s", destino)
    return encontrados
