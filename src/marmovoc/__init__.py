"""
marmovoc — segmentação e classificação de vocalizações de saguis.

Biblioteca feita a partir dos scripts de `marmoset_analysis/`
(classification_script.py, duration.py), que continuam funcionando como
antes. O classificador é o ResNet50 do marmaudio (Lamothe & Best, BSD),
copiado em `marmovoc._vendor.marmaudio`.

Uso rápido:

    from marmovoc import Classificador, processar_arquivos

    clf = Classificador.da_pasta(r"...\\Models\\Models")
    df = processar_arquivos(["sessao_SOM.wav"], "Vocalizations_Extracted", clf)
"""

from marmovoc.duracao import duracao_pasta, escrever_durations_txt
from marmovoc.pipeline import processar_arquivo, processar_arquivos
from marmovoc.segmentacao import (
    ParametrosSegmentacao,
    detect_vocalizations,
    highpass_filter,
    merge_close_vocalizations,
    nome_arquivo_bloco,
    segmentar,
)


def __getattr__(nome):
    # Classificador importa torch só quando é pedido — `import marmovoc`
    # funciona sem torch instalado.
    if nome in ("Classificador", "classificar_bloco"):
        from marmovoc import classificacao

        return getattr(classificacao, nome)
    raise AttributeError(nome)


__version__ = "0.2.1"
