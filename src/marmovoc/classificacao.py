"""
Classificação dos blocos de vocalização com o modelo ResNet50 do marmaudio.

Porta da parte de classificação de `extract_and_classify_vocalizations`
(marmoset_analysis/classification_script.py): mesma votação "maior
confiança vence" entre as vocalizações de um bloco. torch só é importado
quando um `Classificador` é criado — segmentar não precisa dele.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from marmovoc import modelos

# O frontend do modelo (MelFilter em marmaudio/classifier.py) foi treinado
# com áudio a 96 kHz. Áudio gravado em outra taxa não é reamostrado aqui —
# mesma coisa que o script original fazia.
SAMPLE_RATE_MODELO = 96000


class Classificador:
    def __init__(self, caminho_modelo: str, caminho_rotulos: str):
        import pandas as pd

        from marmovoc._vendor.marmaudio.classifier import load_classifier

        for caminho in (caminho_modelo, caminho_rotulos):
            if not os.path.exists(caminho):
                raise FileNotFoundError(caminho)

        rotulos = pd.read_csv(caminho_rotulos, sep="\t").label.unique()
        self.idx_para_rotulo = {i: k for i, k in enumerate(rotulos)}
        self.modelo = load_classifier(caminho_modelo, rotulos)

    @classmethod
    def da_pasta(cls, pasta_modelos: Optional[str] = None) -> "Classificador":
        """Carrega o .stdc e o .tsv de `pasta_modelos` (ex.: `Models\\Models`).
        Sem pasta, usa o cache — e baixa do Zenodo na primeira vez."""

        if pasta_modelos is None:
            return cls(*modelos.baixar_modelos())

        encontrados = modelos.localizar(pasta_modelos)
        if encontrados is None:
            raise FileNotFoundError(f"Classificador (.stdc + .tsv) não encontrado em {pasta_modelos}")
        return cls(*encontrados)

    def classificar(self, segmento: np.ndarray) -> Tuple[str, float]:
        """Devolve (rótulo, confiança em %) para um trecho de áudio."""

        import torch

        from marmovoc._vendor.marmaudio.classifier import run_classifier

        with torch.no_grad():
            predictions = run_classifier(self.modelo, segmento)
        probabilities = torch.nn.functional.softmax(predictions.detach().cpu(), dim=-1)
        confidence, predicted_idx = torch.max(probabilities, dim=-1)
        return self.idx_para_rotulo.get(predicted_idx.item(), "UNKNOWN"), confidence.item() * 100


def classificar_bloco(
    classificador: Classificador,
    audio_filtrado: np.ndarray,
    bloco: Dict[str, Any],
    amplitude_minima: float = 0.01,
) -> Optional[Dict[str, Any]]:
    """Classifica cada vocalização do bloco e fica com a de maior confiança.
    Vocalizações com pico abaixo de `amplitude_minima` são ignoradas.
    Devolve `None` se nenhuma pôde ser classificada."""

    melhor: Optional[Dict[str, Any]] = None

    for v in bloco["vocalizations"]:
        segmento = audio_filtrado[v["start_sample"] : v["end_sample"]]
        if np.max(np.abs(segmento)) < amplitude_minima:
            continue

        try:
            rotulo, confianca = classificador.classificar(segmento)
        except Exception:
            continue

        if melhor is None or confianca > melhor["confidence_percent"]:
            melhor = {"predicted_label": rotulo, "confidence_percent": confianca, "energy_max": v["energy_max"]}

    return melhor
