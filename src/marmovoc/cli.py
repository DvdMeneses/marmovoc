"""
Linha de comando: `marmovoc segmentar | duracao`.

Substitui as constantes no topo dos scripts originais (BASE_DIR,
MODELS_DIR, AUDIO_DIR, ...) por argumentos.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from typing import List

from marmovoc.duracao import escrever_durations_txt
from marmovoc.pipeline import processar_arquivos
from marmovoc.segmentacao import ParametrosSegmentacao

ENV_MODELOS = "MARMOVOC_MODELOS"


def _expandir_wavs(entradas: List[str]) -> List[str]:
    wavs = []
    for entrada in entradas:
        if os.path.isdir(entrada):
            wavs.extend(sorted(glob.glob(os.path.join(entrada, "*.wav"))))
        else:
            wavs.append(entrada)
    return wavs


def _cmd_segmentar(args) -> int:
    wavs = _expandir_wavs(args.wav)
    if not wavs:
        print("Nenhum arquivo WAV encontrado.")
        return 1

    classificador = None
    if not args.sem_classificar:
        pasta_modelos = args.modelos or os.environ.get(ENV_MODELOS)
        if not pasta_modelos:
            print(f"Informe --modelos (ou a variável {ENV_MODELOS}), ou use --sem-classificar.")
            return 1
        from marmovoc.classificacao import Classificador

        classificador = Classificador.da_pasta(pasta_modelos)

    params = ParametrosSegmentacao(cutoff_hz=args.cutoff, merge_threshold=args.merge)
    csv_saida = args.csv or os.path.join(args.saida, "vocalization_analysis_corrected.csv")
    os.makedirs(args.saida, exist_ok=True)

    df = processar_arquivos(wavs, args.saida, classificador, params, csv_saida=csv_saida)

    print(f"{len(df)} bloco(s) em {len(wavs)} arquivo(s). CSV: {csv_saida}")
    if classificador is not None and len(df):
        for rotulo, grupo in df.groupby("predicted_label"):
            print(f"  {rotulo}: {len(grupo)} ({grupo['confidence_percent'].mean():.1f}% conf. média)")
    return 0


def _cmd_duracao(args) -> int:
    for nome in sorted(os.listdir(args.raiz)):
        pasta = os.path.join(args.raiz, nome)
        if os.path.isdir(pasta):
            print(escrever_durations_txt(pasta))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="marmovoc", description="Segmentação e classificação de vocalizações de saguis.")
    parser.add_argument("-v", "--verbose", action="store_true", help="mostra o log detalhado")
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("segmentar", help="recorta (e classifica) as vocalizações de WAVs completos")
    p.add_argument("wav", nargs="+", help="arquivo(s) .wav ou pasta(s) com .wav")
    p.add_argument("--saida", required=True, help="pasta onde ficam os blocos (uma subpasta por WAV) e o CSV")
    p.add_argument("--modelos", help=f"pasta com o .stdc e o .tsv do classificador (ou variável {ENV_MODELOS})")
    p.add_argument("--sem-classificar", action="store_true", help="só recorta, sem carregar o modelo (não precisa de torch)")
    p.add_argument("--csv", help="caminho do CSV (padrão: <saida>/vocalization_analysis_corrected.csv)")
    p.add_argument("--cutoff", type=float, default=ParametrosSegmentacao.cutoff_hz, help="passa-alta em Hz")
    p.add_argument("--merge", type=float, default=ParametrosSegmentacao.merge_threshold, help="silêncio máximo (s) dentro de um bloco")
    p.set_defaults(func=_cmd_segmentar)

    p = sub.add_parser("duracao", help="grava durations.txt em cada subpasta de blocos")
    p.add_argument("raiz", help="pasta que contém as pastas de blocos")
    p.set_defaults(func=_cmd_duracao)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="[%(levelname)s] %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
