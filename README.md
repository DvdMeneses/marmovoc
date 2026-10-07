# marmovoc

Segmentação e classificação de vocalizações de saguis, a partir de um WAV
contínuo (ex.: o `<prefixo>_SOM.wav` que o Marmosync grava).

É a versão "biblioteca" dos scripts de `marmoset_analysis/`
(`classification_script.py` e `duration.py`). **Os scripts originais não
foram alterados** e continuam funcionando — esta pasta é uma cópia
reorganizada, com os mesmos algoritmos e os mesmos valores padrão.

| Script original | Aqui |
|---|---|
| `highpass_filter`, `detect_vocalizations`, `merge_close_vocalizations` | `marmovoc/segmentacao.py` |
| classificação + votação por maior confiança | `marmovoc/classificacao.py` |
| loop de `extract_and_classify_vocalizations` | `marmovoc/pipeline.py` |
| `duration.py` | `marmovoc/duracao.py` |
| `Code_Usage/Code_Usage/marmaudio` | `marmovoc/_vendor/marmaudio` (BSD, ver `LICENCE.txt`) |

A análise antes/durante/depois dos estímulos (`analise_audio_eventos.py`)
ainda não entrou — o protocolo mudou e ela será refeita depois.

## Instalação

Na pasta `marmovoc`:

```powershell
pip install -e .                  # só segmentação (numpy, scipy, soundfile, pandas)
pip install -e ".[classificacao]" # + torch/torchvision para classificar
pip install -e ".[dev]"           # + pytest
```

`-e` (modo editável) faz qualquer alteração no código valer sem reinstalar.

Os modelos (`.stdc`/`.tsv`, ~100 MB) **não** são copiados: aponte para a
pasta onde já estão, `marmoset_analysis\Models\Models`.

## Linha de comando

```powershell
# Recorta e classifica (um WAV, vários, ou uma pasta inteira)
marmovoc segmentar D:\Marmosync\CSVs\Messi\Habituação\2026-10-07\14h12m12s\Messi_Habituação_2026-10-07_14h12m12s_SOM.wav `
    --saida Vocalizations_Extracted `
    --modelos C:\Users\David\Desktop\marmoset_analysis\Models\Models

# Só recorta, sem classificar (não precisa de torch)
marmovoc segmentar My_Audio_Files --saida Vocalizations_Extracted --sem-classificar

# durations.txt em cada pasta de blocos
marmovoc duracao Vocalizations_Extracted
```

Para não repetir `--modelos`, defina uma vez:
`$env:MARMOVOC_MODELOS = "C:\Users\David\Desktop\marmoset_analysis\Models\Models"`.

Saída de `segmentar` (igual ao script original):

- `<saida>/<nome do wav>/<nome do wav>_block_001_0m9s-0m11s.wav` — um arquivo por bloco;
- `<saida>/vocalization_analysis_corrected.csv` — mesmas colunas de antes
  (`--csv` muda o caminho).

## Em Python

```python
from marmovoc import Classificador, ParametrosSegmentacao, processar_arquivos, segmentar

clf = Classificador.da_pasta(r"C:\Users\David\Desktop\marmoset_analysis\Models\Models")
df = processar_arquivos(["sessao_SOM.wav"], "Vocalizations_Extracted", clf)

# Ou só a segmentação, sobre um array já carregado:
audio_filtrado, blocos = segmentar(audio, sample_rate, ParametrosSegmentacao(merge_threshold=0.5))
```

## Testes

```powershell
pip install -e ".[dev]"
python -m pytest
```

Usam sinais sintéticos e um classificador falso — não precisam do modelo nem de torch.

## Pontos de atenção

- **Taxa de amostragem.** O frontend do classificador
  (`MelFilter(96000, ...)` em `_vendor/marmaudio/classifier.py`) foi feito
  para áudio a **96 kHz**. O áudio não é reamostrado, como no script
  original. O WAV do Marmosync é gravado na taxa padrão do microfone (veja
  `samplerate_hz` no `_SOM_INFO.csv`) — se for 44,1/48 kHz, a escala de
  frequência que o modelo vê fica deslocada.
- **Alinhamento com o experimento.** Os tempos dos blocos são relativos
  ao início do WAV. O WAV do Marmosync começa alguns segundos depois do
  vídeo; o atraso está em `inicio_wav_relativo_experimento_s` no
  `_SOM_INFO.csv`.
