# marmovoc

Segmentação e classificação de vocalizações de saguis, a partir de um WAV
contínuo (ex.: o `<prefixo>_SOM.wav` que o Marmosync grava).

É a versão "biblioteca" dos scripts de `marmoset_analysis/`
(`classification_script.py` e `duration.py`). **Os scripts originais não
foram alterados** e continuam funcionando — esta é uma cópia reorganizada,
com os mesmos algoritmos e os mesmos valores padrão.

| Script original | Aqui |
|---|---|
| `highpass_filter`, `detect_vocalizations`, `merge_close_vocalizations` | `marmovoc/segmentacao.py` |
| classificação + votação por maior confiança | `marmovoc/classificacao.py` |
| loop de `extract_and_classify_vocalizations` | `marmovoc/pipeline.py` |
| `duration.py` | `marmovoc/duracao.py` |
| `Code_Usage/Code_Usage/marmaudio` | `marmovoc/_vendor/marmaudio` (BSD, ver `LICENCE.txt`) |
| caminho fixo `Models\Models` | `marmovoc/modelos.py` (download do Zenodo + cache) |

A análise antes/durante/depois dos estímulos (`analise_audio_eventos.py`)
ainda não entrou — o protocolo mudou e ela será refeita depois.

## Instalação

```powershell
pip install "marmovoc[classificacao] @ git+https://github.com/DvdMeneses/marmovoc.git"
marmovoc baixar-modelos   # opcional: baixa o classificador agora, em vez de no 1º uso
```

- `[classificacao]` instala torch/torchvision. Sem ele, dá para segmentar
  (`--sem-classificar`), mas não classificar.
- O classificador (`.stdc` + `.tsv`, ~94 MB) **não** está no repositório:
  é baixado do registro Zenodo do MarmAudio (fonte oficial, ~89 MB) na
  primeira classificação e fica em cache — `%LOCALAPPDATA%\marmovoc`, ou a
  pasta em `MARMOVOC_CACHE`.
- Quem já tem os arquivos (ex.: `marmoset_analysis\Models\Models`) pode
  apontar para eles com `--modelos` / `MARMOVOC_MODELOS` e não baixa nada.

Para desenvolver (na pasta do repositório):

```powershell
pip install -e ".[classificacao,dev]"
python -m pytest
```

## Linha de comando

```powershell
# Recorta e classifica (um WAV, vários, ou uma pasta inteira)
marmovoc segmentar Sessao_SOM.wav --saida Vocalizations_Extracted

# Só recorta, sem classificar (não precisa de torch)
marmovoc segmentar My_Audio_Files --saida Vocalizations_Extracted --sem-classificar

# durations.txt em cada pasta de blocos
marmovoc duracao Vocalizations_Extracted
```

Saída de `segmentar` (igual ao script original):

- `<saida>/<nome do wav>/<nome do wav>_block_001_0m9s-0m11s.wav` — um arquivo por bloco;
- `<saida>/vocalization_analysis_corrected.csv` — mesmas colunas de antes
  (`--csv` muda o caminho).

## Em Python

```python
from marmovoc import Classificador, ParametrosSegmentacao, processar_arquivos, segmentar

clf = Classificador.da_pasta()  # cache; baixa do Zenodo na 1ª vez
df = processar_arquivos(["Sessao_SOM.wav"], "Vocalizations_Extracted", clf)

# Ou só a segmentação, sobre um array já carregado:
audio_filtrado, blocos = segmentar(audio, sample_rate, ParametrosSegmentacao(merge_threshold=0.5))
```

## Testes

```powershell
python -m pytest
```

Usam sinais sintéticos, um classificador falso e um zip local no lugar do
Zenodo — não precisam do modelo, de torch nem de internet.

## Pontos de atenção

- **Taxa de amostragem.** O classificador foi treinado com áudio a
  **96 kHz** (`MelFilter(96000, ...)` em `_vendor/marmaudio/classifier.py`,
  banco Mel de 1 a 48 kHz). O áudio não é reamostrado, como no script
  original. O WAV do Marmosync é gravado na taxa padrão do microfone (veja
  `samplerate_hz` no `_SOM_INFO.csv`) — se for 44,1/48 kHz, a escala de
  frequência que o modelo vê fica deslocada.
- **Corte de confiança.** O pipeline mantém o corte único de 10% do script
  original. O artigo do MarmAudio usa cortes por tipo (Infant cry ≥ 0,5;
  Phee ≥ 0,7; Seep ≥ 0,86; Trill ≥ 0,86; Tsik ≥ 0,7; Twitter ≥ 0,7) e
  rotula o resto como "Vocalization".
- **Alinhamento com o experimento.** Os tempos dos blocos são relativos ao
  início do WAV. O WAV do Marmosync começa alguns segundos depois do vídeo;
  o atraso está em `inicio_wav_relativo_experimento_s` no `_SOM_INFO.csv`.

## Créditos e licenças

O classificador e o código em `src/marmovoc/_vendor/marmaudio` são do
projeto **MarmAudio**:

> Lamothe, C., Obliger-Debouche, M., Best, P. et al. *A large annotated
> dataset of vocalizations by common marmosets.* Scientific Data 12, 782
> (2025). https://doi.org/10.1038/s41597-025-04951-8

- Código (`_vendor/marmaudio`): BSD 3-clause — ver `LICENCE.txt` na pasta.
- Classificador treinado: baixado de https://doi.org/10.5281/zenodo.15017207
  (`Code_Usage.zip`), licença CC BY 4.0.
