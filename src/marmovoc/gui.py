"""
Janela para processar WAVs fora do Marmosync: escolher arquivos ou uma
pasta, segmentar e (opcionalmente) classificar.

Substitui o `input()` + `filedialog` do classification_script.py. Instalado
com o pacote como `marmovoc-gui` (no Windows, `Scripts\\marmovoc-gui.exe`).
Visual: ttkbootstrap com a mesma paleta teal do Marmosync (`setup_tema`).

O processamento roda numa thread de fundo; a thread do Tk só lê a fila de
mensagens (`_drenar_fila`) — nenhum widget é tocado de outra thread.
"""

from __future__ import annotations

import glob
import os
import queue
import sys
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
from typing import Dict, List, Optional

import soundfile as sf
import ttkbootstrap as tb
from ttkbootstrap.style import Style
from ttkbootstrap.style.theme import Theme

from marmovoc.classificacao import SAMPLE_RATE_MODELO
from marmovoc.pipeline import LIMITE_CAMINHO_WINDOWS
from marmovoc.qualidade import PLANURA_MAXIMA

NOME_PASTA_SAIDA = "Vocalizations_Extracted"

# Mesma paleta do Marmosync (marmosync/ui/styles.py + Settings.color_*).
COR_PRIMARIA = "#006666"
COR_SECUNDARIA = "#009999"
COR_TEXTO = "#333333"
COR_SUAVE = "#6c757d"
FONTE = "Segoe UI"


def setup_tema() -> Style:
    style = Style()
    tema = Theme(
        name="marmovoc",
        primary=COR_PRIMARIA,
        secondary=COR_SECUNDARIA,
        success="#198754",
        info="#0dcaf0",
        warning="#ffc107",
        danger="#dc3545",
        neutral="#adb5bd",
        light={"background": "#ffffff", "foreground": COR_TEXTO},
    )
    for definicao in tema.to_definitions():
        style.register_theme(definicao)
    style.theme_use("marmovoc-light")

    tkfont.nametofont("TkDefaultFont").configure(family=FONTE, size=10)
    style.configure("TLabel", font=(FONTE, 10))
    style.configure("TEntry", padding=6)
    style.configure("TButton", font=(FONTE, 10, "bold"), padding=(12, 7))
    style.configure("TCheckbutton", font=(FONTE, 10))
    style.configure("TLabelframe", padding=12)
    style.configure("TLabelframe.Label", font=(FONTE, 10, "bold"))
    style.configure("Treeview", rowheight=28, font=(FONTE, 9))
    style.configure("Treeview.Heading", font=(FONTE, 9, "bold"))
    return style


def _mmss(segundos: float) -> str:
    segundos = int(round(segundos))
    return f"{segundos // 60}:{segundos % 60:02d}"


class JanelaMarmovoc:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("marmovoc — vocalizações de saguis")
        self.root.geometry("900x910")
        self.root.minsize(760, 790)

        self._arquivos: List[str] = []
        self._duracoes: Dict[str, float] = {}
        self._fila: "queue.Queue[tuple]" = queue.Queue()
        self._processando = False
        self._ultima_saida: Optional[str] = None
        self._log_visivel = False

        self._montar()
        self._atualizar_resumo()
        self.root.after(100, self._drenar_fila)

    # -- layout ----------------------------------------------------------

    def _montar(self) -> None:
        cabecalho = tb.Frame(self.root, bootstyle="primary", padding=(20, 14))
        cabecalho.pack(fill="x")
        tb.Label(
            cabecalho, text="marmovoc", font=(FONTE, 18, "bold"), bootstyle="inverse-primary"
        ).pack(anchor="w")
        tb.Label(
            cabecalho,
            text="Segmentação e classificação de vocalizações a partir das gravações do experimento",
            font=(FONTE, 10),
            bootstyle="inverse-primary",
        ).pack(anchor="w")

        corpo = tb.Frame(self.root, padding=(20, 14))
        corpo.pack(fill="both", expand=True)

        # Rodapé empacotado primeiro, na base: fica sempre visível e a
        # tabela de áudios só ocupa o espaço que sobrar.
        inferior = tb.Frame(corpo)
        inferior.pack(side="bottom", fill="x")
        self._montar_rodape(inferior)

        self._montar_audios(corpo)
        self._montar_saida(corpo)
        self._montar_classificacao(corpo)

    def _montar_audios(self, pai) -> None:
        card = tb.Labelframe(pai, text="1   Áudios", bootstyle="primary")
        card.pack(fill="both", expand=True)

        barra = tb.Frame(card)
        barra.pack(fill="x", pady=(0, 8))
        tb.Button(barra, text="＋  Arquivos", bootstyle="primary", command=self._adicionar_arquivos).pack(side="left")
        tb.Button(barra, text="＋  Pasta", bootstyle="primary-outline", command=self._adicionar_pasta).pack(
            side="left", padx=6
        )
        tb.Button(barra, text="Limpar", bootstyle="secondary-link", command=self._limpar).pack(side="right")
        tb.Button(barra, text="Remover selecionados", bootstyle="secondary-link", command=self._remover).pack(
            side="right"
        )

        tabela = tb.Frame(card)
        tabela.pack(fill="both", expand=True)
        colunas = ("arquivo", "duracao", "taxa", "status")
        self.tabela = tb.Treeview(tabela, columns=colunas, show="headings", selectmode="extended", height=5)
        for coluna, titulo, largura, ancora in (
            ("arquivo", "Arquivo", 420, "w"),
            ("duracao", "Duração", 80, "center"),
            ("taxa", "Taxa", 100, "center"),
            ("status", "Status", 140, "center"),
        ):
            self.tabela.heading(coluna, text=titulo, anchor=ancora)
            self.tabela.column(coluna, width=largura, anchor=ancora, stretch=(coluna == "arquivo"))
        self.tabela.tag_configure("ok", foreground="#198754")
        self.tabela.tag_configure("erro", foreground="#dc3545")
        self.tabela.tag_configure("ativo", foreground=COR_PRIMARIA)
        self.tabela.pack(side="left", fill="both", expand=True)
        rolagem = tb.Scrollbar(tabela, orient="vertical", command=self.tabela.yview)
        rolagem.pack(side="right", fill="y")
        self.tabela.configure(yscrollcommand=rolagem.set)

        self.resumo_var = tk.StringVar()
        tb.Label(card, textvariable=self.resumo_var, foreground=COR_SUAVE, font=(FONTE, 9)).pack(
            anchor="w", pady=(6, 0)
        )

    def _montar_saida(self, pai) -> None:
        card = tb.Labelframe(pai, text="2   Pasta de saída", bootstyle="primary")
        card.pack(fill="x", pady=(12, 0))
        card.columnconfigure(0, weight=1)

        self.saida_var = tk.StringVar()
        tb.Entry(card, textvariable=self.saida_var).grid(row=0, column=0, sticky="ew")
        tb.Button(card, text="Escolher…", bootstyle="primary-outline", command=self._escolher_saida).grid(
            row=0, column=1, padx=(8, 0)
        )
        tb.Label(
            card,
            text="Um WAV por bloco em <saída>\\<nome do áudio>\\ e o CSV vocalization_analysis_corrected.csv.",
            foreground=COR_SUAVE,
            font=(FONTE, 9),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.espectrogramas_var = tk.BooleanVar(value=True)
        tb.Checkbutton(
            card,
            text="Salvar espectrogramas (um PNG por bloco + visão geral da gravação) para conferir os blocos",
            variable=self.espectrogramas_var,
            bootstyle="primary",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.filtro_tonalidade_var = tk.BooleanVar(value=True)
        tb.Checkbutton(
            card,
            text="Descartar ruídos de banda larga (cliques, impactos, chiados) antes de classificar",
            variable=self.filtro_tonalidade_var,
            bootstyle="primary",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _montar_classificacao(self, pai) -> None:
        card = tb.Labelframe(pai, text="3   Classificação", bootstyle="primary")
        card.pack(fill="x", pady=(12, 0))
        card.columnconfigure(0, weight=1)

        self.classificar_var = tk.BooleanVar(value=True)
        tb.Checkbutton(
            card,
            text="Classificar cada bloco (Phee, Twitter, Trill, Tsik, Seep, Infant cry)",
            variable=self.classificar_var,
            bootstyle="primary",
            command=self._atualizar_modelos,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        self.modelos_var = tk.StringVar(value=os.environ.get("MARMOVOC_MODELOS", ""))
        self.modelos_entry = tb.Entry(card, textvariable=self.modelos_var)
        self.modelos_entry.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.modelos_botao = tb.Button(
            card, text="Modelo…", bootstyle="primary-outline", command=self._escolher_modelos
        )
        self.modelos_botao.grid(row=1, column=1, padx=(8, 0), pady=(8, 0))
        tb.Label(
            card,
            text="Vazio = usa o classificador do MarmAudio em cache (baixado do Zenodo na primeira vez).",
            foreground=COR_SUAVE,
            font=(FONTE, 9),
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _montar_rodape(self, pai) -> None:
        rodape = tb.Frame(pai)
        rodape.pack(fill="x", pady=(14, 0))
        rodape.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Pronto.")
        tb.Label(rodape, textvariable=self.status_var, font=(FONTE, 9, "bold")).grid(row=0, column=0, sticky="w")
        self.progresso = tb.Progressbar(rodape, mode="determinate", bootstyle="primary-striped")
        self.progresso.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self.botao_abrir = tb.Button(
            rodape, text="Abrir pasta", bootstyle="secondary-outline", command=self._abrir_saida, state="disabled"
        )
        self.botao_abrir.grid(row=0, column=1, rowspan=2, padx=(12, 6), sticky="ns")
        self.botao_processar = tb.Button(rodape, text="▶   Processar", bootstyle="primary", command=self._processar)
        self.botao_processar.grid(row=0, column=2, rowspan=2, sticky="ns")

        self.botao_log = tb.Button(
            pai, text="▸ Mostrar detalhes", bootstyle="secondary-link", command=self._alternar_log
        )
        self.botao_log.pack(anchor="w", pady=(6, 0))
        self.log = tb.Text(pai, height=6, state="disabled", wrap="word", font=("Consolas", 9))

    # -- seleção ---------------------------------------------------------

    def _adicionar(self, caminhos) -> None:
        for caminho in caminhos:
            caminho = os.path.normpath(caminho)
            if caminho in self._arquivos:
                continue
            duracao, taxa = None, None
            try:
                info = sf.info(caminho)
                duracao, taxa = info.duration, info.samplerate
            except Exception:
                pass
            self._arquivos.append(caminho)
            if duracao is not None:
                self._duracoes[caminho] = duracao
            texto_taxa = "?" if taxa is None else (f"{taxa / 1000:g} kHz" + ("" if taxa >= SAMPLE_RATE_MODELO else "  ⚠"))
            self.tabela.insert(
                "", "end", iid=caminho,
                values=(os.path.basename(caminho), "?" if duracao is None else _mmss(duracao), texto_taxa, "na fila"),
            )
        if self._arquivos and not self.saida_var.get():
            self.saida_var.set(os.path.join(os.path.dirname(self._arquivos[0]), NOME_PASTA_SAIDA))
        self._atualizar_resumo()

    def _atualizar_resumo(self) -> None:
        if not self._arquivos:
            self.resumo_var.set("Nenhum áudio. Adicione a gravação completa do experimento (…_SOM.wav).")
            return
        total = sum(self._duracoes.values())
        texto = f"{len(self._arquivos)} arquivo(s) · {_mmss(total)} de áudio"
        if any("⚠" in self.tabela.set(a, "taxa") for a in self._arquivos):
            texto += f"   ·   ⚠ abaixo de {SAMPLE_RATE_MODELO // 1000} kHz, a taxa em que o classificador foi treinado"
        self.resumo_var.set(texto)

    def _adicionar_arquivos(self) -> None:
        self._adicionar(
            filedialog.askopenfilenames(
                title="Selecione um ou mais arquivos de áudio",
                filetypes=[("Arquivos WAV", "*.wav"), ("Todos os arquivos", "*.*")],
            )
        )

    def _adicionar_pasta(self) -> None:
        pasta = filedialog.askdirectory(title="Selecione a pasta com os WAVs")
        if not pasta:
            return
        wavs = sorted(glob.glob(os.path.join(pasta, "*.wav")))
        if not wavs:
            messagebox.showwarning("marmovoc", f"Nenhum .wav em:\n{pasta}")
            return
        self._adicionar(wavs)

    def _remover(self) -> None:
        for iid in self.tabela.selection():
            self.tabela.delete(iid)
            self._arquivos.remove(iid)
            self._duracoes.pop(iid, None)
        self._atualizar_resumo()

    def _limpar(self) -> None:
        for iid in self.tabela.get_children():
            self.tabela.delete(iid)
        self._arquivos.clear()
        self._duracoes.clear()
        self._atualizar_resumo()

    def _escolher_saida(self) -> None:
        pasta = filedialog.askdirectory(title="Pasta de saída")
        if pasta:
            self.saida_var.set(os.path.normpath(pasta))

    def _escolher_modelos(self) -> None:
        pasta = filedialog.askdirectory(title="Pasta com o .stdc e o .tsv do classificador")
        if pasta:
            self.modelos_var.set(os.path.normpath(pasta))

    def _atualizar_modelos(self) -> None:
        estado = "normal" if self.classificar_var.get() else "disabled"
        self.modelos_entry.configure(state=estado)
        self.modelos_botao.configure(state=estado)

    def _abrir_saida(self) -> None:
        if self._ultima_saida and os.path.isdir(self._ultima_saida):
            os.startfile(self._ultima_saida)  # type: ignore[attr-defined]  # só existe no Windows

    def _alternar_log(self) -> None:
        self._log_visivel = not self._log_visivel
        if self._log_visivel:
            self.log.pack(fill="both", expand=False, pady=(4, 0))
            self.botao_log.configure(text="▾ Ocultar detalhes")
        else:
            self.log.pack_forget()
            self.botao_log.configure(text="▸ Mostrar detalhes")

    # -- processamento ---------------------------------------------------

    def _escrever(self, texto: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", texto + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _caminho_mais_longo(self, saida: str) -> str:
        """Pior caso do caminho de um bloco: <saida>\\<wav>\\<wav>_block_999_99m59s-99m59s.wav."""

        maior = max(self._arquivos, key=lambda a: len(os.path.basename(a)))
        base = os.path.splitext(os.path.basename(maior))[0]
        return os.path.join(os.path.abspath(saida), base, f"{base}_block_999_99m59s-99m59s.wav")

    def _processar(self) -> None:
        if self._processando:
            return
        if not self._arquivos:
            messagebox.showwarning("marmovoc", "Adicione pelo menos um arquivo WAV.")
            return
        saida = self.saida_var.get().strip()
        if not saida:
            messagebox.showwarning("marmovoc", "Escolha a pasta de saída.")
            return

        longo = self._caminho_mais_longo(saida)
        if os.name == "nt" and len(longo) >= LIMITE_CAMINHO_WINDOWS:
            messagebox.showerror(
                "marmovoc",
                f"A pasta de saída é funda demais: os blocos ficariam com até {len(longo)} "
                f"caracteres no caminho (o Windows aceita {LIMITE_CAMINHO_WINDOWS - 1}).\n\n"
                "Escolha uma pasta mais curta, por exemplo C:\\Users\\<você>\\Desktop\\saida.",
            )
            return

        if any("_block_" in os.path.basename(a) for a in self._arquivos) and not messagebox.askyesno(
            "marmovoc",
            "Alguns arquivos parecem ser blocos já recortados (\"_block_\" no nome).\n"
            "O marmovoc espera a gravação completa do experimento.\n\nProcessar mesmo assim?",
        ):
            return

        self._processando = True
        self.botao_processar.configure(state="disabled")
        self.botao_abrir.configure(state="disabled")
        self.progresso.configure(maximum=len(self._arquivos), value=0)
        for iid in self._arquivos:
            self.tabela.set(iid, "status", "na fila")
            self.tabela.item(iid, tags=())

        threading.Thread(
            target=self._rodar,
            args=(
                list(self._arquivos), saida, self.classificar_var.get(),
                self.modelos_var.get().strip() or None, self.espectrogramas_var.get(),
                self.filtro_tonalidade_var.get(),
            ),
            daemon=True,
        ).start()

    def _rodar(
        self, arquivos: List[str], saida: str, classificar: bool, pasta_modelos: Optional[str],
        espectrogramas: bool = False,
        filtro_tonalidade: bool = True,
    ) -> None:
        """Thread de fundo: só fala com a UI pela fila."""

        enviar = lambda *msg: self._fila.put(msg)
        try:
            import pandas as pd

            from marmovoc.pipeline import COLUNAS, processar_arquivo

            os.makedirs(saida, exist_ok=True)

            classificador = None
            if classificar:
                enviar("status", "Carregando o classificador… (na 1ª vez baixa ~89 MB do Zenodo)")
                from marmovoc.classificacao import Classificador

                classificador = Classificador.da_pasta(pasta_modelos)

            linhas = []
            for i, caminho in enumerate(arquivos, start=1):
                enviar("status", f"Processando {i} de {len(arquivos)}: {os.path.basename(caminho)}")
                enviar("arquivo", caminho, "processando…", "ativo")
                try:
                    resultado = processar_arquivo(
                        caminho, saida, classificador, espectrogramas=espectrogramas,
                        planura_maxima=PLANURA_MAXIMA if filtro_tonalidade else None,
                    )
                    linhas.extend(resultado)
                    enviar("arquivo", caminho, f"✓  {len(resultado)} bloco(s)", "ok")
                    enviar("log", f"{os.path.basename(caminho)}: {len(resultado)} bloco(s)")
                except Exception as e:
                    enviar("arquivo", caminho, "✗  erro", "erro")
                    enviar("log", f"{os.path.basename(caminho)}: ERRO — {e}")
                enviar("progresso", i)

            caminho_csv = os.path.join(saida, "vocalization_analysis_corrected.csv")
            df = pd.DataFrame(linhas, columns=COLUNAS)
            df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")

            enviar("log", f"\nTotal: {len(df)} bloco(s). CSV: {caminho_csv}")
            resumo = ""
            if classificador is not None and len(df):
                contagem = df["predicted_label"].value_counts()
                resumo = " · ".join(f"{rotulo} {n}" for rotulo, n in contagem.items())
                for rotulo, grupo in df.groupby("predicted_label"):
                    enviar("log", f"  {rotulo}: {len(grupo)} ({grupo['confidence_percent'].mean():.1f}% conf. média)")
            enviar("fim", saida, None, f"{len(df)} bloco(s)" + (f"  —  {resumo}" if resumo else ""))
        except Exception as e:
            enviar("log", traceback.format_exc())
            enviar("fim", saida, str(e), "")

    def _drenar_fila(self) -> None:
        try:
            while True:
                msg = self._fila.get_nowait()
                tipo = msg[0]
                if tipo == "log":
                    self._escrever(msg[1])
                elif tipo == "status":
                    self.status_var.set(msg[1])
                elif tipo == "arquivo":
                    _, iid, texto, tag = msg
                    if self.tabela.exists(iid):
                        self.tabela.set(iid, "status", texto)
                        self.tabela.item(iid, tags=(tag,))
                elif tipo == "progresso":
                    self.progresso.configure(value=msg[1])
                elif tipo == "fim":
                    _, saida, erro, resumo = msg
                    self._processando = False
                    self._ultima_saida = saida
                    self.botao_processar.configure(state="normal")
                    self.botao_abrir.configure(state="normal")
                    if erro:
                        self.status_var.set("Falhou — veja os detalhes.")
                        if not self._log_visivel:
                            self._alternar_log()
                        messagebox.showerror("marmovoc", f"Falhou:\n{erro}")
                    else:
                        self.status_var.set(f"Concluído: {resumo}")
        except queue.Empty:
            pass
        self.root.after(100, self._drenar_fila)


def main() -> int:
    root = tk.Tk()
    setup_tema()
    JanelaMarmovoc(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
