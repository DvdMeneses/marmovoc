"""
Janela para processar WAVs fora do Marmosync: escolher arquivos ou uma
pasta, segmentar e (opcionalmente) classificar.

Substitui o `input()` + `filedialog` do classification_script.py. Instalado
com o pacote como `marmovoc-gui` (no Windows, `Scripts\\marmovoc-gui.exe`).
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
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

NOME_PASTA_SAIDA = "Vocalizations_Extracted"


class JanelaMarmovoc:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("marmovoc — vocalizações")
        self.root.geometry("760x560")
        self.root.minsize(620, 460)

        self._arquivos: List[str] = []
        self._fila: "queue.Queue[tuple]" = queue.Queue()
        self._processando = False
        self._ultima_saida: Optional[str] = None

        self._montar()
        self.root.after(100, self._drenar_fila)

    # -- layout ----------------------------------------------------------

    def _montar(self) -> None:
        pad = {"padx": 8, "pady": 4}
        frame = ttk.Frame(self.root, padding=8)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Áudios").grid(row=0, column=0, sticky="nw", **pad)
        lista_frame = ttk.Frame(frame)
        lista_frame.grid(row=0, column=1, sticky="nsew", **pad)
        lista_frame.columnconfigure(0, weight=1)
        lista_frame.rowconfigure(0, weight=1)
        self.lista = tk.Listbox(lista_frame, height=8, selectmode="extended")
        self.lista.grid(row=0, column=0, sticky="nsew")
        barra = ttk.Scrollbar(lista_frame, orient="vertical", command=self.lista.yview)
        barra.grid(row=0, column=1, sticky="ns")
        self.lista.config(yscrollcommand=barra.set)

        botoes = ttk.Frame(frame)
        botoes.grid(row=0, column=2, sticky="n", **pad)
        ttk.Button(botoes, text="Adicionar arquivos...", command=self._adicionar_arquivos).pack(fill="x", pady=2)
        ttk.Button(botoes, text="Adicionar pasta...", command=self._adicionar_pasta).pack(fill="x", pady=2)
        ttk.Button(botoes, text="Remover selecionados", command=self._remover).pack(fill="x", pady=2)
        ttk.Button(botoes, text="Limpar", command=self._limpar).pack(fill="x", pady=2)

        ttk.Label(frame, text="Pasta de saída").grid(row=1, column=0, sticky="w", **pad)
        self.saida_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.saida_var).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Escolher...", command=self._escolher_saida).grid(row=1, column=2, sticky="ew", **pad)

        self.classificar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame, text="Classificar os blocos (Phee, Twitter, ...)", variable=self.classificar_var,
            command=self._atualizar_modelos,
        ).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(frame, text="Pasta do modelo").grid(row=3, column=0, sticky="w", **pad)
        self.modelos_var = tk.StringVar(value=os.environ.get("MARMOVOC_MODELOS", ""))
        self.modelos_entry = ttk.Entry(frame, textvariable=self.modelos_var)
        self.modelos_entry.grid(row=3, column=1, sticky="ew", **pad)
        self.modelos_botao = ttk.Button(frame, text="Escolher...", command=self._escolher_modelos)
        self.modelos_botao.grid(row=3, column=2, sticky="ew", **pad)
        ttk.Label(
            frame, text="Vazio = baixa do Zenodo na primeira vez e guarda em cache.", foreground="#666666"
        ).grid(row=4, column=1, sticky="w", padx=8)

        acoes = ttk.Frame(frame)
        acoes.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        acoes.columnconfigure(0, weight=1)
        self.progresso = ttk.Progressbar(acoes, mode="determinate")
        self.progresso.grid(row=0, column=0, sticky="ew", padx=8)
        self.botao_abrir = ttk.Button(acoes, text="Abrir pasta de saída", command=self._abrir_saida, state="disabled")
        self.botao_abrir.grid(row=0, column=1, padx=4)
        self.botao_processar = ttk.Button(acoes, text="▶  Processar", command=self._processar)
        self.botao_processar.grid(row=0, column=2, padx=4)

        frame.rowconfigure(6, weight=1)
        self.log = tk.Text(frame, height=10, state="disabled", wrap="word", font=("Consolas", 9))
        self.log.grid(row=6, column=0, columnspan=3, sticky="nsew", **pad)

    # -- seleção ---------------------------------------------------------

    def _adicionar(self, caminhos) -> None:
        for caminho in caminhos:
            caminho = os.path.normpath(caminho)
            if caminho not in self._arquivos:
                self._arquivos.append(caminho)
                self.lista.insert("end", caminho)
        if self._arquivos and not self.saida_var.get():
            self.saida_var.set(os.path.join(os.path.dirname(self._arquivos[0]), NOME_PASTA_SAIDA))

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
        for idx in reversed(self.lista.curselection()):
            self.lista.delete(idx)
            del self._arquivos[idx]

    def _limpar(self) -> None:
        self.lista.delete(0, "end")
        self._arquivos.clear()

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
        self.modelos_entry.config(state=estado)
        self.modelos_botao.config(state=estado)

    def _abrir_saida(self) -> None:
        if self._ultima_saida and os.path.isdir(self._ultima_saida):
            os.startfile(self._ultima_saida)  # type: ignore[attr-defined]  # só existe no Windows

    # -- processamento ---------------------------------------------------

    def _escrever(self, texto: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", texto + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

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

        self._processando = True
        self.botao_processar.config(state="disabled")
        self.botao_abrir.config(state="disabled")
        self.progresso.config(maximum=len(self._arquivos), value=0)

        threading.Thread(
            target=self._rodar,
            args=(list(self._arquivos), saida, self.classificar_var.get(), self.modelos_var.get().strip() or None),
            daemon=True,
        ).start()

    def _rodar(self, arquivos: List[str], saida: str, classificar: bool, pasta_modelos: Optional[str]) -> None:
        """Thread de fundo: só fala com a UI pela fila."""

        enviar = lambda *msg: self._fila.put(msg)
        try:
            import pandas as pd

            from marmovoc.pipeline import COLUNAS, processar_arquivo

            os.makedirs(saida, exist_ok=True)

            classificador = None
            if classificar:
                enviar("log", "Carregando o classificador (na 1ª vez baixa ~89 MB do Zenodo)...")
                from marmovoc.classificacao import Classificador

                classificador = Classificador.da_pasta(pasta_modelos)

            linhas = []
            for i, caminho in enumerate(arquivos, start=1):
                enviar("log", f"[{i}/{len(arquivos)}] {os.path.basename(caminho)}")
                try:
                    resultado = processar_arquivo(caminho, saida, classificador)
                    linhas.extend(resultado)
                    enviar("log", f"    {len(resultado)} bloco(s)")
                except Exception as e:
                    enviar("log", f"    ERRO: {e}")
                enviar("progresso", i)

            caminho_csv = os.path.join(saida, "vocalization_analysis_corrected.csv")
            df = pd.DataFrame(linhas, columns=COLUNAS)
            df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")

            enviar("log", f"\nTotal: {len(df)} bloco(s). CSV: {caminho_csv}")
            if classificador is not None and len(df):
                for rotulo, grupo in df.groupby("predicted_label"):
                    enviar("log", f"  {rotulo}: {len(grupo)} ({grupo['confidence_percent'].mean():.1f}% conf. média)")
            enviar("fim", saida, None)
        except Exception as e:
            enviar("log", traceback.format_exc())
            enviar("fim", saida, str(e))

    def _drenar_fila(self) -> None:
        try:
            while True:
                msg = self._fila.get_nowait()
                if msg[0] == "log":
                    self._escrever(msg[1])
                elif msg[0] == "progresso":
                    self.progresso.config(value=msg[1])
                elif msg[0] == "fim":
                    self._processando = False
                    self._ultima_saida = msg[1]
                    self.botao_processar.config(state="normal")
                    self.botao_abrir.config(state="normal")
                    if msg[2]:
                        messagebox.showerror("marmovoc", f"Falhou:\n{msg[2]}")
                    else:
                        messagebox.showinfo("marmovoc", "Processamento concluído.")
        except queue.Empty:
            pass
        self.root.after(100, self._drenar_fila)


def main() -> int:
    root = tk.Tk()
    JanelaMarmovoc(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
