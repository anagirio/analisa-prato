"""Demo pra apresentação: analisa pratos e mostra o resultado numa janela.

Uso:
  python demo.py dados/Imagens_Varios_Alimentos/prato_2_1.jpg     # uma foto
  python demo.py amostra_pratos/                                    # várias, fecha a janela pra ir pra próxima
  python demo.py --vigiar ~/Desktop/demo_fotos                      # espera fotos novas chegarem (AirDrop, etc.)

Opções: --modelo (padrão resultados/v2_mix_aug/modelo.pt)
"""
import argparse
import os
import time

import matplotlib.pyplot as plt

from analisa_prato import analisar, carregar_modelo, figura
from segmentacao import carregar
from treino import dispositivo

EXTENSOES = (".jpg", ".jpeg", ".png", ".heic")


def mostrar(caminho, modelo, classes, dev):
    t0 = time.time()
    rgb = carregar(caminho)
    r = analisar(rgb, modelo, classes, dev)
    dt = time.time() - t0
    lista = ", ".join(f"{c} ({100 * f:.0f}%)" for c, f, _ in r["encontrados"]) or "nada encontrado"
    print(f"\n{os.path.basename(caminho)}  [{dt:.1f} s, {r['n_janelas']} janelas]\n  -> {lista}", flush=True)
    fig = figura(rgb, r, classes, None, titulo=f"{os.path.basename(caminho)}: {lista}")
    fig.canvas.manager.set_window_title("Analisa prato")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada", nargs="?", help="foto ou pasta")
    ap.add_argument("--modelo", default="resultados/v2_mix_aug/modelo.pt")
    ap.add_argument("--vigiar", help="pasta pra vigiar: cada foto nova é analisada")
    args = ap.parse_args()

    plt.switch_backend("macosx" if os.uname().sysname == "Darwin" else "TkAgg")
    dev = dispositivo()
    modelo, classes = carregar_modelo(args.modelo, dev)
    print(f"modelo {args.modelo} ({len(classes)} classes) em {dev}")

    if args.vigiar:
        os.makedirs(args.vigiar, exist_ok=True)
        vistos = set(os.listdir(args.vigiar))
        print(f"vigiando {args.vigiar} ... (Ctrl+C pra sair)")
        while True:
            novos = sorted(set(os.listdir(args.vigiar)) - vistos)
            for f in novos:
                vistos.add(f)
                if f.lower().endswith(EXTENSOES):
                    time.sleep(0.5)  # espera terminar de copiar
                    mostrar(os.path.join(args.vigiar, f), modelo, classes, dev)
            time.sleep(1)
    elif args.entrada and os.path.isdir(args.entrada):
        for f in sorted(os.listdir(args.entrada)):
            if f.lower().endswith(EXTENSOES):
                mostrar(os.path.join(args.entrada, f), modelo, classes, dev)
    elif args.entrada:
        mostrar(args.entrada, modelo, classes, dev)
    else:
        ap.error("passe uma foto, uma pasta ou --vigiar PASTA")


if __name__ == "__main__":
    main()
