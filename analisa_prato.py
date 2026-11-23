"""Analisa um prato com vários alimentos usando janela deslizante.

1. Segmenta o prato e a comida (segmentacao.py).
2. Passa uma janela (25% do diâmetro do prato, passo de 1/4 da janela) sobre a comida.
3. A CNN classifica cada janela. A probabilidade de cada janela é espalhada nos
   pixels dela com peso gaussiano (o centro da janela vale mais que a borda).
4. Cada pixel de comida recebe a classe de maior probabilidade acumulada.
   Se a confiança for baixa, vira "incerto".
5. Lista os alimentos que ocupam pelo menos --area-min da comida.

Uso:
  python analisa_prato.py dados/Imagens_Varios_Alimentos/prato_2_1.jpg
  python analisa_prato.py pasta_com_pratos/ --modelo resultados/com_aug/modelo.pt --saida pratos_saida/

Com um modelo v2 (treinado com --classe-prato) a janela passa no prato inteiro
e o que a CNN chamar de "Prato" vira fundo, sem depender da segmentação clássica.
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from recortes import recortar, centros_grade, para_tensor
from segmentacao import carregar, segmentar, diametro
from treino import criar_modelo, dispositivo

# tab20 sem os dois cinzas (índices 14 e 15), pra não confundir com "incerto"
CORES = np.array([c for i, c in enumerate(plt.get_cmap("tab20").colors) if i not in (14, 15)])


def carregar_modelo(caminho, dev):
    ck = torch.load(caminho, map_location=dev, weights_only=True)
    modelo = criar_modelo(len(ck["classes"]))
    modelo.load_state_dict(ck["estado"])
    return modelo.to(dev).eval(), ck["classes"]


@torch.no_grad()
def analisar(rgb, modelo, classes, dev, fracao=0.25, passo=0.25, conf_min=0.5, area_min=0.04):
    """v1 (modelo sem classe Prato): janelas só sobre a comida da segmentação clássica.
    v2 (modelo com classe Prato): janelas no prato inteiro; comida = segmentação clássica
    menos o que a CNN classificar como Prato."""
    disco, comida_classica = segmentar(rgb)
    d = diametro(disco)
    v2 = "Prato" in classes
    if v2:
        r = max(3, int(0.03 * d / 2))
        regiao = cv2.erode(disco, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    else:
        regiao = comida_classica
    centros, lado = centros_grade(regiao, d, fracao=fracao, passo=passo, cobertura_min=0.2)
    patches = [recortar(rgb, cx, cy, lado) for cx, cy in centros]
    probs = []
    for i in range(0, len(patches), 256):
        x = torch.from_numpy(para_tensor(patches[i:i + 256])).to(dev)
        probs.append(torch.softmax(modelo(x), 1).cpu().numpy())
    probs = np.concatenate(probs) if probs else np.zeros((0, len(classes)))

    h, w = regiao.shape
    acum = np.zeros((h, w, len(classes)), np.float32)
    peso = np.zeros((h, w), np.float32)
    meio = int(lado / 2)
    g1 = cv2.getGaussianKernel(2 * meio, sigma=meio / 1.5)
    gauss = (g1 @ g1.T).astype(np.float32)
    gauss /= gauss.max()
    for (cx, cy), p in zip(centros, probs):
        y0, x0 = cy - meio, cx - meio
        ys, xs = slice(max(y0, 0), min(y0 + 2 * meio, h)), slice(max(x0, 0), min(x0 + 2 * meio, w))
        gk = gauss[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
        acum[ys, xs] += gk[:, :, None] * p[None, None, :]
        peso[ys, xs] += gk

    prob_pixel = acum / np.maximum(peso, 1e-6)[:, :, None]
    rotulo = prob_pixel.argmax(2)
    conf = prob_pixel.max(2)
    if v2:
        # híbrido: a borda vem da segmentação clássica (precisa no contorno) e a
        # CNN veta o que ela acha que é prato vazio (reflexos, manchas).
        # Só a CNN não basta: janela meio comida, meio prato vira comida, e a
        # classe "vaza" pro prato em volta.
        k_prato = classes.index("Prato")
        comida = ((comida_classica > 0) & (peso > 1e-3) & (rotulo != k_prato)).astype(np.uint8)
    else:
        comida = comida_classica
    mapa = np.full((h, w), -1, np.int32)                # -1 = fundo
    mapa[comida > 0] = rotulo[comida > 0]
    mapa[(comida > 0) & ((conf < conf_min) | (peso < 1e-3))] = -2   # -2 = incerto

    area_total = max(int((comida > 0).sum()), 1)
    encontrados = []
    for k, c in enumerate(classes):
        if c == "Prato":
            continue
        frac = float((mapa == k).sum()) / area_total
        if frac >= area_min:
            conf_media = float(conf[mapa == k].mean())
            encontrados.append((c, frac, conf_media))
    encontrados.sort(key=lambda t: -t[1])
    incerto = float((mapa == -2).sum()) / area_total
    return {"disco": disco, "comida": comida, "comida_classica": comida_classica, "v2": v2, "mapa": mapa, "encontrados": encontrados,
            "incerto": incerto, "n_janelas": len(centros), "lado": lado, "centros": centros, "probs": probs,
            "conf": conf}


def figura(rgb, r, classes, caminho_saida, titulo=""):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    ax[0].imshow(rgb); ax[0].set_title("entrada")
    seg = rgb.copy()
    cc = r["comida_classica"]
    seg[cc == 0] = (seg[cc == 0] * 0.3).astype(np.uint8)
    ax[1].imshow(seg)
    cont, _ = cv2.findContours(r["disco"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cont:
        ax[1].plot(c[:, 0, 0], c[:, 0, 1], "c-", lw=1)
    ax[1].set_title(f"segmentação clássica ({r['n_janelas']} janelas)")
    cor = rgb.astype(np.float32) / 255 * 0.35
    for k in range(len(classes)):
        cor[r["mapa"] == k] = 0.45 * rgb[r["mapa"] == k] / 255 + 0.55 * CORES[k]
    cor[r["mapa"] == -2] = 0.5 * rgb[r["mapa"] == -2] / 255 + 0.5 * np.array([0.5, 0.5, 0.5])
    ax[2].imshow(np.clip(cor, 0, 1))
    ax[2].set_title("alimentos encontrados" + (" (v2: CNN veta o fundo)" if r["v2"] else ""))
    alcas = [plt.Rectangle((0, 0), 1, 1, color=CORES[classes.index(c)]) for c, _, _ in r["encontrados"]]
    nomes = [f"{c} {100 * f:.0f}%" for c, f, _ in r["encontrados"]]
    if r["incerto"] > 0.01:
        alcas.append(plt.Rectangle((0, 0), 1, 1, color=(0.5, 0.5, 0.5)))
        nomes.append(f"incerto {100 * r['incerto']:.0f}%")
    ax[2].legend(alcas, nomes, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9, frameon=False)
    for a in ax:
        a.axis("off")
    if titulo:
        fig.suptitle(titulo)
    fig.tight_layout()
    if caminho_saida is None:
        return fig
    fig.savefig(caminho_saida, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada", help="foto ou pasta de fotos")
    ap.add_argument("--modelo", default="resultados/com_aug/modelo.pt")
    ap.add_argument("--saida", default="saida_pratos")
    ap.add_argument("--conf-min", type=float, default=0.5)
    ap.add_argument("--area-min", type=float, default=0.04)
    args = ap.parse_args()

    dev = dispositivo()
    modelo, classes = carregar_modelo(args.modelo, dev)
    if os.path.isdir(args.entrada):
        fotos = sorted(os.path.join(args.entrada, f) for f in os.listdir(args.entrada) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    else:
        fotos = [args.entrada]
    os.makedirs(args.saida, exist_ok=True)
    for f in fotos:
        rgb = carregar(f)
        r = analisar(rgb, modelo, classes, dev, conf_min=args.conf_min, area_min=args.area_min)
        nome = os.path.splitext(os.path.basename(f))[0]
        figura(rgb, r, classes, os.path.join(args.saida, nome + ".png"), titulo=nome)
        lista = ", ".join(f"{c} ({100 * fr:.0f}%)" for c, fr, _ in r["encontrados"])
        print(f"{nome}: {lista}" + (f" | incerto {100 * r['incerto']:.0f}%" if r["incerto"] > 0.01 else ""), flush=True)


if __name__ == "__main__":
    main()
