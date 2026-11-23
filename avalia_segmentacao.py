"""Compara a segmentação clássica com as máscaras feitas à mão (contorna.py).

Uso: python avalia_segmentacao.py
Saídas: resultados/segmentacao/{por_foto.csv, resumo.txt} e figuras/segmentacao_iou.png

Para cada foto calcula IoU, precisão e recall de pixel da máscara de comida:
  - fotos de um alimento: o mesmo pipeline do treino (segmentar + tira_reflexos + so_principal)
  - pratos: o mesmo pipeline da análise (segmentar)
Compara os limiares finais (8 e 3) com os da primeira versão (14 e 9).
"""
import argparse
import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from segmentacao import carregar, segmentar, so_principal, tira_reflexos

LIMIARES = {"final (8, 3)": (8.0, 3.0), "inicial (14, 9)": (14.0, 9.0)}


def mascara_auto(rgb, tipo, lim_cor, lim_tex):
    disco, comida = segmentar(rgb, lim_cor=lim_cor, lim_textura=lim_tex)
    if tipo == "um":
        comida = so_principal(tira_reflexos(rgb, disco, comida))
    return comida > 0


def metricas(auto, verdade):
    vp = (auto & verdade).sum(); fp = (auto & ~verdade).sum(); fn = (~auto & verdade).sum()
    return {"iou": vp / max(vp + fp + fn, 1), "precisao": vp / max(vp + fp, 1), "recall": vp / max(vp + fn, 1)}


def sobreposicao(rgb, auto, verdade):
    """Verde = acerto, vermelho = sobrou (FP), azul = faltou (FN)."""
    img = rgb.astype(np.float32).copy()
    for m, cor in ((auto & verdade, (0, 200, 0)), (auto & ~verdade, (230, 0, 0)), (~auto & verdade, (0, 90, 255))):
        img[m] = 0.45 * img[m] + 0.55 * np.array(cor)
    return img.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mascaras", default="rotulos/mascaras")
    ap.add_argument("--saida", default="resultados/segmentacao")
    ap.add_argument("--figura", default="figuras/segmentacao_iou.png")
    args = ap.parse_args()
    os.makedirs(args.saida, exist_ok=True)

    with open(os.path.join(args.mascaras, "lista.csv")) as fh:
        lista = [l for l in csv.DictReader(fh) if os.path.exists(os.path.join(args.mascaras, l["id"] + ".png"))]
    if not lista:
        print("nenhuma máscara pronta em", args.mascaras); return

    linhas, exemplos = [], []
    for l in lista:
        rgb = carregar(l["caminho"])
        verdade = cv2.imread(os.path.join(args.mascaras, l["id"] + ".png"), 0) > 0
        for nome, (lc, lt) in LIMIARES.items():
            auto = mascara_auto(rgb, l["tipo"], lc, lt)
            linhas.append({"id": l["id"], "tipo": l["tipo"], "limiares": nome,
                           **{k: round(float(v), 4) for k, v in metricas(auto, verdade).items()}})
            if nome.startswith("final"):
                exemplos.append((l["id"], linhas[-1]["iou"], sobreposicao(rgb, auto, verdade)))

    with open(os.path.join(args.saida, "por_foto.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)

    txt = [f"Segmentação contra máscara manual ({len(lista)} fotos). Média ± desvio padrão.", ""]
    for nome in LIMIARES:
        for tipo, rotulo in (("um", "um alimento"), ("prato", "pratos"), (None, "todas")):
            sel = [x for x in linhas if x["limiares"] == nome and (tipo is None or x["tipo"] == tipo)]
            if not sel:
                continue
            cel = "  ".join(f"{k} {np.mean([x[k] for x in sel]):.3f} ± {np.std([x[k] for x in sel], ddof=1) if len(sel) > 1 else 0:.3f}"
                            for k in ("iou", "precisao", "recall"))
            txt.append(f"{nome:<16} {rotulo:<12} n={len(sel):<3} {cel}")
        txt.append("")
    piores = sorted([x for x in linhas if x["limiares"].startswith("final")], key=lambda x: x["iou"])[:3]
    txt.append("Piores (final): " + ", ".join(f"{x['id']} {x['iou']:.2f}" for x in piores))
    open(os.path.join(args.saida, "resumo.txt"), "w").write("\n".join(txt) + "\n")
    print("\n".join(txt))

    # figura: as 2 piores e as 2 melhores, pra ser justa
    exemplos.sort(key=lambda e: e[1])
    nomes = {}
    n_prato = 0
    for l in lista:
        if l["tipo"] == "um":
            nomes[l["id"]] = l["id"][3:]
        else:
            n_prato += 1; nomes[l["id"]] = f"Prato {n_prato}"
    escolhidos = exemplos[:2] + exemplos[-2:] if len(exemplos) >= 4 else exemplos
    fig, axs = plt.subplots(1, len(escolhidos), figsize=(3.6 * len(escolhidos), 3.8))
    for ax, (i, iou, img) in zip(np.atleast_1d(axs), escolhidos):
        ax.imshow(img); ax.axis("off"); ax.set_title(f"{nomes[i]}\nIoU {iou:.2f}", fontsize=11)
    fig.text(0.5, 0.01, "verde: acerto    vermelho: marcou a mais    azul: faltou", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.figura, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("figura:", args.figura)


if __name__ == "__main__":
    main()
