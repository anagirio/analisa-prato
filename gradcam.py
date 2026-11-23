"""Grad-CAM: mapa de calor de onde a CNN "olhou" pra decidir a classe.

Como funciona (Selvaraju et al., 2017):
1. Guarda a saída A da última camada convolucional (layer4 da ResNet18: 512 mapas 5x5
   pra uma entrada 160x160).
2. Calcula o gradiente da nota da classe escolhida em relação a A.
3. A média do gradiente em cada mapa vira o peso daquele mapa.
4. Soma ponderada dos mapas, ReLU (só o que puxa a favor), redimensiona pra imagem.

Uso:
  python gradcam.py --modelo resultados/v2_com_aug/modelo.pt          # figura com fotos do teste
  python gradcam.py --modelo ... --foto dados/Imagens_um_Alimento/Tomate/IMG_....jpg
"""
import argparse
import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analisa_prato import carregar_modelo
from recortes import recortar, para_tensor, FRACAO_PADRAO
from segmentacao import carregar, segmentar, so_principal, tira_reflexos, diametro, caixa
from treino import dispositivo


class GradCAM:
    def __init__(self, modelo, camada):
        self.modelo = modelo
        self.ativ, self.grad = None, None
        camada.register_forward_hook(lambda m, i, o: setattr(self, "ativ", o))
        camada.register_full_backward_hook(lambda m, gi, go: setattr(self, "grad", go[0]))

    def __call__(self, x, classe=None):
        self.modelo.eval()
        x = x.clone().requires_grad_(True)
        saida = self.modelo(x)
        prob = torch.softmax(saida, 1)[0]
        if classe is None:
            classe = int(saida.argmax(1))
        self.modelo.zero_grad()
        saida[0, classe].backward()
        pesos = self.grad[0].mean(dim=(1, 2))                 # 1 peso por mapa
        cam = torch.relu((pesos[:, None, None] * self.ativ[0]).sum(0))
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), classe, float(prob[classe].detach())


def sobrepor(rgb, cam):
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    calor = plt.get_cmap("jet")(cam)[:, :, :3]
    return np.clip(0.55 * rgb / 255 + 0.45 * calor, 0, 1)


def recorte_da_foto(caminho):
    """Recorte centrado na comida de uma foto de um alimento só (como na avaliação)."""
    rgb = carregar(caminho)
    disco, comida = segmentar(rgb)
    comida = so_principal(tira_reflexos(rgb, disco, comida))
    ys, xs = np.nonzero(comida if comida.sum() else disco)
    return recortar(rgb, xs.mean(), ys.mean(), diametro(disco) * FRACAO_PADRAO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="resultados/v2_com_aug/modelo.pt")
    ap.add_argument("--foto", nargs="*", help="fotos de um alimento só (padrão: 1 por classe do teste)")
    ap.add_argument("--saida", default="figuras/gradcam.png")
    args = ap.parse_args()

    dev = torch.device("cpu")   # backward hooks são mais estáveis na CPU, e é rápido
    modelo, classes = carregar_modelo(args.modelo, dev)
    gc = GradCAM(modelo, modelo.layer4)

    fotos = args.foto
    if not fotos:
        with open("divisao.csv") as fh:
            linhas = [l for l in csv.DictReader(fh) if l["conjunto"] == "teste"]
        vistos, fotos = set(), []
        for l in linhas:
            if l["classe"] not in vistos:
                vistos.add(l["classe"])
                fotos.append(l["caminho"])

    n = len(fotos)
    col = min(n, 8)
    lin = 2 * int(np.ceil(n / col))
    fig, ax = plt.subplots(lin, col, figsize=(2 * col, 2.1 * lin))
    ax = np.atleast_2d(ax)
    for a in ax.ravel():
        a.axis("off")
    for k, f in enumerate(fotos):
        p = recorte_da_foto(f)
        cam, c, pr = gc(torch.from_numpy(para_tensor([p])))
        i, j = 2 * (k // col), k % col
        ax[i, j].imshow(p)
        ax[i, j].set_title(os.path.basename(os.path.dirname(f)) or os.path.basename(f), fontsize=8)
        ax[i + 1, j].imshow(sobrepor(p, cam))
        ax[i + 1, j].set_title(f"{classes[c]} {100 * pr:.0f}%", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
    fig.savefig(args.saida, dpi=130)
    print("salvo em", args.saida)


if __name__ == "__main__":
    main()
