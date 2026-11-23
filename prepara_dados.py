"""Prepara o dataset: segmenta, salva em cache e divide treino/val/teste.

Por que dividir por rajada: as fotos foram tiradas em sequência, várias do mesmo
prato com poucos segundos de diferença. Se a divisão for aleatória por foto,
fotos quase iguais caem no treino e no teste e a acurácia fica inflada.
Aqui cada rajada (fotos com menos de 20 s entre si) vira um grupo, e o grupo
inteiro vai pra um conjunto só.

Saídas:
  cache/<classe>/<nome>.jpg      foto reduzida (lado maior 800)
  cache/<classe>/<nome>_m.png    máscara da comida (0/255)
  divisao.csv                    caminho, classe, grupo, conjunto, diametro_prato, area_comida

Uso: python prepara_dados.py [--dados dados/Imagens_um_Alimento]
"""
import argparse
import csv
import os
import random
import re

import cv2
import numpy as np

from segmentacao import carregar, segmentar, so_principal, tira_reflexos, diametro

FRACOES = {"teste": 0.15, "val": 0.15}


def grupos_da_classe(arquivos):
    """Agrupa arquivos em rajadas. IMG_AAAAMMDD_HHMMSS... por tempo;
    nomes tipo arroz_100_3.jpg pelo prefixo (mesma porção)."""
    com_tempo, outros = [], []
    for f in arquivos:
        m = re.match(r"IMG_(\d{8})_(\d{2})(\d{2})(\d{2})", f)
        if m:
            d, h, mi, s = m.groups()
            com_tempo.append((d, int(h) * 3600 + int(mi) * 60 + int(s), f))
        else:
            outros.append(f)
    com_tempo.sort()
    grupo_de, g, ant = {}, -1, None
    for d, t, f in com_tempo:
        if ant is None or d != ant[0] or t - ant[1] > 20:
            g += 1
        grupo_de[f] = f"r{g:02d}_{d}"
        ant = (d, t)
    for f in outros:
        grupo_de[f] = "p_" + re.sub(r"_\d+\.jpg$", "", f, flags=re.I)
    return grupo_de


def dividir(grupo_de, semente):
    """Sorteia grupos inteiros pra teste e val até ~15% das fotos cada."""
    tamanhos = {}
    for g in grupo_de.values():
        tamanhos[g] = tamanhos.get(g, 0) + 1
    gs = sorted(tamanhos)
    random.Random(semente).shuffle(gs)
    total = sum(tamanhos.values())
    conjunto_de, cont = {}, {"teste": 0, "val": 0}
    for g in gs:
        for nome in ("teste", "val"):
            if cont[nome] < FRACOES[nome] * total:
                conjunto_de[g] = nome
                cont[nome] += tamanhos[g]
                break
        else:
            conjunto_de[g] = "treino"
    return conjunto_de


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="dados/Imagens_um_Alimento")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--saida", default="divisao.csv")
    ap.add_argument("--semente", type=int, default=42)
    args = ap.parse_args()

    linhas = []
    for ci, classe in enumerate(sorted(os.listdir(args.dados))):
        pasta = os.path.join(args.dados, classe)
        if not os.path.isdir(pasta):
            continue
        arquivos = sorted(f for f in os.listdir(pasta) if f.lower().endswith(".jpg"))
        grupo_de = grupos_da_classe(arquivos)
        conjunto_de = dividir(grupo_de, args.semente + ci)
        os.makedirs(os.path.join(args.cache, classe), exist_ok=True)
        for f in arquivos:
            nome = os.path.splitext(f)[0]
            destino = os.path.join(args.cache, classe, nome + ".jpg")
            rgb = carregar(os.path.join(pasta, f))
            disco, comida = segmentar(rgb)
            comida = so_principal(tira_reflexos(rgb, disco, comida))
            segm_ok = comida.sum() >= 0.01 * disco.sum()
            if not segm_ok:
                # plano B (fotos desfocadas de alimento branco): círculo no centro do prato
                ys, xs = np.nonzero(disco)
                comida = np.zeros_like(disco)
                cv2.circle(comida, (int(xs.mean()), int(ys.mean())), int(0.2 * diametro(disco)), 1, -1)
            cv2.imwrite(destino, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(destino[:-4] + "_m.png", comida * 255)
            g = grupo_de[f]
            linhas.append({
                "caminho": destino, "classe": classe, "grupo": f"{classe}/{g}",
                "conjunto": conjunto_de[g], "diametro_prato": round(float(diametro(disco)), 1),
                "area_comida": int(comida.sum()), "segm_ok": int(segm_ok),
            })
        resumo = {c: sum(1 for l in linhas if l["classe"] == classe and l["conjunto"] == c) for c in ("treino", "val", "teste")}
        print(f"{classe:20s} grupos={len(set(grupo_de.values())):2d} {resumo}", flush=True)

    with open(args.saida, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    tot = {c: sum(1 for l in linhas if l["conjunto"] == c) for c in ("treino", "val", "teste")}
    print("total", tot)
    vazias = [l["caminho"] for l in linhas if not l["segm_ok"]]
    print(f"segmentação falhou (usou círculo central): {len(vazias)}", vazias[:10])


if __name__ == "__main__":
    main()
