"""Gera a folha pra rotular os pratos à mão (uma linha por rajada = um prato).

Saídas:
  rotulos/folha_pratos.jpg   miniaturas numeradas, uma por rajada
  rotulos/pratos.csv         rajada, exemplo, n_fotos, alimentos (vazio pra preencher)

Como preencher: na coluna `alimentos`, escreva as classes separadas por ";",
com os mesmos nomes das pastas (ex.: Arroz;FeijaoCarioca;Alface).
Alimento que não está nas 16 classes: escreva OUTRO.
"""
import csv
import os

import cv2
import numpy as np

from prepara_dados import grupos_da_classe
from segmentacao import carregar

PASTA = "dados/Imagens_Varios_Alimentos"


def main():
    os.makedirs("rotulos", exist_ok=True)
    fotos = sorted(f for f in os.listdir(PASTA) if f.lower().endswith(".jpg"))
    grupo_de = grupos_da_classe(fotos)
    rajadas = sorted(set(grupo_de.values()))
    destino_csv = "rotulos/pratos.csv"
    if os.path.exists(destino_csv):
        print(f"{destino_csv} já existe, não vou sobrescrever")
    else:
        with open(destino_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "rajada", "exemplo", "n_fotos", "alimentos"])
            for i, r in enumerate(rajadas):
                sel = [f for f in fotos if grupo_de[f] == r]
                w.writerow([i, r, sel[0], len(sel), ""])
    lado, tiles = 260, []
    for i, r in enumerate(rajadas):
        f = [f for f in fotos if grupo_de[f] == r][0]
        im = carregar(os.path.join(PASTA, f), lado)
        t = np.full((lado, lado, 3), 255, np.uint8)
        t[:im.shape[0], :im.shape[1]] = im
        cv2.rectangle(t, (0, 0), (52, 30), (0, 0, 0), -1)
        cv2.putText(t, str(i), (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        tiles.append(t)
    while len(tiles) % 6:
        tiles.append(np.full((lado, lado, 3), 255, np.uint8))
    folha = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    cv2.imwrite("rotulos/folha_pratos.jpg", cv2.cvtColor(folha, cv2.COLOR_RGB2BGR))
    print(f"{len(rajadas)} pratos -> rotulos/folha_pratos.jpg e {destino_csv}")


if __name__ == "__main__":
    main()
