"""Avaliação quantitativa nos pratos (tarefa multi-rótulo).

Compara o conjunto de alimentos previsto em cada foto com o rótulo da rajada
(rotulos/pratos.csv). "OUTRO" no rótulo é ignorado (a rede não conhece).

Métricas:
  - precisão, recall e F1 por classe e média macro e micro
  - acerto exato (o conjunto previsto é idêntico ao verdadeiro)
  - Jaccard médio por foto (|previsto ∩ verdadeiro| / |previsto ∪ verdadeiro|)

Uso: python avalia_pratos.py --modelo resultados/com_aug/modelo.pt --nome com_aug
"""
import argparse
import csv
import unicodedata
import json
import os

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

from analisa_prato import analisar, carregar_modelo, figura
from prepara_dados import grupos_da_classe
from segmentacao import carregar
from treino import dispositivo

PASTA = "dados/Imagens_Varios_Alimentos"


def sem_acento(t):
    """Almôndega -> Almondega, Feijão -> Feijao."""
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def ler_rotulos(caminho, classes):
    """Lê rotulos/pratos.csv (separado por "," ou ";").
    Devolve ({rajada: conjunto de alimentos}, {rajada: tem alimento fora das classes}).
    Alimento fora das classes vira "OUTRO"."""
    with open(caminho, encoding="utf-8-sig") as fh:
        texto = fh.read()
    # o Numbers em português exporta com ";" entre colunas
    sep = ";" if texto.splitlines()[0].count(";") > texto.splitlines()[0].count(",") else ","
    rot = {l["rajada"]: {sem_acento(a.strip()) for a in l["alimentos"].split(";") if a.strip()}
           for l in csv.DictReader(texto.splitlines(), delimiter=sep)}
    faltando = [r for r, a in rot.items() if not a]
    if faltando:
        print(f"aviso: {len(faltando)} pratos sem rótulo, ficam de fora: {faltando}")
    # alimento fora das 16 classes (Panqueca, FeijaoPreto...) vira OUTRO
    conhecidas = set(classes) - {"Prato"}
    desconhecidos = sorted({a for s in rot.values() for a in s} - conhecidas - {"OUTRO"})
    if desconhecidos:
        print(f"fora das classes, contam como OUTRO: {', '.join(desconhecidos)}")
    tem_outro = {r: bool(a - conhecidas) for r, a in rot.items()}
    rot = {r: (a & conhecidas) | ({"OUTRO"} if tem_outro[r] else set()) for r, a in rot.items()}
    return rot, tem_outro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="resultados/com_aug/modelo.pt")
    ap.add_argument("--nome", default="com_aug")
    ap.add_argument("--rotulos", default="rotulos/pratos.csv")
    ap.add_argument("--area-min", type=float, default=0.04)
    ap.add_argument("--conf-min", type=float, default=0.5)
    ap.add_argument("--figuras", action="store_true", help="salva a figura de cada foto")
    args = ap.parse_args()

    dev = dispositivo()
    modelo, classes = carregar_modelo(args.modelo, dev)
    rot, tem_outro = ler_rotulos(args.rotulos, classes)

    fotos = sorted(f for f in os.listdir(PASTA) if f.lower().endswith(".jpg"))
    grupo_de = grupos_da_classe(fotos)
    saida = os.path.join("resultados", args.nome, "pratos")
    os.makedirs(saida, exist_ok=True)

    Y, P, linhas = [], [], []
    for f in fotos:
        verdade = rot.get(grupo_de[f], set()) - {"OUTRO"}
        if not rot.get(grupo_de[f]):
            continue
        rgb = carregar(os.path.join(PASTA, f))
        r = analisar(rgb, modelo, classes, dev, conf_min=args.conf_min, area_min=args.area_min)
        previsto = {c for c, _, _ in r["encontrados"]}
        Y.append([c in verdade for c in classes])
        P.append([c in previsto for c in classes])
        uniao = verdade | previsto
        jac = len(verdade & previsto) / len(uniao) if uniao else 1.0
        linhas.append({"foto": f, "rajada": grupo_de[f], "verdade": ";".join(sorted(verdade)),
                       "previsto": ";".join(sorted(previsto)), "jaccard": round(jac, 3), "tem_outro": int(tem_outro[grupo_de[f]]),
                       "faltou": ";".join(sorted(verdade - previsto)), "sobrou": ";".join(sorted(previsto - verdade))})
        if args.figuras:
            figura(rgb, r, classes, os.path.join(saida, os.path.splitext(f)[0] + ".png"), titulo=f)
    Y, P = np.array(Y), np.array(P)

    with open(os.path.join(saida, "por_foto.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    presentes = Y.sum(0) > 0
    pr, rc, f1, _ = precision_recall_fscore_support(Y[:, presentes], P[:, presentes], average="macro", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(Y, P, average="micro", zero_division=0)
    metricas = {
        "fotos": len(Y), "pratos": len({l["rajada"] for l in linhas}),
        "precisao_macro": pr, "recall_macro": rc, "f1_macro": f1,
        "precisao_micro": pm, "recall_micro": rm, "f1_micro": fm,
        "acerto_exato": float(np.mean((Y == P).all(1))),
        "jaccard_medio": float(np.mean([l["jaccard"] for l in linhas])),
        "jaccard_sem_outro": float(np.mean([l["jaccard"] for l in linhas if not l["tem_outro"]])),
        "jaccard_com_outro": float(np.mean([l["jaccard"] for l in linhas if l["tem_outro"]] or [np.nan])),
        "area_min": args.area_min, "conf_min": args.conf_min,
    }
    with open(os.path.join(saida, "metricas.json"), "w") as fh:
        json.dump(metricas, fh, indent=2)
    nomes = [c for c, p in zip(classes, presentes) if p]
    rel = classification_report(Y[:, presentes], P[:, presentes], target_names=nomes, digits=3, zero_division=0)
    with open(os.path.join(saida, "relatorio.txt"), "w") as fh:
        fh.write(rel)
    print(json.dumps(metricas, indent=2))
    print(rel)


if __name__ == "__main__":
    main()
