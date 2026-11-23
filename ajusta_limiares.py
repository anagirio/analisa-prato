"""Ajusta os limiares da análise do prato (confiança mínima e área mínima)
sem "colar" na prova: validação cruzada em 2 partes, separando por PRATO.

Pra cada divisão: escolhe o melhor par (conf_min, area_min) numa metade dos
pratos (maior Jaccard médio por prato) e mede na outra metade, e vice-versa.
Assim todo prato é avaliado com limiares que nunca viram aquele prato.
Repete com 10 divisões diferentes e junta tudo.

A rede roda uma vez por foto (cache em resultados/<nome>/pratos/areas.npz);
pra cada confiança da grade guardamos a fração da comida de cada classe.

Uso: python ajusta_limiares.py --modelo resultados/v2_com_aug/modelo.pt --nome v2_com_aug
"""
import argparse
import collections
import json
import os

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from analisa_prato import analisar, carregar_modelo
from avalia_pratos import ler_rotulos, PASTA
from prepara_dados import grupos_da_classe
from segmentacao import carregar
from treino import dispositivo

CONFS = [0.3, 0.4, 0.5, 0.6, 0.7]
AREAS = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10]


def tabela_areas(caminho_modelo, fotos, cache):
    """areas[foto, i_conf, classe] = fração da comida rotulada como a classe com confiança >= CONFS[i]."""
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["fotos"]) == fotos:
            return z["areas"], list(z["classes"])
    dev = dispositivo()
    modelo, classes = carregar_modelo(caminho_modelo, dev)
    areas = np.zeros((len(fotos), len(CONFS), len(classes)), np.float32)
    for n, f in enumerate(fotos):
        r = analisar(carregar(os.path.join(PASTA, f)), modelo, classes, dev, conf_min=0.0, area_min=1.0)
        comida = r["comida"] > 0
        total = max(int(comida.sum()), 1)
        for i, c in enumerate(CONFS):
            ok = comida & (r["conf"] >= c)
            areas[n, i] = np.bincount(r["mapa"][ok].clip(0), minlength=len(classes))[:len(classes)] / total
        if n % 50 == 0:
            print(f"  {n}/{len(fotos)}", flush=True)
    np.savez(cache, areas=areas, fotos=np.array(fotos), classes=np.array(classes))
    return areas, classes


def jaccard(v, p):
    u = v | p
    return len(v & p) / len(u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--nome", required=True)
    ap.add_argument("--rotulos", default="rotulos/pratos.csv")
    ap.add_argument("--repeticoes", type=int, default=10)
    args = ap.parse_args()

    fotos = sorted(f for f in os.listdir(PASTA) if f.lower().endswith(".jpg"))
    grupo_de = grupos_da_classe(fotos)
    saida = os.path.join("resultados", args.nome, "pratos")
    os.makedirs(saida, exist_ok=True)
    areas, classes = tabela_areas(args.modelo, fotos, os.path.join(saida, "areas.npz"))
    rot, _ = ler_rotulos(args.rotulos, classes)
    comidas = [c for c in classes if c != "Prato"]
    idx = [classes.index(c) for c in comidas]

    manter = [i for i, f in enumerate(fotos) if rot.get(grupo_de[f])]
    pratos = sorted({grupo_de[fotos[i]] for i in manter})
    verdade = {i: rot[grupo_de[fotos[i]]] - {"OUTRO"} for i in manter}

    def previsto(i, ic, a):
        return {comidas[k] for k, j in enumerate(idx) if areas[i, ic, j] >= a}

    def jaccard_por_prato(fotos_i, ic, a):
        por = collections.defaultdict(list)
        for i in fotos_i:
            por[grupo_de[fotos[i]]].append(jaccard(verdade[i], previsto(i, ic, a)))
        return np.mean([np.mean(v) for v in por.values()])

    rng = np.random.default_rng(0)
    escolhas = collections.Counter()
    resultados = []   # por repetição: lista de (foto, previsto)
    for _ in range(args.repeticoes):
        perm = rng.permutation(pratos)
        metades = [set(perm[:len(perm) // 2]), set(perm[len(perm) // 2:])]
        prev = {}
        for k in (0, 1):
            ajuste = [i for i in manter if grupo_de[fotos[i]] in metades[k]]
            teste = [i for i in manter if grupo_de[fotos[i]] in metades[1 - k]]
            notas = {(ic, a): jaccard_por_prato(ajuste, ic, a) for ic in range(len(CONFS)) for a in AREAS}
            ic, a = max(notas, key=notas.get)
            escolhas[(CONFS[ic], a)] += 1
            for i in teste:
                prev[i] = previsto(i, ic, a)
        resultados.append(prev)

    def metricas(prev):
        Y = np.array([[c in verdade[i] for c in comidas] for i in manter])
        P = np.array([[c in prev[i] for c in comidas] for i in manter])
        pres = Y.sum(0) > 0
        pr, rc, f1, _ = precision_recall_fscore_support(Y[:, pres], P[:, pres], average="macro", zero_division=0)
        _, _, fm, _ = precision_recall_fscore_support(Y, P, average="micro", zero_division=0)
        por = collections.defaultdict(list)
        for i in manter:
            por[grupo_de[fotos[i]]].append(jaccard(verdade[i], prev[i]))
        return {"precisao_macro": pr, "recall_macro": rc, "f1_macro": f1, "f1_micro": fm,
                "acerto_exato": float((Y == P).all(1).mean()),
                "jaccard_prato": float(np.mean([np.mean(v) for v in por.values()]))}, \
            np.array([np.mean(v) for v in por.values()])

    ms = [metricas(p) for p in resultados]
    media = {k: float(np.mean([m[0][k] for m in ms])) for k in ms[0][0]}
    # IC95 do Jaccard por prato: bootstrap sobre pratos, usando a média das repetições
    jp = np.mean([m[1] for m in ms], axis=0)
    bs = [rng.choice(jp, len(jp)).mean() for _ in range(2000)]
    media["jaccard_ic95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    # referência: limiares padrão (0,5 e 4%) nas mesmas fotos
    padrao = {i: previsto(i, CONFS.index(0.5), 0.04) for i in manter}
    media["padrao_0.5_0.04"] = metricas(padrao)[0]
    media["limiares_escolhidos"] = {f"conf={c} area={a}": n for (c, a), n in escolhas.most_common()}
    with open(os.path.join(saida, "limiares_cv.json"), "w") as fh:
        json.dump(media, fh, indent=2)
    print(json.dumps(media, indent=2))


if __name__ == "__main__":
    main()
