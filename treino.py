"""Treina a CNN (ResNet18 pré-treinada na ImageNet) em recortes de comida.

Uso:
  python treino.py --nome com_aug                 # com data augmentation
  python treino.py --nome sem_aug --sem-aug       # baseline sem augmentation
  python treino.py --nome v2_com_aug --classe-prato   # v2: 17a classe "Prato" (fundo)
  python treino.py --nome mix_aug --mistura 0.4       # + augmentation de prato misturado

Na v2 a rede também aprende o que é prato vazio, com recortes da parte branca do
prato longe da comida. Fotos de Arroz ficam de fora dessa amostragem: a
segmentação clássica falha no arroz, e o recorte "de prato" poderia ter arroz.

Saídas em resultados/<nome>/: modelo.pt, historico.csv, curvas.png,
matriz_confusao.png, relatorio_teste.txt, metricas.json
"""
import argparse
import csv
import json
import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from recortes import recortar, recorte_aumentado, recorte_misturado, centros_grade, para_tensor, FRACAO_PADRAO
from segmentacao import mascara_prato

PRATO = "Prato"


def dispositivo():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def regiao_fundo(rgb, comida, diam):
    """Pixels de prato vazio que servem de centro pra um recorte da classe Prato:
    dentro do prato e a pelo menos 15% do diâmetro de distância da comida."""
    disco = mascara_prato(rgb)
    r = max(3, int(0.15 * diam))
    perto = cv2.dilate(comida.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    fundo = (disco > 0) & (perto == 0)
    return fundo if fundo.sum() >= 200 else None


def carregar_conjunto(linhas, com_prato=False):
    """Lista de (rgb, mascara_comida, diametro_prato, regiao_fundo ou None)."""
    dados = []
    for l in linhas:
        rgb = cv2.cvtColor(cv2.imread(l["caminho"]), cv2.COLOR_BGR2RGB)
        m = cv2.imread(l["caminho"][:-4] + "_m.png", cv2.IMREAD_GRAYSCALE) > 0
        d = float(l["diametro_prato"])
        fundo = None
        if com_prato and l["classe"] != "Arroz" and l.get("segm_ok", "1") == "1":
            fundo = regiao_fundo(rgb, m, d)
        dados.append((rgb, m, d, fundo))
    return dados


def criar_modelo(n_classes):
    modelo = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    modelo.fc = nn.Linear(modelo.fc.in_features, n_classes)
    return modelo


def recortes_fixos(dados, rotulos, k, semente, k_prato=0, rotulo_prato=None):
    """Baseline sem augmentation: k recortes fixos por foto, tamanho padrão, sem giro nem cor.
    Devolve lista de (recorte, rótulo)."""
    rng = np.random.default_rng(semente)
    fixos = []
    for (rgb, m, d, fundo), y in zip(dados, rotulos):
        for regiao, rot, n in ((m, y, k), (fundo, rotulo_prato, k_prato)):
            if regiao is None or n == 0:
                continue
            ys, xs = np.nonzero(regiao)
            for i in rng.integers(len(xs), size=n):
                fixos.append((recortar(rgb, xs[i], ys[i], d * FRACAO_PADRAO), rot))
    return fixos


@torch.no_grad()
def taxa_prato(modelo, dados, dev, idx_prato, k=4, semente=7):
    """Teste binário comida x prato: k recortes fixos de cada tipo por foto.
    Devolve (fração de recortes de prato chamados de Prato,
             fração de recortes de comida chamados de Prato)."""
    modelo.eval()
    rng = np.random.default_rng(semente)
    acertos_prato, n_prato, falso_prato, n_comida = 0, 0, 0, 0
    for rgb, m, d, fundo in dados:
        for regiao, eh_prato in ((fundo, True), (m, False)):
            if regiao is None:
                continue
            ys, xs = np.nonzero(regiao)
            ps = [recortar(rgb, xs[i], ys[i], d * FRACAO_PADRAO) for i in rng.integers(len(xs), size=k)]
            pred = modelo(torch.from_numpy(para_tensor(ps)).to(dev)).argmax(1).cpu().numpy()
            if eh_prato:
                acertos_prato += int((pred == idx_prato).sum()); n_prato += k
            else:
                falso_prato += int((pred == idx_prato).sum()); n_comida += k
    return acertos_prato / max(n_prato, 1), falso_prato / max(n_comida, 1)


@torch.no_grad()
def prever_fotos(modelo, dados, dev, lote=256):
    """Probabilidade média dos recortes da grade, por foto."""
    modelo.eval()
    probs = []
    for rgb, m, d, _ in dados:
        centros, lado = centros_grade(m, d)
        patches = [recortar(rgb, cx, cy, lado) for cx, cy in centros]
        p = []
        for i in range(0, len(patches), lote):
            x = torch.from_numpy(para_tensor(patches[i:i + lote])).to(dev)
            p.append(torch.softmax(modelo(x), 1).cpu())
        probs.append(torch.cat(p).mean(0).numpy())
    return np.stack(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True)
    ap.add_argument("--sem-aug", action="store_true")
    ap.add_argument("--epocas", type=int, default=15)
    ap.add_argument("--recortes-por-foto", type=int, default=8)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--divisao", default="divisao.csv")
    ap.add_argument("--classe-prato", action="store_true", help="v2: adiciona a classe Prato (fundo)")
    ap.add_argument("--mistura", type=float, default=0.0,
                    help="fração dos recortes de comida que viram 'prato misturado' (dois alimentos, rótulo suave)")
    args = ap.parse_args()

    torch.manual_seed(args.semente)
    rng = np.random.default_rng(args.semente)
    saida = os.path.join("resultados", args.nome)
    os.makedirs(saida, exist_ok=True)
    dev = dispositivo()

    with open(args.divisao) as fh:
        linhas = list(csv.DictReader(fh))
    classes = sorted({l["classe"] for l in linhas})
    n_comida = len(classes)
    if args.classe_prato:
        classes = classes + [PRATO]
    idx_prato = classes.index(PRATO) if args.classe_prato else None
    por_conj = {c: [l for l in linhas if l["conjunto"] == c] for c in ("treino", "val", "teste")}
    y = {c: np.array([classes.index(l["classe"]) for l in por_conj[c]]) for c in por_conj}
    print(f"dispositivo={dev} classes={len(classes)} " + " ".join(f"{c}={len(v)}" for c, v in por_conj.items()), flush=True)

    t0 = time.time()
    dados = {c: carregar_conjunto(por_conj[c], args.classe_prato) for c in por_conj}
    print(f"imagens carregadas em {time.time() - t0:.0f}s", flush=True)
    if args.classe_prato:
        n_fundo = sum(1 for t in dados["treino"] if t[3] is not None)
        print(f"fotos de treino com região de prato vazio: {n_fundo}", flush=True)

    modelo = criar_modelo(len(classes)).to(dev)
    otim = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    agenda = torch.optim.lr_scheduler.CosineAnnealingLR(otim, T_max=args.epocas)
    perda_fn = nn.CrossEntropyLoss()

    # recortes de prato por foto: 1 pra cada 8 de comida (fica ~do tamanho de uma classe de comida)
    k_prato = max(1, args.recortes_por_foto // 8) if args.classe_prato else 0
    fixos = None if not args.sem_aug else recortes_fixos(
        dados["treino"], y["treino"], args.recortes_por_foto, args.semente, k_prato, idx_prato)

    historico, melhor_nota = [], -np.inf
    for ep in range(1, args.epocas + 1):
        t0 = time.time()
        modelo.train()
        if args.sem_aug:
            amostras = [fixos[i] for i in rng.permutation(len(fixos))]
        else:
            idxs = np.repeat(np.arange(len(dados["treino"])), args.recortes_por_foto)
            amostras = []
            for i in idxs:
                if rng.random() < args.mistura:
                    # parceiro de outra classe, sorteado
                    j = rng.integers(len(dados["treino"]))
                    while y["treino"][j] == y["treino"][i]:
                        j = rng.integers(len(dados["treino"]))
                    p, f = recorte_misturado(dados["treino"][i][:3], dados["treino"][j][:3], rng)
                    alvo = np.zeros(len(classes), np.float32)
                    alvo[y["treino"][i]] += 1 - f
                    alvo[y["treino"][j]] += f
                    amostras.append((p, alvo))
                else:
                    amostras.append((recorte_aumentado(*dados["treino"][i][:3], rng), y["treino"][i]))
            for _ in range(k_prato):
                amostras += [(recorte_aumentado(rgb, fundo, d, rng), idx_prato)
                             for rgb, _, d, fundo in dados["treino"] if fundo is not None]
            amostras = [amostras[i] for i in rng.permutation(len(amostras))]

        soma_perda, acertos = 0.0, 0
        for i in range(0, len(amostras), args.lote):
            bloco = amostras[i:i + args.lote]
            x = torch.from_numpy(para_tensor([b[0] for b in bloco])).to(dev)
            # rótulo suave (vetor de probabilidades): serve pros recortes misturados
            alvo = torch.from_numpy(np.stack([b[1] if isinstance(b[1], np.ndarray)
                                              else np.eye(len(classes), dtype=np.float32)[int(b[1])] for b in bloco])).to(dev)
            saida_rede = modelo(x)
            perda = perda_fn(saida_rede, alvo)
            otim.zero_grad()
            perda.backward()
            otim.step()
            soma_perda += perda.item() * len(bloco)
            acertos += (saida_rede.argmax(1) == alvo.argmax(1)).sum().item()
        agenda.step()

        pv = prever_fotos(modelo, dados["val"], dev)
        pred_v = pv[:, :n_comida].argmax(1)
        val_acc = accuracy_score(y["val"], pred_v)
        val_f1 = f1_score(y["val"], pred_v, average="macro")
        val_perda = float(-np.log(pv[np.arange(len(pv)), y["val"]] + 1e-9).mean())
        reg = {"epoca": ep, "perda_treino": soma_perda / len(amostras), "acc_treino": acertos / len(amostras),
               "perda_val": val_perda, "acc_val": val_acc, "f1_val": val_f1}
        # nota pra escolher o melhor modelo: F1 e, no empate, menor perda
        nota = val_f1 - 0.1 * val_perda
        extra = ""
        if args.classe_prato:
            ok, falso = taxa_prato(modelo, dados["val"], dev, idx_prato)
            reg["prato_ok_val"], reg["comida_como_prato_val"] = ok, falso
            nota += (ok + (1 - falso)) / 2
            extra = f" | prato ok {ok:.3f} comida->prato {falso:.3f}"
        historico.append(reg)
        print(f"ep {ep:2d} perda {reg['perda_treino']:.3f} acc_treino {reg['acc_treino']:.3f} | "
              f"val perda {val_perda:.3f} acc {val_acc:.3f} f1 {val_f1:.3f}{extra} | {time.time() - t0:.0f}s", flush=True)
        if nota > melhor_nota:
            melhor_nota = nota
            torch.save({"estado": modelo.state_dict(), "classes": classes, "epoca": ep}, os.path.join(saida, "modelo.pt"))

    with open(os.path.join(saida, "historico.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(historico[0]))
        w.writeheader()
        w.writerows(historico)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    e = [h["epoca"] for h in historico]
    ax[0].plot(e, [h["perda_treino"] for h in historico], label="treino (recortes)")
    ax[0].plot(e, [h["perda_val"] for h in historico], label="validação (fotos)")
    ax[0].set_xlabel("época"); ax[0].set_ylabel("perda (entropia cruzada)"); ax[0].legend()
    ax[1].plot(e, [h["acc_treino"] for h in historico], label="acc treino (recortes)")
    ax[1].plot(e, [h["acc_val"] for h in historico], label="acc validação (fotos)")
    ax[1].plot(e, [h["f1_val"] for h in historico], label="F1 macro validação", ls="--")
    ax[1].set_xlabel("época"); ax[1].set_ylim(0, 1.02); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(saida, "curvas.png"), dpi=150); plt.close(fig)

    # teste com o melhor modelo da validação
    ck = torch.load(os.path.join(saida, "modelo.pt"), map_location=dev, weights_only=True)
    modelo.load_state_dict(ck["estado"])
    pt = prever_fotos(modelo, dados["teste"], dev)[:, :n_comida]
    pred = pt.argmax(1)
    top3 = float(np.mean([y["teste"][i] in np.argsort(-pt[i])[:3] for i in range(len(pred))]))
    metricas = {"melhor_epoca": ck["epoca"], "acc_teste": accuracy_score(y["teste"], pred),
                "f1_macro_teste": f1_score(y["teste"], pred, average="macro"), "top3_teste": top3}
    if args.classe_prato:
        ok, falso = taxa_prato(modelo, dados["teste"], dev, idx_prato)
        metricas["prato_ok_teste"], metricas["comida_como_prato_teste"] = ok, falso
    with open(os.path.join(saida, "metricas.json"), "w") as fh:
        json.dump(metricas, fh, indent=2)
    classes = classes[:n_comida]
    rel = classification_report(y["teste"], pred, target_names=classes, digits=3)
    with open(os.path.join(saida, "relatorio_teste.txt"), "w") as fh:
        fh.write(rel)
    np.save(os.path.join(saida, "probs_teste.npy"), pt)

    cm = confusion_matrix(y["teste"], pred, labels=range(len(classes)))
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("previsto"); ax.set_ylabel("verdadeiro")
    ax.set_title(f"{args.nome}: acc {metricas['acc_teste']:.3f}, F1 macro {metricas['f1_macro_teste']:.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(saida, "matriz_confusao.png"), dpi=150); plt.close(fig)

    print(json.dumps(metricas, indent=2))
    print(rel)


if __name__ == "__main__":
    main()
