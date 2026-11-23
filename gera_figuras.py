"""Gera figuras pro relatório e pros slides.

  figuras/segmentacao_etapas.png   etapas da segmentação clássica (arroz e prato misturado)
  figuras/classes_grade.png        uma foto por classe, com o contorno da segmentação
  figuras/augmentation.png         recorte padrão e recortes com data augmentation
  figuras/curvas_aug.png           treino x validação: sem aug, com aug, com aug + colagem
  figuras/resultados_pratos.png    Jaccard por prato de cada modelo (IC 95%)
  figuras/recall_classes.png       recall por alimento nos pratos, antes e depois da colagem
  figuras/exemplos_acertos.png     dois pratos que o modelo final acerta
  figuras/exemplos_erros.png       dois pratos com alimentos fora das classes

Uso: python gera_figuras.py
"""
import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recortes import recortar, recorte_aumentado, FRACAO_PADRAO
from segmentacao import carregar, segmentar, diametro

SAIDA = "figuras"
PRATO = "dados/Imagens_Varios_Alimentos/prato_2_1.jpg"


def linhas_divisao():
    with open("divisao.csv") as f:
        return list(csv.DictReader(f))


def um_por_classe(linhas):
    escolha = {}
    for l in linhas:
        if l["conjunto"] == "teste" and l["segm_ok"] == "1" and l["classe"] not in escolha:
            escolha[l["classe"]] = l["caminho"]
    return dict(sorted(escolha.items()))


def contorno(ax, mascara, cor):
    cont, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cont:
        c = np.vstack([c[:, 0], c[:1, 0]])
        ax.plot(c[:, 0], c[:, 1], "-", color=cor, lw=1.5)


def fig_segmentacao(linhas):
    arroz = next(l["caminho"] for l in linhas if l["classe"] == "Arroz" and l["conjunto"] == "teste")
    fotos = [(arroz, "Arroz (branco no branco)"), (PRATO, "prato com vários alimentos")]
    titulos = ["1. entrada", "2. disco do prato\n(Otsu + convexa)", "3. distância de cor\nao prato (a, b)",
               "4. textura\n(desvio local de L)", "5. cor OU textura", "6. após morfologia\ne limpeza"]
    fig, ax = plt.subplots(2, 6, figsize=(18, 6.4))
    for i, (caminho, nome) in enumerate(fotos):
        rgb = carregar(caminho)
        disco, final, e = segmentar(rgb, devolver_etapas=True)
        ax[i, 0].imshow(rgb)
        ax[i, 0].set_ylabel(nome, fontsize=11)
        d = rgb.copy(); d[disco == 0] = (d[disco == 0] * 0.25).astype(np.uint8)
        ax[i, 1].imshow(d)
        contorno(ax[i, 1], disco, "cyan")
        ax[i, 2].imshow(np.where(disco > 0, e["dist_cor"], 0), cmap="magma", vmin=0, vmax=30)
        ax[i, 3].imshow(np.where(disco > 0, e["textura"], 0), cmap="magma", vmin=0, vmax=10)
        ax[i, 4].imshow(e["bruta"], cmap="gray")
        f = rgb.copy(); f[final == 0] = (f[final == 0] * 0.25).astype(np.uint8)
        ax[i, 5].imshow(f)
        contorno(ax[i, 5], final, "yellow")
    for j, t in enumerate(titulos):
        ax[0, j].set_title(t, fontsize=11)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "segmentacao_etapas.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_classes(linhas):
    escolha = um_por_classe(linhas)
    fig, ax = plt.subplots(4, 4, figsize=(12, 12.6))
    for a, (classe, caminho) in zip(ax.ravel(), escolha.items()):
        rgb = carregar(caminho)
        m = cv2.imread(caminho.replace(".jpg", "_m.png"), cv2.IMREAD_GRAYSCALE)
        a.imshow(rgb)
        contorno(a, m > 0, "yellow")
        a.set_title(classe, fontsize=13)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "classes_grade.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_augmentation(linhas, classes=("Beterraba", "Arroz", "FeijaoCarioca"), n_aum=7):
    escolha = um_por_classe(linhas)
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(len(classes), n_aum + 1, figsize=(2 * (n_aum + 1), 2.25 * len(classes)))
    for i, classe in enumerate(classes):
        caminho = escolha[classe]
        rgb = carregar(caminho)
        m = (cv2.imread(caminho.replace(".jpg", "_m.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        disco, _ = segmentar(rgb)
        d = diametro(disco)
        ys, xs = np.nonzero(m)
        ax[i, 0].imshow(recortar(rgb, xs.mean(), ys.mean(), FRACAO_PADRAO * d))
        ax[i, 0].set_ylabel(classe, fontsize=11)
        for j in range(n_aum):
            ax[i, j + 1].imshow(recorte_aumentado(rgb, m, d, rng))
    ax[0, 0].set_title("sem aug\n(centro, 25% do prato)", fontsize=10)
    for j in range(n_aum):
        ax[0, j + 1].set_title(f"aumentado {j + 1}", fontsize=10)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("posição e tamanho (18 a 32% do prato) sorteados, rotação livre, espelhamento, "
                 "brilho, contraste, saturação, matiz e desfoque", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "augmentation.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


MODELOS = ["sem_aug", "com_aug", "mix_aug", "v2_sem_aug", "v2_com_aug", "v2_mix_aug"]
ROTULO = {"sem_aug": "v1 sem aug", "com_aug": "v1 com aug", "mix_aug": "v1 aug + colagem",
          "v2_sem_aug": "v2 sem aug", "v2_com_aug": "v2 com aug", "v2_mix_aug": "v2 aug + colagem"}


def fig_curvas():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for n, cor in (("sem_aug", "tab:red"), ("com_aug", "tab:blue"), ("mix_aug", "tab:green")):
        h = list(csv.DictReader(open(f"resultados/{n}/historico.csv")))
        e = [int(r["epoca"]) for r in h]
        ax[0].plot(e, [float(r["perda_treino"]) for r in h], color=cor, ls="--", label=f"{ROTULO[n]} (treino)")
        ax[0].plot(e, [float(r["perda_val"]) for r in h], color=cor, label=f"{ROTULO[n]} (validação)")
        ax[1].plot(e, [float(r["f1_val"]) for r in h], color=cor, marker="o", ms=3, label=ROTULO[n])
    ax[0].set_xlabel("época"); ax[0].set_ylabel("perda (entropia cruzada)"); ax[0].legend(fontsize=7)
    ax[0].set_title("sem aug a perda de treino vai a zero (decorou)")
    ax[1].set_xlabel("época"); ax[1].set_ylabel("F1 macro na validação"); ax[1].set_ylim(0.9, 1.005); ax[1].legend(fontsize=8)
    ax[1].set_title("validação: com aug fica mais alta e estável")
    fig.tight_layout(); fig.savefig(os.path.join(SAIDA, "curvas_aug.png"), dpi=150); plt.close(fig)


def jaccard_por_prato(n):
    import collections
    por = collections.defaultdict(list)
    for l in csv.DictReader(open(f"resultados/{n}/pratos/por_foto.csv")):
        por[l["rajada"]].append(float(l["jaccard"]))
    return np.array([np.mean(v) for v in por.values()])


def fig_pratos_modelos():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, n in enumerate(MODELOS):
        jp = jaccard_por_prato(n)
        bs = [rng.choice(jp, len(jp)).mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        cor = "tab:green" if "mix" in n else ("tab:blue" if "com" in n else "tab:red")
        ax.errorbar(i, jp.mean(), yerr=[[jp.mean() - lo], [hi - jp.mean()]], fmt="o", color=cor, capsize=6, ms=8)
        ax.text(i + 0.08, jp.mean(), f"{jp.mean():.2f}", va="center", fontsize=9)
    ax.set_xticks(range(len(MODELOS)), [ROTULO[n] for n in MODELOS], rotation=20)
    ax.set_ylabel("Jaccard médio por prato"); ax.set_ylim(0.6, 0.95)
    ax.set_title("34 pratos: os intervalos de 95% se sobrepõem (diferença dentro do ruído)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(SAIDA, "resultados_pratos.png"), dpi=150); plt.close(fig)


def recall_por_classe(n):
    out = {}
    for linha in open(f"resultados/{n}/pratos/relatorio.txt"):
        p = linha.split()
        if len(p) == 5 and p[0][0].isupper():
            out[p[0]] = float(p[2])
    return out


def fig_recall_classes():
    a, b = recall_por_classe("v2_com_aug"), recall_por_classe("v2_mix_aug")
    nomes = sorted(a, key=lambda c: a[c])
    y = np.arange(len(nomes))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(y - 0.2, [a[c] for c in nomes], 0.4, color="tab:blue", label="v2 com aug")
    ax.barh(y + 0.2, [b[c] for c in nomes], 0.4, color="tab:green", label="v2 aug + colagem")
    ax.set_yticks(y, nomes); ax.set_xlabel("recall nas fotos de pratos (fração das vezes que o alimento foi achado)")
    ax.set_xlim(0, 1.05); ax.legend(loc="lower right")
    ax.set_title("colagem: arroz, purê e maionese sobem; beterraba cai")
    fig.tight_layout(); fig.savefig(os.path.join(SAIDA, "recall_classes.png"), dpi=150); plt.close(fig)


def fig_exemplos():
    from analisa_prato import analisar, carregar_modelo, figura
    from treino import dispositivo
    dev = dispositivo()
    modelo, classes = carregar_modelo("resultados/v2_mix_aug/modelo.pt", dev)
    grupos = {"exemplos_acertos": [("IMG_20140926_140501433", "acertou os 6 alimentos"),
                                   ("IMG_20141114_135647024", "acertou os 4 alimentos")],
              "exemplos_erros": [("IMG_20150309_142923008", "feijão PRETO (fora das classes) virou Almondega e FeijaoCarioca"),
                                 ("IMG_20150319_133153321", "nuggets e nhoque (fora das classes) viraram strogonoff")]}
    for nome, lista in grupos.items():
        ims = []
        for f, txt in lista:
            rgb = carregar(f"amostra_pratos/{f}.jpg")
            r = analisar(rgb, modelo, classes, dev)
            fig = figura(rgb, r, classes, None, titulo=f"{f}: {txt}")
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            ims.append((plt.imread(buf)[:, :, :3] * 255).astype(np.uint8))
        w = max(i.shape[1] for i in ims)
        ims = [np.pad(i, ((0, 0), (0, w - i.shape[1]), (0, 0)), constant_values=255) for i in ims]
        cv2.imwrite(os.path.join(SAIDA, nome + ".png"), cv2.cvtColor(np.vstack(ims), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    os.makedirs(SAIDA, exist_ok=True)
    linhas = linhas_divisao()
    fig_segmentacao(linhas)
    fig_classes(linhas)
    fig_augmentation(linhas)
    fig_curvas()
    fig_pratos_modelos()
    fig_recall_classes()
    fig_exemplos()
    print("ok: figuras geradas em figuras/")
