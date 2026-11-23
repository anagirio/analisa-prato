"""Ferramenta pra contornar a comida à mão (máscara verdadeira pra avaliar a segmentação).

Uso: python contorna.py
     (na primeira vez escolhe 20 fotos e salva a lista em rotulos/mascaras/lista.csv)

Na janela:
  clique esquerdo   adiciona um ponto do contorno
  Enter ou espaço   fecha o polígono (pode fazer vários por foto, um por pedaço de comida)
  z                 desfaz o último ponto (ou o último polígono)
  r                 recomeça a foto
  n                 salva e vai pra próxima
  b                 volta pra foto anterior (refaz o contorno dela)
  q                 salva e sai (dá pra continuar depois, as prontas são puladas)

Contorne só a comida, por dentro do prato. Alimentos encostados podem ir num polígono só.
"""
import csv
import os

import cv2
import numpy as np

from segmentacao import carregar

PASTA = "rotulos/mascaras"
LISTA = os.path.join(PASTA, "lista.csv")
CLASSES_UM = ["Arroz", "PureBatata", "Maionese", "Alface", "Macarrao", "Beterraba", "FeijaoCarioca",
              "BatataFrita", "StrogonoffFrango", "Tomate"]


def escolhe_fotos():
    """10 fotos de um alimento (do teste, uma por classe) + 10 pratos (rajadas espaçadas)."""
    rng = np.random.default_rng(0)
    por_classe = {}
    with open("divisao.csv") as fh:
        for l in csv.DictReader(fh):
            if l["conjunto"] == "teste" and l["classe"] in CLASSES_UM:
                por_classe.setdefault(l["classe"], []).append(os.path.basename(l["caminho"]))
    lista = []
    for c in CLASSES_UM:
        f = por_classe[c][rng.integers(len(por_classe[c]))]
        lista.append({"id": f"um_{c}", "tipo": "um", "caminho": f"dados/Imagens_um_Alimento/{c}/{f}"})
    with open("rotulos/pratos.csv") as fh:
        pratos = list(csv.DictReader(fh, delimiter=";"))
    for l in pratos[::3][:10]:
        lista.append({"id": f"prato_{l['rajada']}", "tipo": "prato",
                      "caminho": f"dados/Imagens_Varios_Alimentos/{l['exemplo']}"})
    os.makedirs(PASTA, exist_ok=True)
    with open(LISTA, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "tipo", "caminho"])
        w.writeheader(); w.writerows(lista)
    return lista


def le_lista():
    if not os.path.exists(LISTA):
        return escolhe_fotos()
    with open(LISTA) as fh:
        return list(csv.DictReader(fh))


def contorna(rgb, titulo):
    """Devolve a máscara (0/1), None se pediu pra sair sem salvar nada, ou 'sair' junto da máscara."""
    poligonos, atual = [], []
    janela = "contorno"

    def clique(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            atual.append((x, y))

    cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(janela, clique)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    while True:
        tela = bgr.copy()
        masc = np.zeros(rgb.shape[:2], np.uint8)
        for p in poligonos:
            cv2.fillPoly(masc, [np.array(p)], 1)
        verde = tela.copy(); verde[masc > 0] = (0, 200, 0)
        tela = cv2.addWeighted(tela, 0.65, verde, 0.35, 0)
        for p in poligonos:
            cv2.polylines(tela, [np.array(p)], True, (0, 255, 0), 1)
        if atual:
            cv2.polylines(tela, [np.array(atual)], False, (0, 255, 255), 1)
            for x, y in atual:
                cv2.circle(tela, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(tela, titulo, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(tela, titulo, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(janela, tela)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 32) and len(atual) >= 3:
            poligonos.append(list(atual)); atual.clear()
        elif k == ord("z"):
            if atual:
                atual.pop()
            elif poligonos:
                poligonos.pop()
        elif k == ord("r"):
            poligonos.clear(); atual.clear()
        elif k in (ord("n"), ord("q"), ord("b")):
            if len(atual) >= 3:
                poligonos.append(list(atual))
            masc = np.zeros(rgb.shape[:2], np.uint8)
            for p in poligonos:
                cv2.fillPoly(masc, [np.array(p)], 1)
            return masc, {ord("n"): "proxima", ord("q"): "sair", ord("b"): "voltar"}[k]


def main():
    lista = le_lista()
    pronta = lambda l: os.path.exists(os.path.join(PASTA, l["id"] + ".png"))
    print(f"{sum(map(pronta, lista))} de {len(lista)} prontas")
    faltam = [i for i, l in enumerate(lista) if not pronta(l)]
    i = faltam[0] if faltam else len(lista) - 1
    while 0 <= i < len(lista):
        l = lista[i]
        aviso = "  (ja feita: refaca ou n pra manter)" if pronta(l) else ""
        masc, acao = contorna(carregar(l["caminho"]), f"{l['id']}  ({i + 1}/{len(lista)})  n=proxima b=volta q=sai{aviso}")
        if masc.sum() > 0:
            cv2.imwrite(os.path.join(PASTA, l["id"] + ".png"), masc * 255)
            print("salva", l["id"])
        if acao == "sair":
            break
        if acao == "voltar":
            i = max(i - 1, 0)
            continue
        i += 1
        while i < len(lista) and pronta(lista[i]):  # pula as que já estão prontas
            i += 1
    cv2.destroyAllWindows()
    print(f"{sum(map(pronta, lista))} de {len(lista)} prontas")


if __name__ == "__main__":
    main()
