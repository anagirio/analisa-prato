"""Segmentação clássica: acha o prato branco e separa a comida em cima dele.

Etapas (cada uma pode ser salva pra ilustrar o relatório):
1. Reduz a imagem e converte pra Lab (L = brilho, a/b = cor).
2. Limiar de Otsu no L: o prato é a região clara grande.
3. Maior componente conexo + envoltória convexa = disco do prato.
4. Dentro do disco, comida = pixel com cor diferente do prato (distância em a/b)
   OU com textura (desvio padrão local alto, pega arroz e purê que são brancos).
5. Morfologia (abertura e fechamento) e remoção de componentes pequenos.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # fotos .heic do iPhone
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

LADO_MAX = 800


def carregar(caminho, lado_max=LADO_MAX):
    """Abre a foto respeitando a rotação do EXIF e reduz pra lado_max (RGB uint8)."""
    img = ImageOps.exif_transpose(Image.open(caminho)).convert("RGB")
    img.thumbnail((lado_max, lado_max), Image.BILINEAR)
    return np.array(img)


def mascara_prato(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = cv2.GaussianBlur(lab[:, :, 0], (7, 7), 0)
    _, claro = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    claro = cv2.morphologyEx(claro, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(claro)
    if n <= 1:
        return np.zeros_like(claro)
    maior = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    comp = (rotulos == maior).astype(np.uint8)
    contornos, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    casca = cv2.convexHull(max(contornos, key=cv2.contourArea))
    disco = np.zeros_like(comp)
    cv2.fillPoly(disco, [casca], 1)
    return disco


def segmentar(rgb, lim_cor=8.0, lim_textura=3.0, devolver_etapas=False):
    # Limiares escolhidos olhando o arroz (o caso mais difícil, branco no branco):
    # ele fica ~13 acima do prato no canal b (amarelado) e tem textura 2-3,5
    # contra ~1,2 do prato liso. Com 14/9 (primeira versão) o arroz sumia.
    """Devolve (disco_do_prato, mascara_comida) como arrays 0/1."""
    disco = mascara_prato(rgb)
    # tira a borda do prato (sombra e aba)
    raio_borda = max(3, int(0.08 * np.sqrt(disco.sum() / np.pi)))
    interior = cv2.erode(disco, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * raio_borda + 1,) * 2))

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # cor de referência do prato: mediana dos pixels claros e lisos do interior
    media = cv2.blur(L, (9, 9))
    desvio = np.sqrt(np.maximum(cv2.blur(L * L, (9, 9)) - media ** 2, 0))
    ref = (interior > 0) & (desvio < np.percentile(desvio[interior > 0], 50))
    if ref.sum() < 50:
        ref = interior > 0
    a0, b0 = np.median(a[ref]), np.median(b[ref])

    dist_cor = np.sqrt((a - a0) ** 2 + (b - b0) ** 2)
    comida = ((dist_cor > lim_cor) | (desvio > lim_textura)) & (interior > 0)
    comida = comida.astype(np.uint8)

    comida = cv2.morphologyEx(comida, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    comida = cv2.morphologyEx(comida, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    # remove pedacinhos (reflexos, migalhas): menos de 0,3% do prato
    n, rot, stats, _ = cv2.connectedComponentsWithStats(comida)
    area_min = 0.003 * max(disco.sum(), 1)
    limpa = np.zeros_like(comida)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            limpa[rot == i] = 1
    # preenche buracos
    contornos, _ = cv2.findContours(limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros_like(limpa)
    cv2.drawContours(final, contornos, -1, 1, thickness=-1)

    if devolver_etapas:
        etapas = {"disco": disco, "dist_cor": dist_cor, "textura": desvio, "bruta": comida, "final": final}
        return disco, final, etapas
    return disco, final


def tira_reflexos(rgb, disco, mascara, lim_croma=10.0, lim_brilho=3.0, raio_borda=0.6):
    """Remove manchas que são reflexo de luz no prato, não comida.
    Reflexo = quase sem cor (distância a/b ao prato < lim_croma) E
    (mais claro que o prato OU perto da borda do prato).
    Medido nas fotos de um alimento: 307 de 310 manchas secundárias eram assim,
    e nenhuma mancha de comida (o arroz tem croma ~7, mas fica no centro e não é
    mais claro que o prato)."""
    mascara = mascara.astype(np.uint8)
    n, rot, _, centros = cv2.connectedComponentsWithStats(mascara)
    if n <= 1:
        return mascara
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = (disco > 0) & (mascara == 0)
    if ref.sum() < 50:
        return mascara
    L0, a0, b0 = (np.median(lab[:, :, i][ref]) for i in range(3))
    ys, xs = np.nonzero(disco)
    c0, R = np.array([xs.mean(), ys.mean()]), np.sqrt(disco.sum() / np.pi)
    manter, cromas = [], []
    for i in range(1, n):
        p = rot == i
        croma = np.median(np.sqrt((lab[:, :, 1][p] - a0) ** 2 + (lab[:, :, 2][p] - b0) ** 2))
        brilho = lab[:, :, 0][p].mean() - L0
        r = np.linalg.norm(centros[i] - c0) / R
        cromas.append(croma)
        if not (croma < lim_croma and (brilho > lim_brilho or r > raio_borda)):
            manter.append(i)
    if not manter:  # tudo parecia reflexo: fica a mancha mais colorida
        manter = [1 + int(np.argmax(cromas))]
    return np.isin(rot, manter).astype(np.uint8)


def so_principal(mascara, fracao=0.2):
    """Mantém a maior mancha e as que têm pelo menos `fracao` da área dela.
    Usado nas fotos de um alimento só, onde sobra reflexo na borda."""
    n, rot, stats, _ = cv2.connectedComponentsWithStats(mascara.astype(np.uint8))
    if n <= 1:
        return mascara
    areas = stats[1:, cv2.CC_STAT_AREA]
    manter = 1 + np.nonzero(areas >= fracao * areas.max())[0]
    return np.isin(rot, manter).astype(np.uint8)


def diametro(disco):
    """Diâmetro equivalente do prato em pixels (pela área)."""
    return 2 * np.sqrt(disco.sum() / np.pi)


def caixa(mascara, margem=0.05):
    """Retângulo (x0, y0, x1, y1) em volta da máscara, com margem relativa."""
    ys, xs = np.nonzero(mascara)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = int((x1 - x0) * margem), int((y1 - y0) * margem)
    h, w = mascara.shape
    return max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
