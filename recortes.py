"""Extração de recortes (patches) quadrados, usada no treino e na janela deslizante.

O tamanho do recorte é medido em fração do diâmetro do prato. Assim ele não
depende da distância da câmera: um recorte de 0,25 cobre a mesma "porção" de
prato em qualquer foto.
"""
import cv2
import numpy as np

LADO_REDE = 160          # lado do recorte que entra na CNN (pixels)
FRACAO_PADRAO = 0.25     # lado do recorte = 0,25 x diâmetro do prato

MEDIA = np.array([0.485, 0.456, 0.406], np.float32)   # normalização ImageNet
DESVIO = np.array([0.229, 0.224, 0.225], np.float32)


def recortar(rgb, cx, cy, lado, angulo=0.0, saida=LADO_REDE):
    """Recorte quadrado centrado em (cx, cy), girado `angulo` graus,
    redimensionado pra saida x saida. Borda fora da imagem vira reflexo."""
    cx, cy, escala = float(cx), float(cy), float(saida / lado)
    M = cv2.getRotationMatrix2D((cx, cy), float(angulo), escala)
    M[0, 2] += saida / 2 - cx
    M[1, 2] += saida / 2 - cy
    return cv2.warpAffine(rgb, M, (saida, saida), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def jitter_cor(patch, rng, forca=0.2):
    """Brilho, contraste, saturação e um pouco de matiz (pouco, porque cor importa)."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-4, 4)) % 180          # matiz +-8 graus
    hsv[:, :, 1] *= rng.uniform(1 - forca, 1 + forca)                 # saturação
    hsv[:, :, 2] *= rng.uniform(1 - forca, 1 + forca)                 # brilho
    p = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    c = rng.uniform(1 - forca, 1 + forca)                             # contraste
    p = (p - p.mean()) * c + p.mean()
    return np.clip(p, 0, 255).astype(np.uint8)


def recorte_aumentado(rgb, mascara, diam, rng, com_mascara=False):
    """Um recorte aleatório com todas as augmentations.
    Com com_mascara=True devolve também o recorte da máscara de comida (mesma geometria)."""
    ys, xs = np.nonzero(mascara)
    i = rng.integers(len(xs))
    lado = diam * rng.uniform(0.18, 0.32)
    ang = rng.uniform(0, 360)
    p = recortar(rgb, xs[i], ys[i], lado, angulo=ang)
    mc = recortar(mascara.astype(np.uint8) * 255, xs[i], ys[i], lado, angulo=ang) if com_mascara else None
    if rng.random() < 0.5:
        p = p[:, ::-1]
        mc = mc[:, ::-1] if com_mascara else None
    p = jitter_cor(p, rng)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        p = cv2.GaussianBlur(p, (k, k), 0)
    if com_mascara:
        return np.ascontiguousarray(p), mc > 127
    return np.ascontiguousarray(p)


def centros_grade(mascara, diam, fracao=FRACAO_PADRAO, passo=0.5, cobertura_min=0.3):
    """Centros de uma grade (passo = metade do recorte) cuja janela tem
    pelo menos `cobertura_min` de comida. Usado na avaliação e nos pratos."""
    lado = diam * fracao
    s = max(1, int(lado * passo))
    h, w = mascara.shape
    integral = cv2.integral(mascara.astype(np.uint8))
    centros = []
    meio = int(lado / 2)
    for cy in range(meio, h - meio + 1, s):
        for cx in range(meio, w - meio + 1, s):
            y0, y1, x0, x1 = cy - meio, cy + meio, cx - meio, cx + meio
            soma = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
            if soma >= cobertura_min * (2 * meio) ** 2:
                centros.append((cx, cy))
    if not centros:  # comida menor que a janela: usa o centro da comida
        ys, xs = np.nonzero(mascara)
        if len(xs):
            centros = [(int(xs.mean()), int(ys.mean()))]
    return centros, lado


def para_tensor(patches):
    """Lista de recortes uint8 HxWx3 -> array float N x 3 x H x W normalizado."""
    x = np.stack(patches).astype(np.float32) / 255.0
    x = (x - MEDIA) / DESVIO
    return x.transpose(0, 3, 1, 2).copy()


def recorte_misturado(a, b, rng, fracao=(0.15, 0.5)):
    """Augmentation de "prato misturado": junta dois alimentos no mesmo recorte.

    a e b são (rgb, mascara, diametro) de fotos de classes diferentes. Faz um
    recorte aumentado de cada e cola um pedaço de b sobre a, separado por uma
    borda ondulada em ângulo sorteado (parece dois alimentos encostados no prato).
    Devolve (recorte, f), onde f é a fração da COMIDA visível que veio de b
    (conta só pixel de comida, não o prato em volta).
    O rótulo vira suave: (1 - f) pra classe de a e f pra classe de b.
    """
    pa, ma = recorte_aumentado(*a, rng, com_mascara=True)
    pb, mb = recorte_aumentado(*b, rng, com_mascara=True)
    n = pa.shape[0]
    alvo = rng.uniform(*fracao)
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n - 0.5
    proj = xx * np.cos(ang) + yy * np.sin(ang)
    # borda ondulada: alimentos não se encontram em linha reta
    perp = -xx * np.sin(ang) + yy * np.cos(ang)
    onda = rng.uniform(0.02, 0.06) * np.sin(2 * np.pi * rng.uniform(1, 3) * perp + rng.uniform(0, 2 * np.pi))
    proj = proj + onda
    corte = np.quantile(proj, 1 - alvo)
    m = (proj > corte).astype(np.float32)
    m = cv2.GaussianBlur(m, (5, 5), 0)[:, :, None]          # costura suave
    p = (pa * (1 - m) + pb * m).astype(np.uint8)
    m = m[:, :, 0]
    comida_a = float((ma * (1 - m)).sum())
    comida_b = float((mb * m).sum())
    if comida_a + comida_b < 1:
        return p, float(m.mean())
    return p, comida_b / (comida_a + comida_b)
