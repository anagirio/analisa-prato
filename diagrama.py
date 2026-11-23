"""Diagrama em blocos do pipeline (treino em cima, análise de um prato embaixo).

Uso: python diagrama.py   ->  figuras/diagrama_blocos.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

CORES = {
    "dado": ("#eeeeee", "#777777"),
    "pdi": ("#dbe8f7", "#2f6db5"),
    "cnn": ("#dcf0dc", "#2e8b3e"),
    "saida": ("#fdebd3", "#d07a12"),
}
L, A = 2.55, 1.25  # largura e altura dos blocos


def bloco(ax, x, y, titulo, detalhe, tipo):
    fundo, borda = CORES[tipo]
    ax.add_patch(FancyBboxPatch((x - L / 2, y - A / 2), L, A, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fundo, ec=borda, lw=1.6))
    ax.text(x, y + 0.3, titulo, ha="center", va="center", fontsize=9.5, weight="bold", color="#222")
    ax.text(x, y - 0.17, detalhe, ha="center", va="center", fontsize=7.9, color="#333", linespacing=1.3)


def seta(ax, x0, y0, x1, y1, texto=None, **kw):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.4, mutation_scale=14, **kw))
    if texto:
        ax.text((x0 + x1) / 2 + 0.12, (y0 + y1) / 2, texto, fontsize=8.2, color="#444", ha="left", va="center",
                style="italic")


def main():
    fig, ax = plt.subplots(figsize=(16, 5.2))
    ax.set_xlim(0, 18.4); ax.set_ylim(0, 5.6); ax.axis("off")

    xs = [1.4 + i * 3.1 for i in range(6)]
    yt, ya = 4.3, 1.5
    ax.text(0.1, 5.35, "Treino (fotos de um alimento só)", fontsize=12, weight="bold", color="#222")
    ax.text(0.1, 2.5, "Análise de um prato", fontsize=12, weight="bold", color="#222")

    treino = [
        ("Fotos de 1 alimento", "2011 fotos, 16 classes\ndivisão por rajada\n(treino/val/teste)", "dado"),
        ("Segmentação clássica", "Lab + Otsu no canal L\ndistância de cor, textura\nmorfologia, tira reflexos", "pdi"),
        ("Recortes", "lado = 25% do diâmetro\ndo prato (escala fixa)", "pdi"),
        ("Data augmentation", "rotação, espelho, cor, blur\n+ colagem de 2 alimentos\n(rótulo suave)", "cnn"),
        ("ResNet18", "pré-treinada na ImageNet\najuste fino, 15 épocas\nv2: + classe Prato", "cnn"),
        ("Modelo treinado", "melhor época\npela validação", "saida"),
    ]
    analise = [
        ("Foto do prato", "vista de cima\nqualquer distância", "dado"),
        ("Segmentação clássica", "disco do prato\nmáscara de comida\n(mesmos limiares)", "pdi"),
        ("Janela deslizante", "passo = 1/4 da janela\nCNN classifica\ncada posição", "cnn"),
        ("Acúmulo gaussiano", "probabilidade por pixel\nv2: CNN veta \"Prato\"", "cnn"),
        ("Filtro", "confiança ≥ 0,5\nárea ≥ 4% da comida", "pdi"),
        ("Resultado", "lista de alimentos\n% do prato de cada um\nmapa colorido", "saida"),
    ]
    for linha, y in ((treino, yt), (analise, ya)):
        for i, (t, d, tipo) in enumerate(linha):
            bloco(ax, xs[i], y, t, d, tipo)
            if i:
                seta(ax, xs[i - 1] + L / 2, y, xs[i] - L / 2, y)

    # o modelo treinado desce pra janela deslizante
    seta(ax, xs[5], yt - A / 2, xs[2] + 0.3, ya + A / 2, connectionstyle="arc3,rad=-0.12")
    ax.text(13.3, 3.1, "pesos da rede", fontsize=8.5, color="#444", style="italic")

    legenda = [("dado", "dados"), ("pdi", "PDI clássico"), ("cnn", "CNN / aprendizado"), ("saida", "saída")]
    for i, (tipo, nome) in enumerate(legenda):
        fundo, borda = CORES[tipo]
        x = 0.3 + i * 2.6
        ax.add_patch(FancyBboxPatch((x, 0.05), 0.35, 0.25, boxstyle="round,pad=0.01", fc=fundo, ec=borda, lw=1.3))
        ax.text(x + 0.5, 0.17, nome, fontsize=9, va="center", color="#333")

    fig.savefig("figuras/diagrama_blocos.png", dpi=200, bbox_inches="tight", facecolor="white")
    print("ok: figuras/diagrama_blocos.png")


if __name__ == "__main__":
    main()
