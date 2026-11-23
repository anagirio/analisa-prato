# Analisa prato: o que tem no meu prato?

Projeto final de IF69D (Processamento Digital de Imagens, UTFPR, 2026.2).
Ana Luisa Girio.

Entrada: foto de cima de um prato. Saída: quais alimentos tem nele, quanto do
prato cada um ocupa e um mapa colorido mostrando onde cada um está.

Dataset: [bd_porcoes_alimentares](https://github.com/yurishiga/bd_porcoes_alimentares)
(16 alimentos fotografados sozinhos, mais 289 fotos de pratos com vários alimentos).

## Conteúdo da entrega

| O quê | Onde |
|---|---|
| Relatório (PDF) | `relatorio/relatorio_final.pdf` |
| Fonte do relatório (LaTeX, modelo da disciplina) | `relatorio/overleaf/` e `relatorio/relatorio_overleaf.zip` |
| Slides da apresentação | `slides/` |
| Código | arquivos `.py` na raiz (tabela "Arquivos" abaixo) |
| Modelo treinado usado na demo | `resultados/v2_mix_aug/modelo.pt` |
| Imagens de teste | `demo_apresentacao/` (4 pratos da demo) e `amostra_pratos/` (1 foto de cada um dos 34 pratos) |
| Rótulos manuais | `rotulos/pratos.csv` (alimentos de cada prato) e `rotulos/mascaras/` (20 contornos de comida) |
| Resultados e figuras | `resultados/` (métricas de todos os modelos) e `figuras/` |

## Rodar só a demo (sem baixar o dataset)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo.py demo_apresentacao/     # fecha a janela pra ir pra próxima foto
python demo.py minha_foto.jpg         # qualquer foto de cima de um prato branco
```

Os outros modelos (24 no total, ~45 MB cada) não estão no repositório; para
regerá-los, siga a seção "Uso" abaixo.

## Como funciona

1. **Segmentação clássica** (`segmentacao.py`): converte pra Lab, acha o prato
   branco com limiar de Otsu + maior componente conexo + envoltória convexa, e
   separa a comida por distância de cor ao prato (canais a/b) ou textura
   (desvio padrão local). Limpeza com abertura, fechamento e remoção de
   componentes pequenos.
2. **CNN em recortes** (`treino.py`): ResNet18 pré-treinada na ImageNet, ajustada
   com recortes quadrados de comida. O lado do recorte é medido em fração do
   diâmetro do prato (25%), então não depende da distância da câmera.
3. **Janela deslizante** (`analisa_prato.py`): a janela percorre o prato, a CNN
   classifica cada posição e as probabilidades são espalhadas nos pixels com
   peso gaussiano. Cada pixel fica com a classe mais provável.
   - v1: janelas só onde a segmentação clássica achou comida.
   - v2: uma 17ª classe "Prato" (prato vazio). A janela passa no prato inteiro;
     a borda da comida vem da segmentação clássica e a CNN veta as regiões que
     ela reconhece como prato vazio (reflexos, manchas). Usar só a CNN pra
     decidir o fundo não funcionou: janela meio comida, meio prato vira comida,
     e a classe "vaza" pro prato em volta.

## Instalação

Precisa de Python 3.10 a 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/yurishiga/bd_porcoes_alimentares dados
```

## Uso

```bash
# 1. segmenta, salva o cache e divide treino/val/teste por rajada de fotos
python prepara_dados.py

# 2. treina (cada um leva ~10 min num Mac M-series, mais em CPU)
python treino.py --nome com_aug
python treino.py --nome sem_aug --sem-aug
python treino.py --nome v2_com_aug --classe-prato
python treino.py --nome v2_sem_aug --classe-prato --sem-aug
python treino.py --nome mix_aug --mistura 0.4                 # + augmentation de prato misturado
python treino.py --nome v2_mix_aug --classe-prato --mistura 0.4   # modelo usado na demo

# 3. analisa pratos (salva figuras)
python analisa_prato.py dados/Imagens_Varios_Alimentos/prato_2_1.jpg --modelo resultados/v2_com_aug/modelo.pt

# 4. demo com janela na tela
python demo.py amostra_pratos/
python demo.py --vigiar ~/Desktop/demo_fotos     # analisa cada foto nova que chegar na pasta

# 5. avaliação nos pratos (precisa de rotulos/pratos.csv preenchido)
python avalia_pratos.py --modelo resultados/v2_com_aug/modelo.pt --nome v2_com_aug

# 5b. ajuste de limiares com validação cruzada por prato
python ajusta_limiares.py --modelo resultados/v2_mix_aug/modelo.pt --nome v2_mix_aug

# 6. Grad-CAM (pra onde a rede olhou)
python gradcam.py --modelo resultados/v2_com_aug/modelo.pt --saida figuras/gradcam.png
```

## Arquivos

| Arquivo | O que faz |
|---|---|
| `segmentacao.py` | segmentação clássica do prato e da comida |
| `prepara_dados.py` | cache das imagens, máscaras e divisão por rajada (`divisao.csv`) |
| `recortes.py` | extração de recortes e data augmentation |
| `treino.py` | treino e avaliação da CNN (acurácia, F1 macro, matriz de confusão) |
| `analisa_prato.py` | janela deslizante num prato |
| `avalia_pratos.py` | métricas multi-rótulo nos pratos rotulados |
| `ajusta_limiares.py` | ajuste de conf_min/area_min com validação cruzada por prato |
| `gera_figuras.py` | figuras de segmentação, classes e augmentation |
| `resultados/comparacao_pratos.txt` | tabela final comparando todos os modelos nos pratos |
| `gradcam.py` | mapas Grad-CAM |
| `demo.py` | demonstração ao vivo |
| `gera_folha_rotulos.py` | gera a folha e o CSV pra rotular os pratos |
| `rotulos/pratos.csv` | rótulos manuais (um por rajada = um prato) |
| `resultados/<modelo>/` | modelo, curvas, matriz de confusão, métricas |

## Detalhes que importam

- **Divisão por rajada:** as fotos foram tiradas em sequência (várias do mesmo
  prato, segundos de diferença). Dividir foto a foto vazaria fotos quase
  iguais entre treino e teste. Cada rajada vai inteira pra um conjunto só.
- **Reflexos de luz:** a segmentação às vezes marca o brilho do flash no prato
  como comida. Nas fotos de treino, `tira_reflexos` (em `segmentacao.py`)
  remove as manchas quase sem cor que são mais claras que o prato ou ficam
  perto da borda. Antes disso, 249 das 2011 máscaras tinham manchas extras;
  depois, nenhuma. Os resultados antigos estão em `resultados/_antes_reflexo/`.
- **Data augmentation:** posição e tamanho do recorte sorteados, rotação livre
  (foto de cima, qualquer ângulo vale), espelhamento, brilho, contraste,
  saturação, um pouco de matiz e desfoque.
- **Augmentation de prato misturado** (`--mistura`): uma parte dos recortes de
  treino junta dois alimentos de classes diferentes, separados por uma borda
  ondulada, com rótulo suave proporcional à comida visível de cada um. Imita
  alimentos encostados no prato, que a rede não via treinando com fotos de um
  alimento só.
- **Rótulos dos pratos:** feitos à mão por rajada. Alimentos fora das 16
  classes foram marcados como OUTRO e não entram nas métricas.
