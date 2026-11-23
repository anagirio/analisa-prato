"""Junta as 3 sementes (0 = resultados/<nome>, 1 e 2 = <nome>_s1, _s2) em média ± desvio.

Uso: python agrega_sementes.py  ->  resultados/sementes.txt
"""
import json
import numpy as np

MODELOS = ["sem_aug", "com_aug", "mix_aug", "v2_sem_aug", "v2_com_aug", "v2_mix_aug"]
METRICAS = [("precisao_macro", "P macro"), ("recall_macro", "R macro"), ("f1_macro", "F1 macro"),
            ("f1_micro", "F1 micro"), ("acerto_exato", "exato"), ("jaccard_medio", "Jaccard")]


def carrega(m):
    return [json.load(open(f"resultados/{n}/pratos/metricas.json")) for n in (m, m + "_s1", m + "_s2")]


linhas = []
cab = f"{'modelo':<12}" + "".join(f"{r:>17}" for _, r in METRICAS)
linhas += ["Pratos (34 rajadas): média ± desvio padrão de 3 sementes (0, 1, 2)", cab]
todos = {m: carrega(m) for m in MODELOS}
for m in MODELOS:
    cel = ""
    for k, _ in METRICAS:
        v = np.array([d[k] for d in todos[m]])
        cel += f"{v.mean():>9.3f} ± {v.std(ddof=1):.3f}"
    linhas.append(f"{m:<12}" + cel)

linhas += ["", "F1 macro por semente"]
for m in MODELOS:
    linhas.append(f"{m:<12}" + "  ".join(f"{d['f1_macro']:.3f}" for d in todos[m]))

linhas += ["", "Diferença de F1 macro na mesma semente (quantas sementes a favor)"]
for a, b in [("com_aug", "sem_aug"), ("mix_aug", "com_aug"), ("mix_aug", "sem_aug"),
             ("v2_com_aug", "v2_sem_aug"), ("v2_mix_aug", "v2_com_aug"), ("v2_mix_aug", "v2_sem_aug"),
             ("v2_mix_aug", "mix_aug")]:
    d = np.array([x["f1_macro"] - y["f1_macro"] for x, y in zip(todos[a], todos[b])])
    linhas.append(f"{a} - {b}: média {d.mean():+.3f}  por semente {' '.join(f'{v:+.3f}' for v in d)}  "
                  f"({(d > 0).sum()}/3 a favor)")

txt = "\n".join(linhas)
open("resultados/sementes.txt", "w").write(txt + "\n")
print(txt)
