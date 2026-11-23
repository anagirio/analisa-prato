#!/bin/bash
# Treina cada configuração com sementes extras e avalia nos pratos.
# A semente 0 é o treino original (resultados/<nome>/).
set -e
flags() {
  case "$1" in
    sem_aug) echo "--sem-aug" ;;
    com_aug) echo "" ;;
    mix_aug) echo "--mistura 0.4" ;;
    v2_sem_aug) echo "--classe-prato --sem-aug" ;;
    v2_com_aug) echo "--classe-prato" ;;
    v2_mix_aug) echo "--classe-prato --mistura 0.4" ;;
  esac
}
source .venv/bin/activate
for s in 1 2; do
  for m in sem_aug com_aug mix_aug v2_sem_aug v2_com_aug v2_mix_aug; do
    n="${m}_s${s}"
    if [ -f "resultados/$n/pratos/metricas.json" ]; then echo "pulando $n"; continue; fi
    echo "== $n  $(date +%H:%M)"
    python treino.py --nome "$n" --semente "$s" $(flags $m) 2>&1 | grep -E "^ep 15|acc_teste|f1_macro_teste"
    python avalia_pratos.py --modelo "resultados/$n/modelo.pt" --nome "$n" 2>&1 | tail -3
  done
done
echo "FIM $(date +%H:%M)"
