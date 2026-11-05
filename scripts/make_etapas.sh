#!/bin/bash
# Genera un savestate por hito con el agente de reglas, encadenando tramos: cada uno parte del savestate del anterior.
cd "C:/Users/kwide/OneDrive/Documents/Repositorios Git/pokemon-ia"
export PYTHONIOENCODING=utf-8
E=runs/etapas
step() {  # step <desde> <hasta> <meta...>
  local from=$1 to=$2; shift 2
  echo "=== $to (desde $from) metas: $* ==="
  uv run python scripts/run_rules.py --savestate $from --goals "$@" --iterations 3000 --save-at-end $E/$to.ss0 2>&1 | grep -vE "^\s*banderas"
  [ -f $E/$to.ss0 ] || { echo "FALLO: no se creo $to"; exit 1; }
}
step runs/lab_mudkip.ss0           e1_oldale            0,10
step $E/e1_oldale.ss0              e2_ruta103_llegada   0,18
step $E/e2_ruta103_llegada.ss0     e3_ruta103_nivel8    grind:0,18:8
step $E/e3_ruta103_nivel8.ss0      e4_rival_vencido     talk:0,18:240:0.9
step $E/e4_rival_vencido.ss0       e5_pokedex           1,4
step $E/e5_pokedex.ss0             e6_petalburg         0,0
step $E/e6_petalburg.ss0           e7_wally             talk:8,1:129
step $E/e7_wally.ss0               e8_rustboro          0,3
step $E/e8_rustboro.ss0            e9_segundo_pokemon   catch:0,19:2:11,7
step $E/e9_segundo_pokemon.ss0     e10_gimnasio         11,3
echo "=== FIN ==="
