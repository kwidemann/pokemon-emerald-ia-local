# Registro de corridas de entrenamiento

Una fila por corrida. Para comparar corridas, lo único que debe cambiar entre ellas es lo que se quiere probar: la
métrica de comparación son los **hitos** (cuántos, en cuántos pasos, con qué tasa de éxito), no la recompensa de
entrenamiento. Desde `ppo_500k` en adelante, `train.py` guarda además un `config.json` junto al modelo con los argumentos,
las recompensas y el commit; las dos primeras corridas se registran aquí a mano porque se lanzaron antes.

## Recompensas comunes a todas las corridas hasta ahora (R0)

| Qué | Valor |
|---|---|
| Cada hito del Tramo 1 (ids 3, 5 a 15) | +1,0 la primera vez que se cumple en el episodio |
| Hito 16 (medalla 1) | +5,0, y termina el episodio |
| Casilla nueva (mapa, x, y) en el episodio | `--new-tile-bonus`, ver tabla |
| Penalizaciones | **ninguna** |
| Confirmación | un hito necesita 3 lecturas seguidas (llega 2 pasos tarde); el mapa `(0,0)` necesita 2 lecturas |
| Lo ya cumplido en el savestate | no paga |

Observación: vector de 12 valores (mapa, posición, diálogo, batalla, equipo, HP y nivel del líder, medallas, dinero).
Algoritmo: PPO de Stable-Baselines3, `MlpPolicy` (2 capas de 64, `tanh`), CPU, hiperparámetros por defecto salvo los de la
tabla. Descuento `gamma` 0,99.

## Corridas

| Corrida | Fecha | Pasos | Instancias y puntos de partida | Bonus casilla | Episodio | `ent_coef` | `n_steps` | Resultado |
|---|---|---|---|---|---|---|---|---|
| `ppo_curriculum_tile` | 2026-11-02 | 100 000 | 3: laboratorio, sur de la Ruta 101, norte de la Ruta 101 (a 3 casillas de Oldale) | 0,005 | 1000 pasos | 0,01 | 512 | Hito 6 solo desde el punto a 3 casillas (22 de 33 episodios); 0 de 66 en los otros dos. Sin `Tracker` corregido: 2 falsos hito 9 |
| `ppo_500k` | 2026-11-02 | 500 000 | igual que la anterior | 0,005 | 1000 pasos | 0,01 | 512 | Terminada (498 episodios, ~2 h 10 min). Hito 6 (Oldale): **norte 125 de 166 (75 %)**, con tendencia al alza por quintos del entrenamiento (16, 28, 21, 29 y 31 de 33-34, o sea 48 % a 91 %); **laboratorio 0 de 166 y sur de la Ruta 101 0 de 166**. Ningún hito más allá del 6. Entropía de la política de 2,07 a 1,31; `explained_variance` 0,86 |

Código: `ppo_curriculum_tile` se lanzó con `Tracker` sin la corrección del mapa `(0,0)`; `ppo_500k` ya la incluye, sobre el
commit `c74e1d0` más cambios sin commit (varios savestates en `train.py` y esa corrección).

Los savestates de partida están en `runs/` (no versionados): `lab_mudkip.ss0` (se regenera con
`scripts/make_savestate.py`), `ruta101_sur.ss0` y `ruta101_norte.ss0` (creados con el agente de reglas).

### Evaluación de `ppo_500k` (modelo final, 10 episodios por caso, 1000 pasos máximo, 2026-11-02)

| Punto de partida | Acción más probable (determinista) | Muestreando (estocástica) |
|---|---|---|
| Laboratorio | 0 de 10 llegan a Oldale | 0 de 10 |
| Sur de la Ruta 101 | 0 de 10 | 0 de 10 |
| Norte de la Ruta 101 (a 3 casillas) | **0 de 10** | **7 de 10** |

Medalla 1 (hito 16): 0 en todos los casos. Lectura: la política "greedy" se queda atascada pulsando siempre el botón más
probable (ARRIBA) contra la pared, y solo llega a Oldale si se le deja el azar del muestreo. Es una sola evaluación de 10
episodios por caso, sin repetir. Archivo de salida: `runs/eval_ppo_500k.txt` (no versionado).

## Corrida con el agente de reglas, para comparar

| Agente | Fecha | Partida | Recompensa | Resultado |
|---|---|---|---|---|
| Reglas (`eval_rules.py`) | 2026-11-02 | `lab_mudkip.ss0`, 10 ejecuciones | no aplica (no aprende) | 10 de 10 hasta la medalla 1; mediana 167 489 frames |

## Variantes previstas

Una variante por vez, con los mismos pasos, instancias y savestates que `ppo_500k`:

- **Calibración (corrida corta, sin castigos):** el entorno ya lleva contadores por episodio (`info["contadores"]`, también en
  `train.log` y TensorBoard): `choques`, `quieto`, `bucles`, `rev3`/`rev10`/`rev30`, `menu`, `dialogo`, `batalla`,
  `casillas`. Se lanza una corrida corta con castigos en 0 para ver cuánto ocurre cada cosa y fijar los valores. Las corridas
  anteriores no los tienen.
- **R1 (castigo al tiempo perdido, por calibrar):** `--bump-penalty` (chocar), `--idle-penalty` (quedarse quieto: A, B,
  START o nada con el personaje controlable), `--loop-penalty` (volver a una casilla ya pisada 2 veces o más en las últimas 30
  casillas; volver a propósito a un sitio no cuenta) y `--menu-penalty` (pasos en menús). Todos valen 0 por defecto. Esto
  sustituye a la idea de "premio por acercarse al siguiente hito", que queda para una R3.
- **R2 (R1 más memoria a corto plazo en la observación):** última acción, si se movió y dónde estuvieron las últimas casillas.
  Comparar R1 con R2 mide cuánto aporta la memoria.
