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
| `ppo_r1_castigos` | 2026-11-02 | 500 000 | igual que `ppo_500k` | 0,005 | 1000 pasos | 0,01 | 512 | Terminada (498 episodios, ~1 h 55 min). Castigos: choque 0,002, quieto 0,002, menú 0,002, bucle 0,005; giros 0. Hito 6 (Oldale): **norte 109 de 166 (66 %)**, por tercios 19, 45 y 45 de 55-56; **laboratorio 0 de 166 y sur 0 de 166**. Ningún hito más allá del 6. Quieto y menús bajaron un 80-90 % respecto a la calibración, pero los giros subieron (349 a 420-630 por episodio): el tiempo perdido se desplazó a girar |
| `ppo_r3_acercamiento` | 2026-11-02 | 500 000 | igual que `ppo_500k` | 0,005 | 1000 pasos | 0,01 | 512 | Terminada (498 episodios, ~1 h 53 min). Hito 6 (Oldale): **norte 133 de 166 (80 %)** (27, 53 y 53 de 55-56 por tercios), **sur 2 de 166**, laboratorio 0 de 166; ningún hito más allá del 6. Distancia mínima a Oldale por tercios: laboratorio 54, 40 y 29 (de 65), sur 22, 22 y 23 (de 35), norte 3, 0,2 y 0,2 (de 8). R1 (los mismos castigos) más **premio por acercarse**: 0,01 por casilla de distancia a pie hacia Oldale que se reduce, y −0,01 por cada una que se aleja (tope de ±3 casillas por paso). Distancia inicial: laboratorio 65, sur 35, norte 8 casillas |
| `ppo_r3b_choques_escalados` | 2026-11-02 | 500 000 | igual que `ppo_500k` | 0,005 | 1000 pasos | 0,01 | 512 | Terminada (498 episodios, ~2 h 13 min). Hito 6 (Oldale): **norte 130 de 166 (78 %)** (29, 46 y 55 de 55-56 por tercios), **sur 1 de 166**, laboratorio 0 de 166; ningún hito más allá del 6, igual que R3. Choques por episodio −40 % a −55 % respecto a R3 (laboratorio 71, sur 87, norte 59), pero los giros subieron (631, 575 y 367) y la racha de choques más larga llega al tope (~10-12). Distancia mínima a Oldale por tercios: laboratorio 56, 43 y 30, sur 22, 20 y 22, norte 2,3, 0,7 y 0. R3 (castigos de R1 más premio por acercarse) con **un solo cambio**: el castigo por choque se multiplica por los choques seguidos en la misma casilla, hasta 10 veces (de 0,002 a 0,02 por paso); la racha se reinicia al moverse a otra casilla. Motivo: en la evaluación determinista de R3 el agente se quedó pulsando ARRIBA contra un desnivel en 9 de 9 episodios |
| `ppo_r3c_castigo_giros` | 2026-11-03 | 500 000 | igual que `ppo_500k` | 0,005 | 1000 pasos | 0,01 | 512 | Terminada (498 episodios, ~1 h 54 min). Hito 6 (Oldale): norte 116 de 166 (70 %; R3b 130), sur 0 de 166, laboratorio 0 de 166; ningún hito más allá del 6. Los giros bajaron (laboratorio 631 a 35, sur 575 a 237, norte 367 a 152) pero **el agente aprendió a esconderse en los diálogos**, que no llevan castigo: en los últimos 55 episodios del laboratorio gasta ~989 de 1000 pasos en diálogo, pisa 1 casilla y la distancia mínima a Oldale se queda en 65 (sin avance); en sur y norte los pasos en diálogo y 'quieto' también subieron (quieto 240 y 125 por episodio). R3b con **un solo cambio**: cada giro en el lugar (la pulsación solo gira al personaje, sin moverlo) cuesta 0,002, igual que un choque fijo; el castigo por choque escalado (tope 10) sigue igual. Motivo: en R3b los choques bajaron 40-55 % pero los giros subieron (367 a 631 por episodio) y el atasco determinista siguió |
| `ppo_r3d_castigo_dialogos` | 2026-11-03 | 500 000 | igual que `ppo_500k` | 0,005 | 1000 pasos | 0,01 | 512 | En curso. **R3b** (sin castigo por giros) con **un solo cambio**: cada paso con un diálogo o script activo cuesta 0,002. Motivo: en R3c el agente se escondió en el diálogo del laboratorio (~989 de 1000 pasos), que no llevaba castigo |

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

### Evaluación de `ppo_r1_castigos` (modelo final, 10 episodios por caso, 1000 pasos máximo, 2026-11-02)

| Punto de partida | Determinista | Estocástica |
|---|---|---|
| Laboratorio | 0 de 10 llegan a Oldale | 0 de 10 |
| Sur de la Ruta 101 | 0 de 10 | 0 de 10 |
| Norte de la Ruta 101 (a 3 casillas) | **0 de 10** | **7 de 10** |

Idéntico a `ppo_500k` en los seis casos: los castigos no cambiaron el resultado por hitos. Medalla 1: 0 en todos. Archivo de
salida: `runs/eval_ppo_r1_castigos.txt` (no versionado).

### Evaluación de `ppo_r3_acercamiento` (modelo final, 10 episodios por caso, 1000 pasos máximo, 2026-11-02)

| Punto de partida | Determinista: Oldale (distancia mínima) | Estocástica: Oldale (distancia mínima) |
|---|---|---|
| Laboratorio (inicial 65) | 0 de 10 (56 en los 10) | 0 de 10 (24 a 55; mejor 24) |
| Sur de la Ruta 101 (inicial 35) | 0 de 10 (30 en los 10) | 0 de 10 (21 a 27) |
| Norte de la Ruta 101 (inicial 8) | 0 de 10 (7 en los 10) | **10 de 10** |

Lectura: en modo determinista los 10 episodios de cada caso son idénticos: el agente se queda pulsando la misma tecla (ARRIBA
contra el desnivel de la casilla (10,14) en el sur) y no se mueve más. Con muestreo, el norte pasó de 7 de 10 (`ppo_500k` y R1)
a 10 de 10, y el laboratorio y el sur se acercan más que antes (no hay medida de distancia anterior). Medalla 1: 0 en todos.
Archivo de salida: `runs/eval_ppo_r3_acercamiento.txt` (no versionado).

### Evaluación de `ppo_r3b_choques_escalados` (modelo final, 10 episodios por caso, 1000 pasos máximo, 2026-11-02)

| Punto de partida | Determinista: Oldale (distancia mínima) | Estocástica: Oldale (distancia mínima media) |
|---|---|---|
| Laboratorio (inicial 65) | 0 de 10 (53 en los 10) | 0 de 10 (26,7) |
| Sur de la Ruta 101 (inicial 35) | 0 de 10 (35 en los 10) | 0 de 10 (21,4) |
| Norte de la Ruta 101 (inicial 8) | 0 de 10 (5 en los 10) | **10 de 10** |

Lectura: igual que R3 en llegadas a Oldale (el castigo escalado no abrió nada nuevo). En modo determinista los 10 episodios de cada
caso vuelven a ser idénticos y el agente se queda atascado (en el laboratorio, contra la esquina de una casa de Pueblo Hojarasca, a 53 casillas). Con muestreo, la
distancia mínima es parecida a la de R3 (laboratorio 24 a 55 y sur 21 a 27 en R3). Medalla 1: 0 en todos. Archivo de salida:
`runs/eval_ppo_r3b_choques_escalados.txt` (no versionado).

### Evaluación de `ppo_r3c_castigo_giros` (modelo final, 10 episodios por caso, 1000 pasos máximo, 2026-11-03)

| Punto de partida | Determinista: Oldale (distancia mínima) | Estocástica: Oldale (distancia mínima) |
|---|---|---|
| Laboratorio (inicial 65) | 0 de 10 (65 en los 10) | 0 de 10 (65 en los 10) |
| Sur de la Ruta 101 (inicial 35) | 0 de 10 (35 en los 10) | 0 de 10 (22 a 29) |
| Norte de la Ruta 101 (inicial 8) | 0 de 10 (7 en los 10) | 9 de 10 |

Lectura: peor que R3b. El laboratorio no avanza ni una casilla ni con muestreo (atrapado en el diálogo del inicio) y el norte pierde un
episodio. El castigo por girar quitó los giros pero no mejoró ninguna llegada. Archivo de salida: `runs/eval_ppo_r3c_castigo_giros.txt`
(no versionado).

### Calibración de los contadores (`calibracion`, 2026-11-02)

30 000 pasos, 3 instancias, mismos puntos de partida y bonus 0,005 que `ppo_500k`, castigos en 0, política casi aleatoria
(los primeros pasos del entrenamiento). Media por episodio de 1000 pasos, 10 episodios por punto de partida (una primera
versión, `calibracion_v1_sin_giros`, mezclaba giros con choques y se descartó):

| Contador | Laboratorio | Sur Ruta 101 | Norte Ruta 101 | Media |
|---|---|---|---|---|
| Choques (pulsó hacia donde ya miraba y no se movió) | 213 | 192 | 156 | 187 |
| Giros (solo giró; no se castiga) | 369 | 367 | 311 | 349 |
| Quieto (A, B, START o nada, con el personaje controlable) | 256 | 266 | 230 | 251 |
| Menús (pasos en pantallas de menú) | 100 | 96 | 108 | 101 |
| Bucles (2.ª visita o más en las últimas 30 casillas) | 20 | 26 | 20 | 22 |
| Vueltas a las últimas 3 / 10 / 30 casillas | 20 / 24 / 26 | 23 / 34 / 35 | 19 / 26 / 27 | 21 / 28 / 29 |
| Batalla / diálogo | 0 / 11 | 0 / 4 | 137 / 4 | 46 / 6 |
| Casillas distintas pisadas | 16 | 31 | 21 | 23 |

Lectura: de cada 1000 pasos, unos 50-75 mueven al personaje (por resta); el resto son giros (35 %), quieto (25 %), choques
(19 %) y menús (10 %). Los bucles son poco frecuentes (2 %). El agente aleatorio malgasta casi todo su tiempo en no avanzar.

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
- **R3 (R1 más premio por acercarse al siguiente hito, lanzada):** `--shaping-coef`. La meta es el mapa del primer hito de
  `SHAPING_GOALS` aún sin cumplir (6 Oldale, 7 Ruta 103, 9 Petalburg, 12 Rustboro, 15 gimnasio; el 5 se omite porque los
  puntos de partida ya lo cumplieron en la partida real). La distancia sale de un grafo de casillas construido solo con la ROM
  (`env/distance.py`: pasos, saltos de desnivel en un sentido, warps y conexiones; sin agua ni NPC) y se comprobó en vivo con
  `scripts/check_distance.py`: caminando por el mejor camino del grafo, desde el norte llegan a Oldale en 8 pasos (8 previstos) y
  desde el sur en 35 (35 previstos), y desde el laboratorio se cumple hasta la Ruta 101 (los cortes fueron combates salvajes).
  Es una función potencial: ir y volver suma cero. Nuevos contadores por episodio: `dist_ini`, `dist_min`, `dist_fin`. Con
  esto se mide el progreso aunque no llegue al hito. También se imprimen al evaluar.
- **R2 (R1 más memoria a corto plazo en la observación):** última acción, si se movió y dónde estuvieron las últimas casillas.
  Comparar R1 con R2 mide cuánto aporta la memoria.

- **R4 (observación con terreno y memoria; preparada, sin lanzar):** `--obs-terrain` añade a los 12 valores un parche de 5x5
  casillas alrededor del jugador (norte arriba, 6 clases en one-hot: libre, puerta y desnivel hacia cada dirección; sale de la
  ROM, sin NPC, árboles ni rocas) y `--obs-memory` añade memoria corta (casillas del parche ya pisadas, últimas 4 acciones,
  hacia dónde mira el personaje y resultado del último paso con la racha de choques / 10). Tamaños: básica 12, terreno 162,
  memoria 77, ambas 227. Por defecto las dos valen False, así que los modelos anteriores se siguen evaluando igual (`--evaluate`
  lee la observación del `config.json` junto al modelo). Motivo: el agente no ve paredes ni desniveles (se pega al desnivel del
  sur y a la esquina de la casa) y no recuerda lo que acaba de hacer (R1, R3b y R3c: castigos sin mejora). Referencia para
  comparar: la mejor corrida sin cambio de observación (R3, o R3b/R3d si resulta mejor). Orden sugerido, un cambio por vez:
  R4a solo `--obs-terrain`, R4b solo `--obs-memory`, R4c las dos.
