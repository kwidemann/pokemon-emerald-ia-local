# Bitácora de pokemon-ia

Registro cronológico de lo que se hizo, lo que se decidió y lo que se descubrió. Las especificaciones y el diseño viven en [`specs.md`](../specs.md); aquí va el "qué pasó y cuándo".

**Cómo se mantiene:** al terminar cada paso se agrega una entrada al final de su día (más reciente abajo) con *Hecho*, *Verificado*, *Decisiones* y *Pendiente*. Lo que no se comprobó en vivo se dice. Cada entrada de un paso que cambió archivos termina con el commit que la contiene, cuando el usuario decide hacerlo.

---

## 2026-11-01

### Diseño y decisiones iniciales
- **Objetivo:** proyecto educativo local en el que agentes de IA juegan Pokémon Esmeralda (GBA), orquestados con scripts de Python invocables desde Claude Code, aprovechando la GPU y la CPU.
- **Enfoques a comparar:** (1) reglas / máquina de estados, (2) aprendizaje por refuerzo (PPO), (3) modelo de mundo tipo JEPA sobre la RAM, (4) híbrido: ejecutor + Laya (decisor rápido local) + Ollama (supervisor lento).
- **Decisiones del usuario:** emulador mGBA con puente Lua por socket; observación estructurada leída de la RAM; meta a largo plazo de completar el juego mediante hitos; JEPA sobre RAM y no sobre píxeles; supervisor textual con `qwen2.5:7b` o `gemma:7b`; se acepta el diseño híbrido de dos niveles.
- **Hardware:** RTX 2060 Max-Q de 6 GB, i7-10750H (6 núcleos), 11,8 GB de RAM.

### Fase 1: fundaciones
- **Hecho:** proyecto `uv` con Python 3.12 (el 3.13 del sistema se evitó); mGBA 0.10.5 instalado con winget; `emulator/server.lua` (servidor TCP en el puerto 8888), `emulator/client.py` (`MGBAClient`) y `scripts/ping.py`.
- **Verificado con la ROM real:** `ping`, `frame`, `step`, `readrange`, `savestate`, `loadstate` y `screenshot`.
- **Descubierto:** `mGBA.exe --script` no existe en 0.10.5 y el programa se cierra, así que el Lua se carga a mano en cada sesión (*Tools > Scripting > File > Load script*). `step N` avanza N+1 frames, una constante que no afecta el determinismo.
- **Incidente:** el parser de rutas cortaba en el primer espacio y mGBA escribió un archivo suelto llamado `Repositorios` en `Documents`. Se borró y se corrigió `server.lua`; la corrección **aún no está cargada** en mGBA, y mientras tanto las capturas de pantalla vuelven a dejar ese archivo suelto, que se borra a mano.

### Fase 2: mapa de RAM
- **Hecho:** `env/ram_map.py` (direcciones con su fuente y estado de verificación), `env/state.py` (`read_state` y `parse_party_mon`) y `scripts/inspect_state.py`.
- **ROM:** SHA-1 `f3ae0881…07b7`, idéntico al de pret/pokeemerald (v1.0 USA/Europa, `BPEE`).
- **Descubierto:** los save blocks cambian de dirección en cada arranque; se leen siempre a través de `gSaveBlock1Ptr` y `gSaveBlock2Ptr`.
- **Verificado en vivo:** posición, mapa, dinero (3000, con el XOR de la clave de cifrado), nombre del jugador y estado de script (0 con texto abierto, 2 sin script).

### Automatización de la intro y verificación del equipo
- **Hecho:** `scripts/drive.py` (`Driver` con `advance_dialogue`, `move`, `press`). Con él el agente llevó la partida desde Littleroot hasta el laboratorio de Birch: casa, reloj de pared, casa del vecino y su cuarto, Ruta 101, elección de **Mudkip** (elegido por el agente), primera batalla y laboratorio.
- **Verificado en vivo:** el descifrado del Pokémon (`species=283`, nivel 5, HP 21/21, Placaje y Gruñido, `checksum_ok=True`); los cambios tras la batalla (PP 35→32, HP 21→18, experiencia 135→152); `in_battle` pasa a `True` al empezar y a `False` al terminar; `gBattleTypeFlags = 0x14` en la primera batalla; `loadstate` devuelve la posición original.
- **Descubierto:** un NPC bloquea la salida norte de Littleroot ("Um, um, um!") hasta visitar la casa del vecino. En la pantalla del reloj el cursor arranca en NO. Una pulsación corta (1–6 frames) mueve exactamente una casilla; mantener la tecla mientras se lee la posición hace avanzar varias. El emulador corre a 60 fps, tiempo real.
- **Savestate inicial:** `C:\AI\pokemon-ia-states\lab_mudkip.ss0` (laboratorio, `(6,5)`, Mudkip nivel 5). Está fuera del proyecto por el problema de las rutas con espacios.

### Fase 3: entorno Gymnasium (primera versión)
- **Hecho:** `readmulti` en `server.lua` y `MGBAClient.read_multi` (con caída automática al método antiguo si el Lua cargado es viejo); `read_state` ahora usa 2 llamadas; `env/milestones.py`, `env/pokemon_env.py` y `scripts/random_agent.py`.
- **Verificado:** `reset()` desde el savestate marca los hitos 3 y 5 sin recompensa; 60 pasos del agente aleatorio sin errores; una acción mueve una casilla si el personaje ya mira en esa dirección, y si no solo lo gira.
- **Medido:** 1,7 pasos/s a velocidad real (20 frames por acción, 60 fps). Con tiempo real no se puede entrenar.
- **Sin verificar:** `readmulti` en el servidor (el Lua cargado en mGBA es el viejo y se usó la caída automática), y los IDs de mapa inferidos de Oldale, Ruta 103, Petalburg y Rustboro.
- **Después de recargar el Lua y activar el avance rápido:** `readmulti` y las rutas con espacios funcionan; el savestate se movió a `runs/lab_mudkip.ss0`. Velocidad: ~52 pasos/s (antes 1,7). Se descubrió que al cruzar una puerta la ubicación se lee un paso como `(0,0)` en `(0,0)`, lo que disparó el hito de Petalburg por error; se añadió `GameState.in_transition` y el entorno lo ignora. Las rutas relativas del savestate fallaban porque mGBA las resuelve desde su directorio; el entorno ahora las vuelve absolutas.
- **Generador del savestate (`scripts/make_savestate.py`):** comando `reset` en el Lua; el script juega la intro desde el logo hasta el laboratorio en ~16 s y se comprueba contra la RAM en cada tramo. Dos ejecuciones seguidas funcionaron y el `.sav` quedó intacto (mismo MD5). Se descubrió que con el avance rápido el emulador sigue corriendo (~2200 fps) entre comandos, así que el RNG no es reproducible: el flujo sí, los valores no. Problemas resueltos por el camino: la pantalla de nombre perdía la primera pulsación (se espera el fundido), el camión no deja moverse hasta que termina su sacudida (~900 frames), el SÍ/NO del reloj arranca en NO y se detecta por píxeles, y mamá del vecino vuelve a hablar al bajar la escalera.
- **Sin verificar:** el arranque sin `.sav` previo; la reproducibilidad exacta.
- **Pendiente:** verificar los detectores de hitos; paralelizar emuladores.

### Fase 4: agente de reglas
- **Hecho:** `env/mapdata.py` (colisiones, NPC, warps, conexiones, hierba, desniveles; BFS), `env/world.py` (grafo de 518 mapas con zonas conexas, rutas, NPC, centros Pokémon), `env/battle_data.py` (movimientos y tipos), `agents/rules/agent.py` (navegar, pelear, curar, hablar, entrenar, capturar), `scripts/run_rules.py` (con `--tramo1`). `GameState` ganó `callback2`, `in_overworld` y `flags`; el Lua ganó `reg`.
- **Verificado:** una ejecución continua desde el savestate del laboratorio completó los **13 hitos del Tramo 1**: medalla 1 en 2 min 40 s de reloj (191 818 frames, 37 combates). Datos de la ROM (movimientos, tipos, grafo) contrastados con el juego.
- **Descubierto:** (1) tras hablar con un NPC, una `A` de más lo vuelve a "hablar"; se usa B. (2) El cursor del menú de acciones recuerda HUIR. (3) El **cuelgue** de combate no era del emulador: el juego graba las acciones en un búfer y un bucle con contador de 8 bits no termina si se acumulan 256 bytes o más (el agente había elegido HUIR cientos de veces); se encontró leyendo el `pc` con el comando `reg`. El agente ahora vigila el búfer. (4) En combates dobles el agente mira ambos combatientes del jugador; `gActiveBattler` no es fiable (valía 4 con la mochila abierta). (5) La bolsa de Poké Balls está en `SaveBlock1+0x650`. (6) Flags: `0x861` Pokédex, `0x867` medalla 1.
- **Evaluación (10 ejecuciones, `scripts/eval_rules.py`):** la primera tanda dio 10 de 10, pero una ejecución era un falso positivo (medalla leída en Petalburg con un Mudkip de nivel 10); se añadió `MilestoneMonitor` (3 lecturas seguidas) y `GameState.sane`, y se repitió: **7 de 10 (70 %)**, mediana de 176 480 frames y 162 s de reloj en los éxitos. Los 3 fallos son el mismo: el agente se queda sin Poké Balls en la captura. Confirmado en una repetición aparte (bolsa vacía, sin captura).
- **Arreglo del fallo de captura:** `RulesAgent.buy_balls` compra hasta 8 Poké Balls en la tienda de Rustboro `(11,7)` antes de capturar (meta `catch:0,19:2:11,7`; `talk_to` ganó `settle_after=False` para seguir la conversación de la tienda). Flujo verificado en vivo con capturas de pantalla; el dinero baja de ₽2584 a ₽984.
- **Segunda evaluación (10 ejecuciones):** **10 de 10** hasta la medalla 1, mediana de 167 489 frames y 144 s de reloj.
- **Pendiente:** identificar la lectura falsa de banderas; repetir con más ejecuciones o puntos de partida; debilitar al salvaje antes de lanzar (hoy se lanza a HP completo).

### Control de versiones y organización
- **Hecho:** `git init` con rama `main`. Se creó esta bitácora y se añadió a `specs.md` la sección de hitos propuestos. A pedido del usuario se hizo el primer commit del proyecto (el hash se ve con `git log`).
- **Publicación en GitHub:** el equipo no tiene `gh` instalado, así que el repositorio público lo crea el usuario en GitHub y después se hace el `push` desde aquí.
- **Hitos:** el usuario aprobó la lista de 16 hitos del Tramo 1 (`specs.md`, 9.1). El orden de las rutas viene de Bulbapedia, pero la página no se pudo descargar completa (403), así que los detalles marcados **[memoria]** (Pokédex, Wally, Team Aqua, nivel de evolución y equipo de Roxanne) quedan por verificar jugando.
- **Decisiones:** se usará **solo Laya** (no Jev real por API); se versiona el proyecto con git; no se hacen commits sin que el usuario lo pida.
- **Pendiente:**
  - Elegir hitos definitivos y su orden (propuesta en `specs.md`, sección 9.1).
  - Avance rápido sin límite en mGBA y nueva medición de velocidad.
  - Recargar `server.lua` y mover el savestate a `runs/`.
  - Comando `readmulti` en el Lua.
  - Verificar medallas y batallas contra entrenadores.
  - Fase 3: entorno Gymnasium.

## 2026-11-02

### Fase 5: RL con PPO (arranque)
- **Hecho:** `env/pokemon_env.py` usa `MilestoneMonitor` (3 lecturas seguidas por hito); `emulator/server.lua` lee el puerto de `POKEMON_IA_PORT` para poder abrir varias instancias de mGBA. El entorno se probó sin emulador con estados falsos.
- **Sin verificar en vivo:** el entorno con el emulador y que `os.getenv` exista en el Lua de mGBA.
- **Decidido:** Stable-Baselines3 con PyTorch en CPU (`uv sync --extra rl`); `agents/rl/train.py` escrito y probado con un emulador falso.
- **Probado en vivo:** lancé mGBA desde la terminal con una copia de la ROM, cargué `server.lua` y puse el avance rápido sin límite automatizando los menús (UI Automation); el puerto de `POKEMON_IA_PORT` funciona (8899); ~3300 frames/s; entorno y PPO funcionan con 1 emulador (~51 pasos/s). El filtro de ESC y los diálogos modales requieren cuidado (ver specs 7.6).
- **Lanzador de N emuladores:** `scripts/launch_emulators.py` y `emulator/setup_instance.ps1` abren, cargan el script y aceleran N instancias automáticamente. Costó tres arreglos (primera letra perdida con `SendKeys`, UI Automation colgado con el diálogo abierto, precedencia de la coma en PowerShell). Con 4 instancias PPO va a ~76 pasos/s (1,5 veces una sola): el límite parece ser la CPU. Medición por instancias (pasos/s): 1 ~51, 2 ~58, 3 ~72, 4 ~75-77, con 4 la CPU al 80 %.
- **Primer entrenamiento PPO (3 instancias, 100 000 pasos, bonus por casilla 0,005, tres savestates):** solo llegan a Oldale los episodios que parten a 3 casillas (22 de 33); desde el laboratorio y desde el sur de la Ruta 101, 0 de 66. Sin mejora visible con el tiempo. Falso positivo del hito 9 (lectura transitoria de `(0,0)`) en 2 episodios: corregido en `Tracker`.
- **Plan acordado:** entrenar con varias instancias en paralelo (una por puerto, `SubprocVecEnv`). Falta resolver cómo cargar el script sin hacerlo a mano en cada ventana (`mGBA.exe` no acepta `--script`).
- **Evaluación del agente de reglas con compra de Poké Balls:** 10 de 10 hasta la medalla 1 (incluida arriba, en la Fase 4).

### Fase 5: PPO de 500 000 pasos (`ppo_500k`) y diagnóstico del atasco
- **Hecho:** corrida de referencia `ppo_500k` (3 emuladores: laboratorio, sur de la Ruta 101 y norte de la Ruta 101 a 3 casillas de Oldale; bonus por casilla nueva 0,005; episodios de 1000 pasos; sin castigos). ~2 h 10 min, 498 episodios. Cada corrida guarda ahora un `config.json` junto al modelo (argumentos, recompensas, commit).
- **Resultado:** el hito 6 (Oldale) lo logra **solo el punto de partida del norte (125 de 166, 75 %, subiendo de 48 % a 91 % durante el entrenamiento)**; laboratorio y sur, 0 de 166. Ningún hito más allá del 6; medalla 1 en ningún caso. En la evaluación (10 episodios por caso) el modo determinista queda en 0 de 10 en los tres puntos de partida y el estocástico en 7 de 10 en el norte: la política "greedy" se queda pulsando una tecla contra una pared.
- **Convención de comparación acordada:** se comparan corridas por hitos alcanzados y distancia a la meta, no por recompensa de entrenamiento (cada variante cambia la recompensa); una sola variante por corrida; mismos pasos, instancias y savestates. Orden de episodios en `train.log`: el episodio i corre en el entorno (i−1) mod 3, o sea laboratorio, sur, norte.
- **Arquitectura de la red (explicada al usuario):** PPO de Stable-Baselines3 2.9.2 con `MlpPolicy`: actor y crítico separados, cada uno 12 → 64 → 64 (tanh); cabeza del actor 64 → 8 acciones y del crítico 64 → 1; 10 569 parámetros. Una sola red compartida por los tres entornos, sin memoria. `lr` 3e-4, `gamma` 0,99, `n_steps` 512, `ent_coef` 0,01, lote 128, 10 épocas. Observación: 12 valores (ID de mapa escalado como número, no one-hot; x/64, y/64; diálogo, batalla, equipo, HP y nivel del líder, medallas, dinero). Acciones: A, B, ARRIBA, ABAJO, IZQUIERDA, DERECHA, START, NADA (pulsa 4 frames y espera 16).
- **Consulta sobre más capas o otras activaciones:** se respondió que no valía la pena todavía: el límite parecía estar en lo que la red ve (sin terreno ni memoria) y no en su capacidad; con una sola semilla por corrida no se distinguiría una mejora real del ruido. Quedó como idea de baja prioridad (ver la lista al final de esta entrada).
- **Partida guardada dentro del juego:** el usuario vio que una instancia guardó la partida (la instancia 2, a las 17:22). No importa: cada episodio recarga el savestate y eso restaura todo el estado del juego; el `.sav` de la ROM no se tocó. Se decidió no relanzar las instancias solo por eso.

### Calibración de contadores y R1 (castigos)
- **Hecho:** el entorno lleva contadores por episodio (`info["contadores"]`, también en `train.log` y TensorBoard): choques (se pulsa hacia donde ya mira y no se mueve), giros (se pulsa otra dirección y solo gira; sin castigo), quieto (A, B, START o nada sin moverse), bucles (volver a una casilla pisada 2 veces o más en las últimas 30), `rev3`/`rev10`/`rev30`, menús, diálogos, batallas y casillas distintas. Una primera calibración que mezclaba giros con choques se descartó (`calibracion_v1_sin_giros`).
- **Calibración (30 000 pasos, castigos en 0, política casi aleatoria):** de cada 1000 pasos solo ~50-75 mueven al personaje; el resto son giros (35 %), quieto (25 %), choques (19 %) y menús (10 %); los bucles son raros (2 %). Valores de castigo elegidos con eso: choque 0,002, quieto 0,002, menú 0,002, bucle 0,005; giros 0.
- **R1 (`ppo_r1_castigos`, 500 000 pasos):** Oldale en el norte 109 de 166 (66 %), laboratorio y sur 0; **evaluación idéntica a `ppo_500k`** en los seis casos. Quieto y menús bajaron 80-90 %, pero los giros subieron (349 a 420-630 por episodio): el tiempo perdido se desplazó a girar.

### R3: premio por acercarse a la meta
- **Hecho:** `env/distance.py` construye con solo la ROM un grafo de casillas de todo el juego (pasos entre casillas con la misma regla de elevación del mundo, saltos de desnivel en un solo sentido, warps, conexiones entre mapas, sin agua ni NPC; ≈103 000 nodos, ≈1,7 s) y calcula por búsqueda inversa la distancia a pie hasta cualquier casilla de un mapa meta. `env/milestones.py` ganó `SHAPING_GOALS` (6 Oldale, 7 Ruta 103, 9 Petalburg, 12 Rustboro, 15 gimnasio; el 5 se omite). El entorno premia `coef × clip(d_anterior − d_nueva, −3, +3)` hacia la meta del primer hito sin cumplir; es una **función potencial**: ir y volver suma cero, así que dar vueltas no da recompensa. Nuevos contadores `dist_ini`/`dist_min`/`dist_fin` y salida de distancia en `--evaluate`. `scripts/check_distance.py` camina en vivo por el mejor camino del grafo: desde el norte llega a Oldale en 8 pasos (8 previstos), desde el sur en 35 (35 previstos); los cortes vistos fueron un combate salvaje y un NPC, no errores del grafo. Distancias iniciales: laboratorio 65, sur 35, norte 8.
- **Verificado:** pruebas con emulador simulado (`fake_shaping_test.py`: ir y volver suma cero, el salto se limita a 3 casillas, nada durante un warp); en `heatmap.py render` se corrigió un `KeyError` por un mapa `(−1,−1)` espurio.
- **Resultado `ppo_r3_acercamiento` (R1 + premio, coef 0,01):** Oldale norte 133 de 166 (80 %), **sur 2 de 166 (primeras llegadas)**, laboratorio 0. Distancia mínima media por tercios: laboratorio 54 → 40 → 29 (de 65), sur 22-23 (de 35), norte ≈0. Evaluación estocástica: **norte 10 de 10**, laboratorio y sur 0 (distancias mínimas 24-55 y 21-27); determinista: los 10 episodios de cada caso idénticos, sin llegar.
- **Descubierto:** la geometría de la Ruta 101 explica el estancamiento del sur: el camino correcto va por el bolsillo oeste y luego al este por y=10-12 hasta x≈15 y al norte, y los desniveles solo se saltan hacia abajo; el agente se acerca en línea recta a ~22 casillas, choca contra la fila de desniveles (y=13 o y=6) y no se aleja para rodearla, porque alejarse cuesta premio inmediato. En modo determinista se queda pulsando ARRIBA contra el desnivel de (10,14).
- **Pregunta del usuario (¿el modelo se quedó pegado aprendiendo a caminar hacia el salto?):** con la captura de pantalla se confirmó que el modelo estaba atascado contra el desnivel, no el emulador.
- **Mapas de calor:** se regeneran por corrida (`scripts/heatmap.py watch` muestrea las posiciones mientras se entrena, `render` las dibuja sobre los gráficos reales de la ROM, también por puerto). Un error de rotulado: se llamó Oldale al mapa (0,9), que es Pueblo Hojarasca; Oldale es (0,10).

### R3b: castigo de choques que escala
- **Motivo:** en R3 el agente se quedaba pulsando la misma tecla contra una pared. R3b = R3 con **un solo cambio**: el castigo por choque se multiplica por los choques seguidos en la misma casilla, hasta un tope de 10 (de 0,002 a 0,02); la racha se reinicia al moverse a otra casilla y los giros no la reinician. Nuevo contador `racha_max`. Prueba con emulador simulado (`fake_escalation_test.py`).
- **Resultado `ppo_r3b_choques_escalados` (498 episodios, ~2 h 13 min):** Oldale norte 130 de 166 (78 %), sur 1, laboratorio 0; igual que R3. Los choques bajaron 40-55 % respecto a R3 (laboratorio 71, sur 87, norte 59 por episodio) pero **los giros subieron** (631, 575, 367) y la racha máxima llega al tope. Evaluación: estocástica norte 10 de 10, laboratorio y sur 0 (distancias mínimas medias 26,7 y 21,4); determinista: 10 episodios idénticos, atascado (en el laboratorio contra la esquina de una casa de Pueblo Hojarasca, a 53 casillas).
- **Observación del usuario:** los tres modelos "aprendieron a rotar en su lugar después de caminar unos pasos" y se ve muy ineficiente. Es el contador `giros`, que hasta aquí no se castigaba; motivó R3c.
- **Atasco determinista:** se atribuyó a que la observación no incluye cuántos choques lleva (el castigo escalado no es markoviano para el agente). La evaluación se hace en las dos modalidades (determinista y estocástica) porque la determinista queda atascada.
- **Carga del equipo durante el entrenamiento (consulta del usuario):** CPU total 87 % (i7-10750H), RAM 8,1 de 11,8 GB, tres procesos de mGBA de ~160 MB, python ~370 MB, ~62-70 pasos/s; normal para 3 emuladores más PPO más el muestreador de mapas de calor.

## 2026-11-03

### R3c: castigo por girar, y el agente se esconde en los diálogos
- **Cambio:** R3b + `--turn-penalty 0,002` por cada giro en el lugar (nuevo parámetro `turn_penalty` en el entorno; no reinicia ni aumenta la racha de choques). Prueba simulada `fake_turn_test.py`.
- **Resultado `ppo_r3c_castigo_giros` (498 episodios, ~1 h 54 min):** los giros bajaron (laboratorio 631 → 35, sur 575 → 237, norte 367 → 152) pero Oldale **empeoró**: norte 116 de 166 (R3b 130), sur 0, laboratorio 0. Evaluación: laboratorio sin avanzar una casilla (distancia mínima 65 en los 10, con y sin muestreo), sur 0 de 10, norte 9 de 10 estocástico.
- **Descubierto (reward hacking):** **el agente aprendió a esconderse en los diálogos**, que no llevan castigo. En los últimos 55 episodios del laboratorio gasta ~989 de 1000 pasos en diálogo y pisa 1 casilla; en sur y norte subieron también los pasos en diálogo y "quieto" (240 y 125 por episodio). Con castigos mayores que el premio por moverse, lo más rentable es no actuar donde no cuesta. Es el mismo tipo de atajo que en R1 con los giros, más extremo.

### R3d: castigo por pasos en diálogo
- **Cambio:** R3b (sin castigo por giros) + `--dialog-penalty 0,002` por paso con diálogo o script activo (nuevo parámetro `dialog_penalty`; prueba simulada `fake_dialog_test.py`). Lanzada el 2026-11-03 con los mismos emuladores.
- **Estado al escribir esta entrada (344 000 de 500 000 pasos, 345 episodios):** no hay escondite en diálogos (0 a 2 pasos de diálogo por episodio en los últimos 25 episodios de cada partida). Oldale: norte 85 de 115, sur 1 de 115, laboratorio 0 de 115. Giros aún altos (391 a 636). Distancia mínima en los últimos 25: laboratorio 31, sur 22, norte 0. **Sin evaluar aún: los resultados finales de R3d se anotan en `docs/corridas.md` y en la entrada siguiente.**

### Balance de las variantes de castigo y búsqueda de otras mejoras
- **Tabla comparativa (entrenamiento, episodios con Oldale por punto de partida; una semilla por corrida):**

| Corrida | Cambio | Norte | Sur | Laboratorio |
|---|---|---|---|---|
| `ppo_500k` | referencia | 125/166 | 0 | 0 |
| `ppo_r1_castigos` | castigos (choque, quieto, menú, bucle) | 109/166 | 0 | 0 |
| `ppo_r3_acercamiento` | R1 + premio por acercarse | 133/166 | 2 | 0 |
| `ppo_r3b_choques_escalados` | R3 + choque escalado | 130/166 | 1 | 0 |
| `ppo_r3c_castigo_giros` | R3b + castigo por giros | 116/166 | 0 | 0 |
| `ppo_r3d_castigo_dialogos` | R3b + castigo por diálogos | en curso | | |

- **Lectura:** el premio por acercarse (R3) fue lo único que movió algo (primeras llegadas desde el sur y el laboratorio se acerca más); los ajustes de castigos siguientes dejaron las llegadas igual o peor. Cada vez que se cierra una salida barata (chocar, quedarse quieto, girar) el agente encuentra otra (girar, esconderse en el diálogo). Diferencias de pocos episodios (por ejemplo 133 contra 130) son ruido con una semilla.
- **Investigación (web):** el artículo de Pleines et al. sobre PPO en Pokémon Rojo y el trabajo PokeRL (de este último solo se leyó un resumen secundario; sus cifras no se verificaron) coinciden en que el shaping denso es necesario pero se explota, y en que los fallos típicos son bucles, spam de botones y exploración sin memoria; sus remedios son detectores de bucles y una máscara de casillas visitadas en la observación. Para exploración dispersa, RND (Burda et al.) y las políticas recurrentes (GRU/LSTM) mejoraron resultados. Fuentes: https://arxiv.org/pdf/2502.19920, https://arxiv.org/pdf/2604.10812, https://arxiv.org/pdf/1810.12894.pdf.
- **Ideas priorizadas (solo se ejecutaron la 1 y la 2, como preparación):** (1) terreno local en la observación; (2) memoria corta en la observación; (3) política recurrente (LSTM de sb3-contrib); (4) currículo hacia atrás con más savestates a lo largo de la Ruta 101; (5) bonus de curiosidad (RND o conteos); (6) varias semillas por corrida y más pasos; (7) redes más grandes. Advertencia anotada: dar al agente la dirección hacia la meta (`dx`, `dy` del camino mínimo) casi resuelve el problema, pero se parece al enfoque de reglas y desdibuja la comparación del proyecto.

### R4: observación con terreno y memoria (preparada, sin lanzar)
- **Hecho:** dos opciones nuevas, apagadas por defecto, para que los modelos anteriores se sigan evaluando igual (`--evaluate` lee `obs_terrain` y `obs_memory` del `config.json` junto al modelo).
  - `--obs-terrain`: parche de 5×5 casillas alrededor del jugador (norte arriba) con 6 clases en one-hot (libre, puerta, desnivel hacia abajo, arriba, izquierda y derecha; todo lo demás es 0), leído de la ROM con `WalkGraph.tile_kind` (nuevo, en `env/distance.py`, junto con `get_graph()`); no incluye NPC, árboles ni rocas. Observación de 162 valores.
  - `--obs-memory`: casillas del parche ya pisadas en el episodio (25), últimas 4 acciones en one-hot (32), orientación del personaje (4) y resultado del último paso: se movió, chocó, giró y racha de choques / 10 (4). Observación de 77 valores; con ambas, 227.
- **Verificado:** `terrain_test.py` contra la ROM: desde (10,14) en la Ruta 101 la fila de desniveles aparece justo encima del jugador (donde se atascaba el modelo) y desde (10,12) justo debajo; casos de Pueblo Hojarasca, laboratorio y esquina de mapa coherentes. `fake_obs_test.py` con emulador simulado: formas 12, 162, 77 y 227, memoria de choque, giro, movimiento, racha, límite de 4 acciones y limpieza en el reinicio. Los procesos de R3d ya habían importado el código, así que no se vieron afectados.
- **Plan (un cambio por vez):** R4a solo terreno (la siguiente en lanzarse, pedida por el usuario), R4b solo memoria, R4c ambas. Los castigos y el premio serán los de R3b. Se compara contra la mejor corrida con observación básica. Se lanza cuando R3d termine y se evalúe, porque comparten emuladores.
- **Límites conocidos:** el parche no ve NPC ni árboles; con una semilla por corrida las diferencias pequeñas no valen; el costo de velocidad es la primera lectura de cada casilla.

### Estado del repositorio al escribir esta entrada
- Último commit: `736feda`. **Todo lo de esta sesión está sin commitear** (el usuario no lo ha pedido): `env/distance.py`, `scripts/check_distance.py`, `env/pokemon_env.py`, `env/milestones.py`, `agents/rl/train.py`, `scripts/heatmap.py`, `docs/corridas.md`, `docs/bitacora.md`. `specs.md` aún no refleja R1 a R4.
- No versionado: `runs/rl/*`, `runs/heatmaps/*`, `runs/eval_*.sh/.txt`, `runs/rl_console_*.txt` y los scripts de prueba con emulador simulado (carpeta temporal de la sesión: `fake_shaping_test.py`, `fake_escalation_test.py`, `fake_counters_test.py`, `fake_turn_test.py`, `fake_dialog_test.py`, `fake_obs_test.py`, `terrain_test.py`).
- Reglas de trabajo vigentes: no commitear ni hacer push sin que se pida; no tocar `roms/*.sav`; no cerrar las ventanas de mGBA durante una corrida; una variante de recompensa por corrida.
