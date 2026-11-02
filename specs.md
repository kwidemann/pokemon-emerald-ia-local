# pokemon-ia — Especificaciones

Proyecto educativo: agentes de IA que juegan **Pokémon Esmeralda** (GBA) de forma local, comparando distintos enfoques de decisión sobre un mismo entorno.

> Este documento es la referencia viva del proyecto. Se actualiza cuando cambia una decisión de diseño. Fecha de la primera versión: 2026-11-01.

## 1. Objetivo y alcance

- **Horizonte:** completar el juego. Se avanza por hitos; el juego completo es meta de largo plazo, no de la primera fase.
- **Propósito:** educativo. Interesa comparar enfoques, no maximizar rendimiento.
- **Ejecución:** 100 % local, orquestada con scripts de Python que se gatillan desde Claude Code.
- **Fuera de alcance (por ahora):** multijugador, intercambios, trucos o manipulación de la ROM.

## 2. Hardware disponible (medido el 2026-11-01)

| Recurso | Valor | Implicancia |
|---|---|---|
| GPU | NVIDIA RTX 2060 Max-Q, 6 GB VRAM | Redes pequeñas (MLP/CNN). Un LLM 7B cuantizado cabe, pero no a la vez que un entrenamiento grande |
| CPU | Intel i7-10750H, 6 núcleos / 12 hilos | Varios emuladores en paralelo (empezar con 4) |
| RAM | 11,8 GB | Ajustada: cada instancia de mGBA y cada worker suma |
| Python del sistema | 3.13.7 | No se usa para el proyecto: ver sección 3 |
| Ollama | `gemma:7b`, `qwen2.5:14b` | Sin visión. El 14B no cabe en 6 GB de VRAM y es lento por offload a CPU |

## 3. Stack

- **Gestor de entorno:** `uv`, con Python **3.12** aislado (el 3.13 del sistema aún no lo soportan bien varias librerías de RL/ML).
- **Emulador:** **mGBA** con script **Lua** que expone un servidor por **socket TCP** (lectura de memoria, botones, savestates, captura de pantalla).
- **Entorno:** `gymnasium` (interfaz `reset/step`).
- **RL:** Stable-Baselines3 o CleanRL (PPO). PyTorch con CUDA se instala aparte, ver `pyproject.toml`.
- **LLM local:** Ollama (supervisor lento).
- **Decisor rápido:** Laya (PyTorch + `transformers`), pendiente de validar en esta máquina (ver 5.1.1).
- **ROM:** la aporta el usuario, legalmente obtenida. Va en `roms/` (ignorada por git). El proyecto no la descarga ni la distribuye.

## 4. Observación y acciones

- **Observación base:** estado estructurado leído de la RAM (no píxeles). Servirá a los cuatro enfoques.
- **Observación por píxeles:** no se usa por ahora. JEPA también trabajará sobre RAM (decisión del 2026-11-01). Podría agregarse después como variante aparte, sin mezclarla con la base.
- **Acciones:** `A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R` (más "ninguna"). Se aplican durante N frames (`frameskip`, por defecto 8–24).
- **Mapa de RAM:** definido en `env/ram_map.py`, con la fuente de cada dirección y si está verificada en vivo. Las direcciones salen de los símbolos de pokeemerald publicados en pokebot-gen3 y de los structs de pret/pokeemerald. La lectura está en `env/state.py` (`read_state`) y se inspecciona con `scripts/inspect_state.py`.

Campos mínimos del estado a leer: mapa actual y posición del jugador, dirección, estado de batalla, party (especie, nivel, HP, movimientos), medallas, dinero, objetos.

## 5. Enfoques a comparar

Todos implementan la misma interfaz `Agent.act(obs) -> action` para poder cambiar con `--agent`.

1. **Reglas / FSM / behavior tree** (`agents/rules/`). Lógica explícita de navegación y combate. Línea base.
2. **RL con PPO** (`agents/rl/`). Sobre la RAM estructurada.
3. **JEPA como modelo del mundo** (`agents/jepa/`). Predicción en espacio latente y planificación sobre él. Experimental. Ver sección 5.2.
4. **Híbrido con supervisor** (`agents/hybrid/`). Un decisor de alto nivel fija sub-objetivos; un ejecutor (reglas o RL) los cumple.

### 5.1 Sobre "Jev" (aclarado el 2026-11-01)

- **Jev** es un modelo de **TypeSafe AI** (lanzamiento en acceso anticipado el 15-09-2026). Recibe una pregunta tipada sobre un texto o JSON y devuelve un valor estructurado (opción, puntaje, probabilidad) con una estimación de confianza, sin generar explicaciones en lenguaje.
- **No es lo mismo que JEPA** (Yann LeCun / Meta), que es una familia de arquitecturas de aprendizaje de representaciones.
- **Disponibilidad:** es un modelo **alojado y propietario**; no hay pesos descargables. Se accede por API (también vía OpenRouter, `typesafe/jev-1.13`, según una búsqueda web del 2026-11-01 que no pude contrastar con la documentación oficial). Por lo tanto **no corre localmente**.
- **Existen reimplementaciones comunitarias** con pesos abiertos (por ejemplo "jevos", 1B, para CPU con llama.cpp, y "Vev", a partir de Qwen3.5 4B/9B). Son de terceros, sin verificar, y sus licencias varían.
- **Cómo lo usamos aquí (decisión del 2026-11-01):** el decisor rápido ("System 1") recibe el estado del juego y una pregunta cerrada (por ejemplo `¿próximo sub-objetivo? {ir_a_pueblo, curar, entrenar, ...}`) y responde con un valor de la lista más una confianza. Se implementa localmente con **Laya** (ver 5.1.1). Opcionalmente se compara contra Jev real vía API, solo si el usuario decide crear la cuenta y la clave.

#### 5.1.1 Laya como decisor local

- Repo: <https://github.com/NandhaKishorM/laya> (código Apache-2.0; la licencia de los pesos está **sin verificar**, revisar la ficha en Hugging Face, repo `convaiinnovations/laya`).
- Es un **encoder** (ModernBERT-large, 421M parámetros; variante multilingüe de 322M) que responde preguntas tipadas (`choice`, `score`, `noul`) en una sola pasada, sin generar texto. Devuelve probabilidades y confianza.
- Según el repo, ofrece un servidor `POST /v1/systemone` con el mismo protocolo que la API alojada de Jev (no verificado contra la documentación de TypeSafe).
- Corre con PyTorch + `transformers` (no con Ollama ni GGUF). Descarga ~2,3 GB de checkpoints la primera vez; admite uso offline desde un directorio local.
- **No es zero-shot** para decisiones tipadas: el checkpoint base queda casi al azar (0,362 en el benchmark del autor) y el afinado `laya-typed-decisions` llega a 0,766. Cifras medidas por el autor, no independientes. Para Pokémon hay que **afinarlo con datos del juego**.
- Etiquetas de entrenamiento posibles: decisiones del agente de reglas, partidas exitosas del PPO, o un LLM (`qwen2.5`) como profesor.
- Su confianza sale sobreconfiada de fábrica, según el repo: calibrar antes de usarla como umbral.
- Antes de comprometer el diseño: prueba corta de instalación y de latencia en esta máquina (RTX 2060 / CPU), confirmar la versión de Python compatible (su ejemplo usa 3.11) y la licencia de los pesos.

### 5.2 JEPA sobre RAM (decisión del 2026-11-01)

Adaptación a pequeña escala de la idea JEPA a vectores de estado, no una réplica de I-JEPA/V-JEPA (que trabajan sobre imágenes y video).

- **Encoder** `f(s_t) -> z_t`: MLP pequeño sobre el estado de RAM (posición, mapa, party, etc.).
- **Predictor** `g(z_t, a_t) -> ẑ_{t+1}`: predice el siguiente estado **en el espacio latente**, no en bruto.
- **Objetivo:** el encoder objetivo (copia con media móvil exponencial) codifica `s_{t+1}`; la pérdida se calcula en el espacio latente. Hace falta un mecanismo contra el colapso (stop-gradient/EMA y/o regularización tipo VICReg).
- **Planificación:** búsqueda de secuencias de acciones sobre el modelo (CEM/MPC) que acerquen `z` a un objetivo latente o maximicen una cabeza de recompensa.
- **Datos:** transiciones `(s_t, a_t, s_{t+1})` recogidas por el agente aleatorio y el de reglas. Con RAM el entrenamiento cabe holgado en 6 GB de VRAM e incluso en CPU.
- **Riesgo conocido:** gran parte del estado de RAM es discreto (IDs de mapa, especies). Se codificará con embeddings; hay que validar que el espacio latente no ignore lo relevante.

### 5.3 Enfoque híbrido de dos niveles (decisión del 2026-11-01)

Tres capas con velocidades distintas:

1. **Ejecutor** (cada frame/paso): reglas o PPO. Pulsa botones.
2. **Decisor rápido, System 1** (cada pocos pasos): **Laya** elige el sub-objetivo desde una lista cerrada, con confianza.
3. **Supervisor lento, System 2** (cada N decisiones o ante baja confianza): LLM de Ollama que planifica y explica. También sirve de **profesor** para generar etiquetas con las que afinar a Laya.

Detalle del supervisor lento:

- Planificador textual con `qwen2.5:7b` o `gemma:7b` (decisión del usuario, 2026-11-01).
- Recibe el estado como texto/JSON y devuelve el siguiente sub-objetivo, **no** botones.
- Debe responder en formato estructurado y validado; si la respuesta es inválida o de baja confianza, el ejecutor sigue con el último sub-objetivo.
- Se corre con un solo modelo cargado a la vez para no competir por la VRAM con el entrenamiento.

## 6. Estructura del repositorio

```
pokemon-ia/
├─ specs.md
├─ docs/
│  └─ bitacora.md   # registro cronológico de pasos, decisiones y hallazgos
├─ pyproject.toml
├─ .gitignore
├─ roms/            # ROM del usuario (ignorada por git)
├─ emulator/        # script Lua de mGBA y cliente del socket
├─ env/             # pokemon_env.py (PokemonEmeraldEnv), ram_map.py, state.py, milestones.py
├─ agents/
│  ├─ rules/
│  ├─ rl/
│  ├─ jepa/
│  └─ hybrid/
├─ scripts/         # run.py, train.py, eval.py
└─ runs/            # logs, checkpoints, videos (ignorada por git)
```

## 7. Fases

1. **Fundaciones:** `uv sync`, instalar mGBA, puente Python↔Lua, leer una posición de RAM y apretar un botón.
2. **Entorno Gym:** mapa de RAM verificado, savestates, recompensas configurables.
3. **Línea base:** agente aleatorio y agente de reglas.
4. **RL:** PPO con varios emuladores en paralelo.
5. **Híbrido:** decisor rápido (Laya, afinado con datos del juego) y supervisor lento (Ollama) sobre un ejecutor. Incluye una prueba previa de instalación y latencia de Laya.
6. **JEPA:** modelo del mundo latente y planificación.
7. **Evaluación:** métricas comparables entre enfoques y registro de resultados en este documento.

### 7.1 Estado de la Fase 1 (2026-11-01)

- **Hecho:** `uv sync` con Python 3.12 (gymnasium, numpy, pillow, ollama); mGBA 0.10.5 instalado con winget en `C:\Program Files\mGBA\` (el binario incluye scripting Lua con sockets); `emulator/server.lua` (servidor TCP, puerto 8888), `emulator/client.py` (`MGBAClient`) y `scripts/ping.py`.
- **Verificado en mGBA con la ROM real:** `ping`, `frame`, `step` (pulsa teclas N frames), `readrange`, `savestate`, `loadstate` y `screenshot` (la captura mostró el menú principal NEW GAME / OPTION).
- **ROM:** `roms/Pokemon - Emerald Version (USA, Europe).zip` (6,55 MB). mGBA lee el `.zip` directamente.
- **Carga del script:** `mGBA.exe` 0.10.5 **no acepta `--script`** (se cierra). Se carga a mano en cada sesión: *Tools > Scripting > File > Load script*. Para reiniciar con una versión nueva del Lua hay que cerrar mGBA y volver a cargarlo.
- **Rutas con espacios:** se cortaban en el primer espacio (`Repositorios Git`) y mGBA escribió un archivo suelto en `Documents`. Corregido en `server.lua` y verificado tras recargarlo (7.3): `savestate`/`loadstate` aceptan rutas con espacios.
- **`step` avanza un frame de más** (pedir 30 avanza 31). Es constante, así que no afecta el determinismo, pero hay que tenerlo en cuenta al contar frames.
- **Velocidad:** a tiempo real (60 fps) `step` de 8 frames tardaba ~150 ms por llamada. Con el avance rápido sin límite activado, ~1400 fps (medición en 7.3).
- **Protocolo:** una línea por comando (`ping`, `frame`, `read8|16|32`, `readrange`, `step <máscara> <frames>`, `savestate`, `loadstate`, `screenshot`); respuesta `OK <datos>` o `ERR <motivo>`. Detalle en el encabezado de `emulator/server.lua`.

### 7.2 Estado de la Fase 2: mapa de RAM (2026-11-01)

- **ROM confirmada:** SHA-1 `f3ae088181bf583e55daf962a92bb46f4f1d07b7`, idéntico al de pret/pokeemerald (v1.0 USA/Europa, código `BPEE`, versión 0).
- **Los save blocks se reubican** en cada arranque (se vieron desplazamientos `+0x48` y `+0x58` respecto de la dirección estática). Hay que leer siempre `gSaveBlock1Ptr` (`0x03005D8C`) y `gSaveBlock2Ptr` (`0x03005D90`) y sumar offsets.
- **Verificado en vivo:** posición `(x, y)` y mapa `(grupo, número)` en `SaveBlock1+0x00/0x04` (derecha suma `x`, abajo suma `y`, el interior del camión es el mapa `(25, 40)`, Littleroot es `(0, 9)`); dinero en `SaveBlock1+0x490` con XOR de `SaveBlock2+0xAC` (dio 3000, el valor inicial del juego); nombre del jugador en `SaveBlock2+0x00` (`AAAA`); estado del script en `0x03000E38` (0 con un cuadro de texto abierto, 2 sin script activo); `gPlayerPartyCount` (0 sin Pokémon).
- **Verificado en vivo con el starter (Mudkip, elegido por el agente):** `parse_party_mon` completo. Dio `species=283` (Mudkip: el índice interno de Gen III, no el número de la Pokédex), nivel 5, HP 21/21, movimientos `(33, 45)` = Placaje y Gruñido, PP `(35, 40)` y `checksum_ok=True`. Tras ganar la primera batalla con 3 Placajes, el PP bajó a 32, el HP a 18 y la experiencia pasó de 135 a 152, así que los campos cambian como se espera.
- **Verificado en vivo en la primera batalla:** `in_battle` (byte `gMain+0x439`, máscara `0x02`) pasó a `True` al empezar y a `False` al terminar. `gBattleTypeFlags` valió `0x14` = `IS_MASTER (0x04) | FIRST_BATTLE (0x10)`, que coincide con los bits de pokeemerald (`DOUBLE=0x01`, `LINK=0x02`, `IS_MASTER=0x04`, `TRAINER=0x08`, `FIRST_BATTLE=0x10`). Esos valores de los bits vienen de memoria de la fuente y no los revisé línea por línea; lo comprobado es solo `0x14` en esa batalla.
- **Sin verificar en vivo:** las medallas (flags `0x867..0x86E`, hoy leen 0, que es lo correcto, pero no hay un caso con medallas) y `gBattleTypeFlags` con un entrenador (bit `0x08`).
- **Guion de la intro, automatizado con `scripts/drive.py`** (mapas como `(grupo, número)`): camión → Littleroot `(0,9)` → casa del jugador `(1,0)`, dormitorio `(1,1)` y reloj de pared (hora por defecto, el cursor arranca en NO) → casa del vecino `(1,2)` y su cuarto `(1,3)`, donde aparece el rival → Ruta 101 `(0,16)`, grito de Birch, bolso con los tres Pokémon → batalla contra un Zigzagoon nivel 2 → laboratorio de Birch `(1,4)`. Mientras no se visita al vecino, un NPC bloquea la salida norte de Littleroot con "Um, um, um!".
- **Savestate inicial:** `C:\AI\pokemon-ia-states\lab_mudkip.ss0`, en el laboratorio `(1,4)` en `(6,5)` con Mudkip nivel 5 y el reloj ya puesto. Está fuera del proyecto porque la copia de `server.lua` cargada en mGBA corta las rutas en el primer espacio. Al recargar el Lua hay que moverlo a `runs/`. `loadstate` se probó: tras caminar 2 casillas, cargar devolvió la posición `(6,5)` original.
- **No implementado todavía:** orientación del jugador, Pokémon enemigo en batalla, bolsa/objetos, banderas de progreso (`vars`), y el valor `1` (esperando) del estado de script.
- **Movimiento medido:** una pulsación corta (1-6 frames) mueve exactamente una casilla, y la posición en RAM se actualiza unos frames después de soltar la tecla. Si el personaje no miraba hacia esa dirección, la primera pulsación solo lo gira. Mantener la tecla mientras se lee la posición cada frame hace avanzar varias casillas por error; `Driver.move()` usa pulsaciones de 4 frames con 20 de pausa.
- **Velocidad medida:** 600 frames en 10 s, o sea 60 fps (tiempo real). Sigue pendiente activar el avance rápido sin límite en mGBA.
- **Lecciones del control por botones:** la `A` se usa para avanzar texto, pero si no hay cuadro abierto interactúa con lo que tiene delante (en la camioneta abrió mensajes de las cajas). Para decidir qué pulsar hay que mirar `script_active` en vez de contar pulsaciones.
- **Rendimiento:** `read_state` hacía ~7 llamadas por el socket; con `readmulti` (ver 7.3) hace 2. Hasta que se recargue el Lua, el cliente cae solo a una lectura por rango.

### 7.3 Estado de la Fase 3: entorno Gymnasium (2026-11-01)

- **Hecho:** `env/pokemon_env.py` (`PokemonEmeraldEnv`), `env/milestones.py` (catálogo del Tramo 1 y `Tracker` del historial), `scripts/random_agent.py` y el comando `readmulti` en `server.lua` con `MGBAClient.read_multi`. `read_state` pasó de ~7 llamadas por el socket a 2.
- **Acciones:** `Discrete(8)` = `A, B, UP, DOWN, LEFT, RIGHT, START, NOOP`. Una acción mantiene la tecla `hold_frames=4` y espera `wait_frames=16`. Verificado en vivo: si el personaje ya mira en esa dirección avanza exactamente **una casilla**; si no, la primera acción solo lo gira (el agente tiene que aprender eso).
- **Observación:** vector `Box(12,)` en [0, 1]: grupo y número de mapa (escalados), `x`, `y`, script activo, en batalla, batalla de entrenador, tamaño del equipo, fracción de HP, nivel del líder, medallas y `log(dinero)`. Es **provisional**: los IDs discretos de mapa escalados dan una falsa noción de cercanía (ver 5.2). El estado completo viaja en `info["state"]`.
- **Recompensa:** solo por hitos (1,0 cada uno; 5,0 la medalla), "enganchados" por episodio. Lo que ya se cumple en el savestate al hacer `reset()` se marca sin recompensa. Ayuda opcional `new_tile_bonus` (0 por defecto). La pregunta Q5 (una recompensa común o una por enfoque) sigue abierta; estos valores son provisionales.
- **Episodio:** termina (`terminated`) al conseguir el hito `terminal_milestone` (16, medalla 1) y se trunca a `max_steps` (2000 por defecto). El `reset()` carga el savestate y deja 2 frames para que la RAM se asiente.
- **Hitos con detector:** 3 (starter) y 5 (laboratorio), verificados. Los hitos 6, 7, 9, 12, 13, 14 y 16 tienen detector sin verificar (los IDs de mapa de Oldale, Ruta 103, Petalburg y Rustboro están **inferidos** del orden de pokeemerald; Littleroot `(0,9)` y Ruta 101 `(0,16)` sí coinciden con ese orden). Los hitos 8, 10, 11 y 15 todavía no tienen detector.
- **Verificado en vivo:** `reset()` carga el savestate y marca los hitos 3 y 5 sin recompensa; el agente aleatorio corrió 60 pasos sin errores; los movimientos del entorno cambian la posición como se espera.
- **Sin verificar:** el bit de batalla de entrenador (`0x08`) y, por tanto, el contador de batallas ganadas; las medallas.
- **Velocidad:** con el Lua viejo y a tiempo real el agente aleatorio daba **1,7 pasos/s**. Tras recargar el Lua y activar el avance rápido: `step 0 600` tarda 0,42 s (~1400 fps) y el agente aleatorio da **~52 pasos/s** (1500 pasos en ~28 s, 20 frames por acción, 2 llamadas de lectura por paso). Ya es viable para RL (~190 mil pasos por hora con un solo emulador); el siguiente cuello de botella es el socket, no el emulador.
- **Verificado tras recargar el Lua:** `readmulti` funciona; `savestate` y `loadstate` aceptan rutas con espacios; el savestate inicial vive ahora en `runs/lab_mudkip.ss0` (ignorado por git; se regenera con `scripts/make_savestate.py`, ver 7.4). El entorno resuelve la ruta del savestate a absoluta porque mGBA interpreta las relativas desde su propio directorio.
- **Transición de mapa:** al cruzar una puerta, un paso de 6000 aleatorios leyó la ubicación como mapa `(0,0)` en `(0,0)`, lo que disparó el hito de Petalburg (`(0,0)`) por error. `GameState.in_transition` detecta ese caso y el entorno lo ignora para hitos e historial. Es un filtro empírico, sin bandera de transición verificada.
- **Pendiente de la Fase 3:** verificar los hitos con detector sin probar; medir con varios emuladores en paralelo.

### 7.4 Generador del savestate inicial (2026-11-01)

- **Hecho:** `scripts/make_savestate.py` reinicia la consola (comando `reset` nuevo en `server.lua`) y juega la intro completa desde el logo hasta el laboratorio de Birch: NUEVO JUEGO, chico, nombre "AAAA", camión, mamá, reloj, vecino y rival, rescate de Birch con Mudkip, primera batalla. Guarda `runs/lab_mudkip.ss0`.
- **Verificado:** dos ejecuciones seguidas desde cero terminaron bien, en unos 16 s cada una, en el laboratorio `(1,4)` en `(6,5)` con Mudkip nivel 5 (283) y el jugador "AAAA". El entorno (`random_agent.py`, 300 pasos) funciona con el savestate generado. El `.sav` de `roms/` quedó idéntico (mismo MD5 que antes de ejecutar).
- **Cómo comprueba cada tramo:** tras cada fase compara el mapa y el estado contra la RAM y se detiene con un error y el estado si algo no coincide. Los diálogos se avanzan con `Driver.settle()` (A hasta que `script_active` sea falso y siga así 90 frames, porque algunos scripts se reanudan).
- **Pantallas previas al juego:** antes de la partida no hay estado válido en RAM (SaveBlock1 apunta a datos viejos), así que se usan `gMain.callback2` y el color de píxeles de una captura. Valores de `callback2` observados: `0x080E4F59` pantalla de nombre, `0x08085E5D` mundo abierto, `0x08134C9D` pantalla del reloj; `0x0802F6B1` lo comparten el menú principal y el discurso de Birch, por eso el menú se detecta por los colores de los bordes inferiores `(140,148,255)`.
- **Sin verificar:** el caso de **no haber `.sav`** (menú con NUEVO JUEGO en primer lugar): el script lo prevé (solo pulsa ABAJO si ve CONTINUE) pero no se probó. Tampoco se probó con otro ritmo de emulación.
- **Límite de reproducibilidad:** entre comandos el emulador sigue corriendo a velocidad libre, así que el RNG cambia y cada ejecución da un Mudkip distinto (por ejemplo 20 HP en vez de 21) y una batalla distinta. El **flujo** es reproducible, los **valores** no. Esto también afecta a los pasos del entorno: unos frames de más pasan entre comandos. Para determinismo estricto habría que pausar la emulación entre comandos (no hay una llamada verificada para eso).

### 7.5 Estado de la Fase 4: agente de reglas (2026-11-01)

Primer enfoque (reglas explícitas, sin aprendizaje). Código en `agents/rules/agent.py`, lanzador `scripts/run_rules.py` (con `--tramo1` para el guion completo), y tres módulos de apoyo reutilizables por los demás enfoques: `env/mapdata.py` (mapa y colisiones desde la RAM), `env/world.py` (grafo de mapas desde la ROM) y `env/battle_data.py` (movimientos y tipos desde la ROM).

**Resultado:** desde el savestate del laboratorio, una ejecución continua (`uv run python scripts/run_rules.py --tramo1`) completó los **13 hitos del Tramo 1** (3, 5 a 16): llegó a la medalla 1 de Roxanne en 2 min 40 s de reloj y 191 818 frames de juego (53 min de juego), con 37 combates y 1213 movimientos. Los frames por hito de esa ejecución fueron: Oldale 3258, rival 40 849, Pokédex 45 923, Petalburg 72 242, Wally 77 866, bosque 104 018, Rustboro 110 589, segundo Pokémon 117 709, gimnasio 163 164, medalla 191 100 (el hito 14 aparece al evolucionar en el combate contra Roxanne). Es **una** ejecución: no hay todavía tasa de éxito ni variación entre semillas, y el azar del juego cambia entre ejecuciones (en otras ejecuciones el agente perdió contra el rival o necesitó más entrenamiento).

- **Cómo decide (prioridad en cada iteración):** batalla abierta; pantalla que no es el mundo (justo tras un combate se acepta con A para no cancelar la evolución, después se sale con B); diálogo (B por defecto); curarse si algún Pokémon baja del 50 % de HP; y por último navegar hacia el mapa objetivo.
- **Navegación:** el grafo de la ROM da el camino entre mapas (warps y conexiones) y `find_path` (BFS sobre las colisiones leídas de la RAM) lleva al jugador hasta la salida. El grafo distingue **zonas conexas** de cada mapa (calculadas con la cuadrícula de la ROM): la Ruta 104 se recorre como sur, Bosque Petalburgo y norte. Las puertas de edificio tienen colisión: se llega a la casilla de al lado y se empuja. Las alfombras de salida se pisan y se pulsa la dirección que activa el warp.
- **Combate:** elige el movimiento con más daño esperado (poder x STAB x efectividad x precisión) con los datos de la ROM. Se guía por los cursores de la RAM (`gActionSelectionCursor`, `gMoveSelectionCursor`): prueba con DERECHA cuál de los cursores se mueve, vuelve a FIGHT comprobándolo, y solo entonces acepta. En combates dobles mira los dos combatientes del jugador (0 y 2). Contesta SÍ a las preguntas (aprender un movimiento nuevo).
- **Capturas:** abre la mochila (BAG), va al bolsillo de Poké Balls (`gBagPosition+5`) y usa el primer objeto; una vez capturado contesta NO al apodo con B. Bolsa verificada en vivo: bolsillo de Poké Balls en `SaveBlock1+0x650` (objeto 4 = Poké Ball, cantidad cifrada con la clave de `SaveBlock2+0xAC`).
- **Otras rutinas:** `talk_to` (camina hasta un NPC por su sprite y le habla, también a través de un mostrador; puede curarse antes), `heal` (Centro Pokémon más cercano, enfermera con el sprite 58), `grind` (entrena en la hierba alta, detectada con el comportamiento de casilla del tileset), `catch`.
- **Datos verificados en vivo:** flags `0x861` (Pokédex), `0x867` (medalla 1); batalla de entrenador `gBattleTypeFlags = 0xC`, doble `0xD`, tutorial de Wally `0x204`; grupos y números de mapa de todas las paradas; el grafo de la ROM (518 mapas, 18 con enfermera); datos de movimientos y tipos de la ROM contra el juego.

**El cuelgue del juego (resuelto):** el juego se colgó dos veces en combates de entrenador. La causa **no** era del emulador ni del avance rápido: el agente elegía HUIR (o abría la mochila) cientos de veces en un mismo combate, y el juego graba cada acción en un búfer (`sBattleRecords`, `0x0203BD34`; tamaños en `0x0203C794`, previos en `0x0203C79C`). Al terminar, `RecordedBattle_BufferNewBattlerData` copia lo acumulado con un contador `u8`: si la diferencia es de 256 bytes o más, el bucle no termina nunca (en el estado colgado el combatiente 0 tenía 664 bytes grabados, todos `03` = HUIR, contra 3 previos). Se encontró leyendo el `pc` de la CPU (comando `reg`), que estaba dentro de esa función. Medidas: el agente vigila ese búfer cada 10 turnos y se detiene con un error si pasa de 200 bytes; no usa HUIR; no acepta un menú si el cursor no volvió a FIGHT; cierra la mochila con B si no está capturando.

- **Problemas conocidos:**
  1. Una sola ejecución completa; falta medir la tasa de éxito con varias semillas. El agente depende del azar: puede perder contra el rival de la Ruta 103 si va con poco nivel.
  2. El entrenamiento y la curación son básicos (umbral fijo de HP; los combates salvajes con poca vida siguen peleándose).
  3. El combate no se probó contra ataques de estado, objetos ni cambios de Pokémon; con un solo Pokémon vivo si cae se pierde el combate.
  4. Los datos de mapas, sprites (58 enfermera, 240 rival, 129 Norman, 125 Roxanne) y la lista de pasos del Tramo 1 están escritos a mano para este recorrido.

## 8. Métricas de comparación

Pasos (frames) hasta cada hito, tasa de éxito sobre N semillas/savestates, tiempo de reloj, uso de VRAM/RAM, y tasa de respuestas inválidas del supervisor.

## 9. Preguntas abiertas

- ~~**Q1.** JEPA sobre píxeles o RAM~~ → resuelta el 2026-11-01: **sobre RAM**.
- ~~**Q2.** Jev real por API~~ → resuelta el 2026-11-01: **solo Laya** (emulación local); no se usa Jev real.
- ~~**Q3.** Hitos intermedios~~ → resuelta el 2026-11-01 para el Tramo 1 (16 hitos hasta la medalla 1, ver 9.1). El Tramo 2 queda sin detallar.
- ~~**Q4.** `git init`~~ → resuelta el 2026-11-01: repositorio local en `main`, sin commits hasta que el usuario los pida. Sigue abierto dónde alojarlo (hoy sin remoto).
- **Q5.** Recompensas: ¿mantener una sola función común o una por enfoque?
- **Q6.** Laya: licencia de los pesos, versión de Python compatible, latencia real en esta máquina y fuente de las etiquetas para afinarlo.

### 9.1 Hitos (Tramo 1 aprobado el 2026-11-01)

Cada hito debe ser **detectable desde la RAM** (mapa, bandera, cantidad de medallas, equipo) para poder medirlo igual en los cuatro enfoques. Los identificadores de mapa y de bandera concretos se anotan al visitarlos y verificarlos.

**Fuente y fiabilidad:** solo el orden de las rutas (Littleroot, Ruta 101, Oldale, Ruta 103, Ruta 102, Petalburg, Ruta 104, Bosque Petalburgo, Rustboro) viene de una fuente ([Bulbapedia](https://bulbapedia.bulbagarden.net/wiki/Walkthrough:Pok%C3%A9mon_Emerald)); la página no se pudo descargar completa (error 403). Los detalles marcados con **[memoria]** salen de la memoria del asistente y hay que corregirlos al jugarlos, leyendo la RAM.

**Tramo 1, primer objetivo realista para los agentes (hasta la primera medalla):**

| # | Hito | Cómo se detecta | Estado |
|---|---|---|---|
| 1 | Reloj puesto y salida de casa | Mapa distinto del interior de la casa y script del reloj terminado | Hecho en la intro |
| 2 | Visitar al vecino y ver al rival en su cuarto | Mapa `(1,3)` visitado | Hecho |
| 3 | Starter conseguido | `party_count == 1` | Verificado |
| 4 | Primera batalla ganada (Ruta 101) | `in_battle` vuelve a `False` con HP > 0 | Verificado |
| 5 | Entrar al laboratorio de Birch | Mapa `(1,4)` | Hecho |
| 6 | Llegar a Oldale Town | Mapa `(0,10)` | Verificado (agente de reglas) |
| 7 | Vencer al rival en la Ruta 103 | Mapa `(0,18)` y combate de entrenador (`gBattleTypeFlags = 0xC`) ganado | Verificado |
| 8 | Volver al laboratorio y recibir la Pokédex | Bandera `0x861` | Verificado (la bandera se activó al hablar con Birch) |
| 9 | Llegar a Petalburg City por la Ruta 102 | Mapa `(0,0)` | Verificado |
| 10 | Escena de captura de Wally (hablar con Norman en el gimnasio de Petalburg) | `gBattleTypeFlags & 0x200` durante el combate (valió `0x204`) | Verificado |
| 11 | Cruzar el Bosque Petalburgo y vencer a un entrenador | Mapa `(24,11)` y combate de entrenador ganado | Verificado (no se distingue si fue el recluta de Aqua) |
| 12 | Llegar a Rustboro City por la Ruta 104 | Mapa `(0,3)` | Verificado |
| 13 | Capturar un segundo Pokémon | `party_count >= 2` | Verificado |
| 14 | Starter a nivel 16: Mudkip evoluciona a Marshtomp | La especie interna pasa de 283 a 284 | Verificado |
| 15 | Entrar al gimnasio de Rustboro | Mapa `(11,3)` | Verificado |
| 16 | Vencer a Roxanne | Bandera `0x867` y 1 medalla | Verificado |

Roxanne usa, de memoria, Geodude nivel 14, Geodude nivel 14 y Nosepass nivel 15 (tipo Roca); no está comprobado.

**Tramo 2, el juego completo (sin detallar):**
- Medallas 2 a 8, en el orden de Esmeralda: Brawly, Wattson, Flannery, Norman, Winona, Tate y Liza, Wallace (banderas `0x868`–`0x86E`).
- Liga Pokémon: los cuatro miembros de la Elite y el campeón.

**Sugerencias de diseño:**
- **Un solo catálogo de hitos para todos los enfoques**, con la recompensa de cada hito definida aparte. Así la comparación es justa; lo que cambia por enfoque es la ayuda de recompensa (por ejemplo bonificación por casillas nuevas o por subir de nivel, que solo tiene sentido en RL y en JEPA).
- **Empezar el RL y el híbrido solo con el Tramo 1.** Es el primer objetivo que podemos alcanzar y medir sin pasar semanas entrenando. El Tramo 2 queda para cuando haya un agente que complete el 1.
- **Subdividir los hitos largos en puntos de control con savestate**, por ejemplo uno por ciudad, de modo que un agente fallido se pueda reiniciar sin repetir todo el trayecto.
- **Medir cada hito con tres números:** frames hasta alcanzarlo, tasa de éxito sobre N semillas y número de derrotas (equipo sin vida).

## 10. Convenciones

- Cada paso que cambie el proyecto se registra en `docs/bitacora.md` (hecho, verificado, decidido, pendiente). Las especificaciones y el diseño quedan en este documento.
- No se hacen commits automáticos; el usuario decide cuándo.
- Toda dirección de RAM se documenta con su fuente y cómo se verificó.
- Los scripts de `scripts/` son el único punto de entrada; deben ser invocables desde Claude Code con argumentos explícitos.
