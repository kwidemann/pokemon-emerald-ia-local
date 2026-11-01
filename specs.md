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
├─ env/             # PokemonEmeraldEnv, ram_map.py, rewards.py
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
- **Corrección pendiente de cargar:** las rutas con espacios se cortaban en el primer espacio (`Repositorios Git`), y mGBA escribió un archivo suelto en `Documents`. Ya se corrigió en `server.lua` (la ruta es el resto de la línea), pero la copia cargada en mGBA ahora es la anterior. **Recargar `server.lua`** y volver a probar `savestate` con una ruta con espacios.
- **`step` avanza un frame de más** (pedir 30 avanza 31). Es constante, así que no afecta el determinismo, pero hay que tenerlo en cuenta al contar frames.
- **Velocidad:** el emulador va a velocidad real (60 fps): `step` de 8 frames tarda ~150 ms por llamada. Para entrenar hay que activar el avance rápido sin límite (menú *Emulation* de mGBA) y medir de nuevo.
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
- **Rendimiento:** `read_state` hace ~7 llamadas por el socket. Más adelante conviene un comando `readmulti` en el Lua para leerlo todo en una sola.

## 8. Métricas de comparación

Pasos (frames) hasta cada hito, tasa de éxito sobre N semillas/savestates, tiempo de reloj, uso de VRAM/RAM, y tasa de respuestas inválidas del supervisor.

## 9. Preguntas abiertas

- ~~**Q1.** JEPA sobre píxeles o RAM~~ → resuelta el 2026-11-01: **sobre RAM**.
- ~~**Q2.** Jev real por API~~ → resuelta el 2026-11-01: **solo Laya** (emulación local); no se usa Jev real.
- **Q3.** Hitos intermedios concretos del juego y su orden. Hay una propuesta en 9.1, pendiente de que el usuario la apruebe o la cambie.
- ~~**Q4.** `git init`~~ → resuelta el 2026-11-01: repositorio local en `main`, sin commits hasta que el usuario los pida. Sigue abierto dónde alojarlo (hoy sin remoto).
- **Q5.** Recompensas: ¿mantener una sola función común o una por enfoque?
- **Q6.** Laya: licencia de los pesos, versión de Python compatible, latencia real en esta máquina y fuente de las etiquetas para afinarlo.

### 9.1 Hitos propuestos (borrador, sin aprobar)

Cada hito debe ser **detectable desde la RAM** (mapa, bandera, cantidad de medallas, equipo) para poder medirlo igual en los cuatro enfoques. Los identificadores de mapa y de bandera concretos se anotan al visitarlos y verificarlos; los de abajo son descripciones, no valores comprobados.

**Tramo 0, ya cubierto por el guion de la intro (sirve como savestate y prueba del detector):**
1. Salir de la casa y poner el reloj.
2. Conseguir el starter (`party_count >= 1`; verificado).
3. Ganar la primera batalla (`in_battle` vuelve a `False` con el equipo vivo; verificado).

**Tramo 1, primer objetivo realista para los agentes (hasta la primera medalla):**
4. Recibir la Pokédex en el laboratorio y salir de Littleroot.
5. Llegar a Oldale Town.
6. Llegar a Petalburg City.
7. Llegar a Rustboro City.
8. Ganar un combate contra un entrenador (`gBattleTypeFlags` con el bit de entrenador, aún sin ver).
9. Subir el starter a nivel 10, y luego a nivel 15.
10. Capturar un Pokémon (`party_count >= 2`).
11. Vencer a Roxanne y conseguir la **medalla 1** (bandera `0x867`).

**Tramo 2, el juego completo:**
12. Medallas 2 a 8, en el orden de Esmeralda: Brawly, Wattson, Flannery, Norman, Winona, Tate y Liza, Wallace (banderas `0x868`–`0x86E`).
13. Liga Pokémon: los cuatro miembros de la Elite y el campeón.

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
