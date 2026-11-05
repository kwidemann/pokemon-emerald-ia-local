"""Direcciones de RAM y offsets de estructuras de Pokémon Esmeralda (BPEE, USA/Europa, v1.0).

ROM verificada: SHA-1 f3ae088181bf583e55daf962a92bb46f4f1d07b7 (la misma que construye pret/pokeemerald).

Fuentes:
  [SYM] https://raw.githubusercontent.com/40CAKES/pokebot-gen3/main/modules/data/symbols/pokeemerald.sym
  [G]   https://github.com/pret/pokeemerald/blob/master/include/global.h
  [P]   https://github.com/pret/pokeemerald/blob/master/include/pokemon.h
  [M]   https://github.com/pret/pokeemerald/blob/master/include/main.h
  [F]   https://github.com/pret/pokeemerald/blob/master/include/constants/flags.h
  [EV]  https://github.com/pret/pokeemerald/blob/master/src/event_data.c
  [MON] https://github.com/pret/pokeemerald/blob/master/src/money.c

Estado de verificación (contra el juego en ejecución, 2026-11-01):
  OK  = comprobado en vivo
  --  = tomado de la fuente, aún sin comprobar en vivo
"""

# --- Punteros y variables globales -------------------------------------------------------

# Los save blocks se REUBICAN en cada arranque/carga (se observó un desplazamiento distinto cada vez),
# así que siempre hay que leerlos a través de estos punteros, nunca por dirección estática.
G_SAVE_BLOCK1_PTR = 0x03005D8C  # OK [SYM]: apunta a SaveBlock1 (IWRAM, u32)
G_SAVE_BLOCK2_PTR = 0x03005D90  # OK [SYM]: apunta a SaveBlock2 (IWRAM, u32)

G_PLAYER_PARTY = 0x020244EC  # OK [SYM]: 6 x Pokemon (0x64 bytes cada uno); Mudkip L5 con checksum correcto
G_PLAYER_PARTY_COUNT = 0x020244E9  # OK [SYM]: u8
G_ENEMY_PARTY_COUNT = 0x020244EA  # -- [SYM]: u8
G_ENEMY_PARTY = 0x02024744  # -- [SYM]

G_MAIN_CALLBACK2 = 0x030022C4  # OK [SYM]+[M]: gMain + 0x04 (cambia según la pantalla/estado del juego)
# Valores de gMain.callback2 observados en vivo (con el bit Thumb). 0x08085E5D = mundo abierto (camión, casas,
# laboratorio); otros valores = menús, batalla, pantalla de nombre (0x080E4F59), reloj (0x08134C9D), etc.
CB2_OVERWORLD = 0x08085E5D

# Valores de gMain.callback2 dentro de una batalla, vistos con capturas de pantalla en el combate con el rival (R5f):
CB2_BATTLE_START = 0x08036FAD  # arranque del combate (transición)
CB2_BATTLE_MAIN = 0x08038421  # pantalla principal: menú «¿Qué hará?» (LUCHAR/MOCHILA/POKéMON/HUIR), elegir ataque, mensajes
CB2_BAG = 0x081AAD5D  # la Mochila abierta desde el combate
CB2_FADE = 0x080A933D  # pantalla negra al entrar y salir de la Mochila

G_MAIN_IN_BATTLE = 0x030026F9  # OK [M]: byte gMain+0x439; bit 0x02 = inBattle (True/False al empezar/terminar la 1ª batalla)
IN_BATTLE_MASK = 0x02

G_BATTLE_TYPE_FLAGS = 0x02022FEC  # OK [SYM]: u32; valió 0x14 (IS_MASTER|FIRST_BATTLE) en la 1ª batalla
G_BATTLE_OUTCOME = 0x0202433A  # -- [SYM]: u8

# Estado del script/diálogo. Observado en vivo: 0 mientras hay un cuadro de texto de script abierto,
# 2 cuando no hay ningún script activo. El valor 1 (esperando) no se ha visto todavía.
S_GLOBAL_SCRIPT_CONTEXT_STATUS = 0x03000E38  # OK [SYM]
SCRIPT_CONTEXT_DISABLED = 2

# --- struct SaveBlock1 (offsets desde el puntero) [G] -----------------------------------

SB1_POS = 0x00  # OK s16 x, s16 y
SB1_LOCATION = 0x04  # OK s8 mapGroup, s8 mapNum, s8 warpId, (pad), s16 x, s16 y
SB1_MONEY = 0x490  # -- u32, cifrado: dinero = raw XOR SaveBlock2.encryptionKey [MON]
SB1_FLAGS = 0x1270  # -- u8[300]; flag N: byte N>>3, bit N&7 [EV]
SB1_VARS = 0x139C  # --

FLAG_BADGE01_GET = 0x867  # [F] las 8 medallas son los flags 0x867..0x86E
NUM_BADGES = 8

# --- struct SaveBlock2 [G] -----------------------------------------------------------------

SB2_PLAYER_NAME = 0x00  # OK
SB2_PLAYER_GENDER = 0x08  # --
SB2_ENCRYPTION_KEY = 0xAC  # -- u32

# --- struct Pokemon (100 bytes) [P] ----------------------------------------------------------

POKEMON_SIZE = 0x64
PARTY_MAX = 6
MON_PERSONALITY = 0x00
MON_OT_ID = 0x04
MON_NICKNAME = 0x08
MON_CHECKSUM = 0x1C
MON_SECURE = 0x20  # 48 bytes cifrados (4 subestructuras de 12 bytes)
MON_STATUS = 0x50  # u32 (sin cifrar)
MON_LEVEL = 0x54  # u8
MON_HP = 0x56  # u16
MON_MAX_HP = 0x58  # u16
MON_ATTACK = 0x5A
MON_DEFENSE = 0x5C
MON_SPEED = 0x5E
MON_SP_ATTACK = 0x60
MON_SP_DEFENSE = 0x62

# Orden de las subestructuras según personality % 24 [P, GetSubstruct].
# Cada fila: posición dentro de `secure` de (Growth, Attacks, EVs/Condition, Misc).
SUBSTRUCT_ORDER = (
    (0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 3, 1, 2), (0, 2, 3, 1), (0, 3, 2, 1),
    (1, 0, 2, 3), (1, 0, 3, 2), (2, 0, 1, 3), (3, 0, 1, 2), (2, 0, 3, 1), (3, 0, 2, 1),
    (1, 2, 0, 3), (1, 3, 0, 2), (2, 1, 0, 3), (3, 1, 0, 2), (2, 3, 0, 1), (3, 2, 0, 1),
    (1, 2, 3, 0), (1, 3, 2, 0), (2, 1, 3, 0), (3, 1, 2, 0), (2, 3, 1, 0), (3, 2, 1, 0),
)

# Hacia dónde mira el jugador: gObjectEvents[0].facingDirection & 0xF (1 abajo, 2 arriba, 3 izquierda, 4 derecha).
# La base de gObjectEvents (0x02037350) y el offset 0x18 los usa env/mapdata.py; que el objeto 0 sea el jugador se
# comprueba en vivo antes de fiarse (ver specs.md, 7.6).
G_PLAYER_FACING = 0x02037350 + 0x18
