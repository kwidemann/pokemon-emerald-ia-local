"""Entrena un agente PPO (Stable-Baselines3, PyTorch en CPU) sobre `PokemonEmeraldEnv`.

Cada entorno usa su propia instancia de mGBA con `emulator/server.lua` cargado, en puertos consecutivos desde
`--base-port` (cada instancia se abre con la variable de entorno `POKEMON_IA_PORT`). Con `--n-envs 1` se usa un solo
emulador en el puerto 8888.

Uso:
    uv run python agents/rl/train.py --savestate runs/lab_mudkip.ss0 --total-steps 100000
    uv run python agents/rl/train.py --savestate runs/lab_mudkip.ss0 --n-envs 4 --base-port 8888 --total-steps 500000
    uv run python agents/rl/train.py --evaluate runs/rl/<corrida>/model.zip --episodes 10

Salidas en `runs/rl/<corrida>/` (ignorado por git): `model.zip`, checkpoints, curvas de TensorBoard (`tb/`) y
`train.log` con los hitos de cada episodio.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from env.pokemon_env import PokemonEmeraldEnv  # noqa: E402

FINAL_MILESTONE = 16


def make_env(savestate: str, port: int, max_steps: int, new_tile_bonus: float) -> PokemonEmeraldEnv:
    return PokemonEmeraldEnv(savestate, port=port, max_steps=max_steps, new_tile_bonus=new_tile_bonus)


def make_vec_env(savestate: str, n_envs: int, base_port: int, max_steps: int, new_tile_bonus: float):
    fns = [partial(make_env, savestate, base_port + i, max_steps, new_tile_bonus) for i in range(n_envs)]
    return DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)


class MilestoneLogger(BaseCallback):
    """Al terminar cada episodio anota los hitos alcanzados (log en archivo y curvas de TensorBoard)."""

    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self.episodes = 0
        self.t0 = time.time()

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self.episodes += 1
            ms = info.get("milestones", [])
            self.logger.record("milestones/count", len(ms))
            self.logger.record("milestones/final_reached", float(FINAL_MILESTONE in ms))
            line = (
                f"episodio {self.episodes} pasos_totales={self.num_timesteps} hitos={ms} "
                f"reloj={time.time() - self.t0:.0f}s"
            )
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return True


def evaluate(model_path: Path, args) -> None:
    env = make_vec_env(args.savestate, 1, args.base_port, args.max_steps, 0.0)
    model = PPO.load(model_path, device="cpu")
    ok = 0
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _, dones, infos = env.step(action)
            done = bool(dones[0])
        ms = infos[0].get("milestones", [])
        ok += FINAL_MILESTONE in ms
        print(f"[{ep + 1}/{args.episodes}] pasos={infos[0].get('steps')} hitos={ms}", flush=True)
    print(f"Éxito (medalla 1): {ok}/{args.episodes}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--savestate", default="runs/lab_mudkip.ss0")
    parser.add_argument("--n-envs", type=int, default=1, help="emuladores en paralelo (puertos consecutivos)")
    parser.add_argument("--base-port", type=int, default=8888)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--max-steps", type=int, default=2000, help="acciones por episodio antes de truncar")
    parser.add_argument("--new-tile-bonus", type=float, default=0.0, help="premio por casilla nueva (0 = solo hitos)")
    parser.add_argument("--n-steps", type=int, default=512, help="pasos por entorno antes de cada actualización")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="premia explorar (más alto = más aleatorio)")
    parser.add_argument("--name", help="nombre de la corrida (por defecto, la fecha)")
    parser.add_argument("--checkpoint-every", type=int, default=50_000, help="pasos entre checkpoints (0 = ninguno)")
    parser.add_argument("--evaluate", help="evaluar un model.zip en vez de entrenar")
    parser.add_argument("--episodes", type=int, default=10, help="episodios al evaluar")
    parser.add_argument("--stochastic", action="store_true", help="al evaluar, muestrear acciones en vez de la mejor")
    args = parser.parse_args()

    if args.evaluate:
        evaluate(Path(args.evaluate), args)
        return

    out = Path("runs/rl") / (args.name or datetime.now().strftime("ppo_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    vec = make_vec_env(args.savestate, args.n_envs, args.base_port, args.max_steps, args.new_tile_bonus)
    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=args.n_steps,
        batch_size=min(128, args.n_steps * args.n_envs),
        ent_coef=args.ent_coef,
        device="cpu",
        tensorboard_log=str(out / "tb"),
        verbose=1,
    )
    callbacks = [MilestoneLogger(out / "train.log")]
    if args.checkpoint_every:
        callbacks.append(
            CheckpointCallback(max(args.checkpoint_every // args.n_envs, 1), str(out / "checkpoints"), name_prefix="ppo")
        )
    print(f"Salida en {out}; {args.n_envs} emulador(es) desde el puerto {args.base_port}", flush=True)
    try:
        model.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=False)
    finally:
        model.save(out / "model")
        vec.close()
        print(f"Modelo guardado en {out / 'model.zip'}")


if __name__ == "__main__":
    main()
