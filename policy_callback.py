import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from enviroment import SpiderEnv


class PolicyMapCallback(BaseCallback):
    """
    Callback que cada 'freq' timesteps genera y guarda un policy map.
    """
    def __init__(self, freq=50000, save_path="policy_maps", verbose=1):
        super().__init__(verbose)
        self.freq = freq
        self.save_path = save_path

    def _init_callback(self):
        import os
        os.makedirs(self.save_path, exist_ok=True)
        # Creamos el entorno solo para generar los maps
        self.eval_env = SpiderEnv(render_mode=None)

    def _on_step(self) -> bool:
        if self.n_calls % self.freq != 0:
            return True

        # Construir el policy map
        mapX = np.arange(-2, 2.01, 0.1)
        mapY = np.arange(-2, 2.01, 0.1)
        policy_map = np.zeros((len(mapY), len(mapX)), dtype=int)

        for iy, y in enumerate(mapY):
            for ix, x in enumerate(mapX):
                obs, info = self.eval_env.reset(
                    options={"target_init_pos": np.array([x, y], dtype=np.float32)}
                )
                action, _ = self.model.predict(obs, deterministic=True)
                policy_map[iy, ix] = action

        # Guardar imagen
        filename = f"{self.save_path}/policy_map_step_{self.num_timesteps}.png"
        plt.figure(figsize=(8, 8))
        plt.imshow(policy_map, extent=[-2, 2, -2, 2], origin="lower", cmap="turbo")
        plt.colorbar(label="Acción PPO")
        plt.xlabel("x del target")
        plt.ylabel("y del target")
        plt.title(f"Policy map PPO - Step {self.num_timesteps}")
        plt.savefig(filename, dpi=150)
        plt.close()

        if self.verbose > 0:
            print(f"[Callback] Guardado {filename}")

        return True
