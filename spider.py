import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from enviroment import SpiderEnv        
from policy_callback import PolicyMapCallback  


def make_env(seed: int = 0, **env_kwargs):
    """
    Crea una función que inicializa una instancia de SpiderEnv.
    Esto es lo que necesita SubprocVecEnv para crear entornos en paralelo.
    """
    def _init():
        env = SpiderEnv(**env_kwargs)
        env.reset(seed=seed)
        return env
    return _init


def main():
    # ---------- CONFIGURACIÓN ----------
    total_timesteps = 2_000_000     
    n_envs = 4                    
    log_dir = "./logs_spider"
    models_dir = "./models_spider"
    policy_maps_dir = "./policy_maps"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(policy_maps_dir, exist_ok=True)

    # ---------- ENTORNOS EN PARALELO ----------
    env_kwargs = dict(render_mode=None)
    env = SubprocVecEnv(
        [make_env(seed=i, **env_kwargs) for i in range(n_envs)]
    )

    # ---------- MODELO PPO ----------
    model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.000355207825977822,
    n_steps=256,                      
    batch_size=8,                     
    n_epochs=20,
    gamma=0.9791381032304535,         
    gae_lambda=0.9975914694455162,    
    clip_range=0.1,
    ent_coef=1.440070886288556e-06,
    vf_coef=0.5,
    max_grad_norm=0.7456096284142648,
    policy_kwargs=dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],  
        activation_fn=torch.nn.Tanh,                
    ),
    verbose=0,
    tensorboard_log=log_dir,
)

    # ---------- CALLBACKS ----------
    policy_cb = PolicyMapCallback(
        freq=50_000,             
        save_path=policy_maps_dir,
        verbose=1
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // n_envs,   
        save_path=models_dir,
        name_prefix="ppo_spider"
    )

    callbacks = [policy_cb, checkpoint_cb]

    # ---------- ENTRENAMIENTO ----------
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True  
    )

    # ---------- GUARDAR MODELO FINAL ----------
    final_model_path = os.path.join(models_dir, "ppo_spider_final")
    model.save(final_model_path)
    print(f"Modelo final guardado en: {final_model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
