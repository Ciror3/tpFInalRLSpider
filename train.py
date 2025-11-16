from rl_zoo3.train import train
from gymnasium.envs.registration import register

register(
    id="SpiderEnv-v0",
    entry_point="enviroment:SpiderEnv",  
)
if __name__ == "__main__":
    train()


# python train.py --algo ppo --env SpiderEnv-v0 -n 50000 --optimize --n-trials 50 --n-jobs 20 --sampler tpe --pruner median --conf-file hyperparams/ppo_spider.yml --device cpu --verbose 