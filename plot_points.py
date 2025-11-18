import numpy as np
from stable_baselines3 import PPO
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from enviroment import SpiderEnv


model = PPO.load("/home/ciror/Desktop/rl/tps/tpfinal/tpFInalRLSpider/models_spider/ppo_spider.zip")
env = SpiderEnv(render_mode=None)
mapX = np.arange(-2, 2.01, 0.1)
mapY = np.arange(-2, 2.01, 0.1)
policy_map = np.zeros((len(mapY), len(mapX)), dtype=int)

for iy, y in enumerate(mapY):
    for ix, x in enumerate(mapX):
        obs, info = env.reset(options={"target_init_pos": np.array([x, y], dtype=np.float32)})
        action, _ = model.predict(obs, deterministic=True)
        policy_map[iy, ix] = action


plt.figure(figsize=(8, 8))
plt.imshow(policy_map, extent=[-2, 2, -2, 2], origin="lower",cmap="turbo")
plt.colorbar(label="Acción PPO")
plt.xlabel("x del target")
plt.ylabel("y del target")
plt.title("Policy map PPO")
plt.savefig("policy_map.png", dpi=150)
print("Guardado como policy_map.png")
