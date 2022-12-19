from src.gym_environment import GymEnvironment
from datetime import datetime
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

if __name__ == "__main__":
    env_kwargs = {
        "timestamp": datetime.now().strftime("%y%m-%d-%H%M"),
        "reward_key": "occupancy",
        "document": False,
        "adjust_free": True,
        "group_pricing": True,
        "model_size": "training",
        "nl_path": None,
        "render_mode": "human",
        "test": False,
    }
    env = GymEnvironment(**env_kwargs)
    #check_env(env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=96)
