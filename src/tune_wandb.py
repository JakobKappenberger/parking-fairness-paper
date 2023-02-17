import wandb
import sys
from datetime import datetime
from src.custom_environment import CustomEnvironment


import numpy as np


from external.tensorforce import Runner


def train():
    wandb_instance = wandb.init()

    timestamp = datetime.now().strftime("%y%m-%d-%H%M")
    env_kwargs = {
        "timestamp": timestamp,
        "reward_key": wandb_instance.config.reward_key,
        "document": False,
        "wandb": wandb_instance,
        "adjust_free": True,
        "nl_path": None,
        "group_pricing": wandb_instance.config.group_pricing,
    }

    agent = {
        "agent": "ppo",
        "learning_rate": wandb_instance.config.learning_rate,
        "discount": wandb_instance.config.discount,
        "exploration": wandb_instance.config.exploration,
        "entropy_regularization": wandb_instance.config.entropy_regularization,
        "batch_size": wandb_instance.config.batch_size,
    }

    if wandb_instance.config.num_parallel is None:
        runner = Runner(
            agent=agent,
            environment=CustomEnvironment,
            max_episode_timesteps=24,
            **env_kwargs,
        )
        runner.run(num_episodes=wandb_instance.config.num_episodes, use_tqdm=True)
    else:
        runner = Runner(
            agent=agent,
            environment=CustomEnvironment,
            max_episode_timesteps=24,
            num_parallel=min(
                wandb_instance.config.num_parallel, wandb_instance.config.batch_size
            ),
            remote="multiprocessing",
            **env_kwargs,
        )
        runner.run(
            num_episodes=wandb_instance.config.num_episodes,
            batch_agent_calls=True,
            sync_episodes=True,
            use_tqdm=True,
        )
    runner.close()

    mean_average_reward = float(np.mean(runner.episode_returns, axis=0))
    # TODO: Change number of rewards for mean_final_reward
    mean_final_reward = float(np.mean(runner.episode_returns[-200:], axis=0))
    wandb.log({"loss": -(mean_average_reward + mean_final_reward)})


if __name__ == "__main__":
    train()
