import wandb
import sys
from pathlib import Path
from datetime import datetime
from src.parking_environment import ParkingEnvironment


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.evaluation import evaluate_policy

from wandb.integration.sb3 import WandbCallback

import numpy as np
from util import linear_schedule


SEED = 2023


def train():
    wandb_instance = wandb.init()

    timestamp = datetime.now().strftime("%y%m-%d-%H%M")

    log_path = (
        Path(".").absolute().parent
        / "tuning"
        / wandb_instance.config.reward_key
        / ("group" if wandb_instance.config.group_pricing else "zone")
        / timestamp
    )
    # Create directory (if it does not exist yet)
    log_path.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "timestamp": timestamp,
        "reward_key": wandb_instance.config.reward_key,
        "document": False,
        "adjust_free": True,
        "nl_path": wandb_instance.config.nl_path,
        "model_path" : wandb_instance.config.model_path,
        "group_pricing": wandb_instance.config.group_pricing,
    }

    wandb_instance.config["batch_size"] = (
        wandb_instance.config.batch_size_factor
        * wandb_instance.config.n_steps
        * wandb_instance.config.num_parallel
    )
    ppo_config = {
        "batch_size": wandb_instance.config["batch_size"],
        "learning_rate": wandb_instance.config.learning_rate,
        "lr_schedule": wandb_instance.config.lr_schedule,
        "gamma": wandb_instance.config.gamma,
        "n_steps": wandb_instance.config.n_steps,
    }
    if ppo_config["lr_schedule"] == "linear":
        ppo_config["learning_rate"] = linear_schedule(ppo_config["learning_rate"])
    # Remove lr schedule from ppo_config dictionary
    del ppo_config["lr_schedule"]

    eval_env = make_vec_env(
        ParkingEnvironment,
        env_kwargs=env_kwargs,
        n_envs=1,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,
    )

    if wandb_instance.config["normalize"]:
        eval_env = VecNormalize(eval_env, norm_reward=False, training=False)

    callbacks = [
        # EvalCallback(
        #     eval_env,
        #     best_model_save_path=str(log_path / "log" / "eval" / "model"),
        #     eval_freq=20
        #     * wandb_instance.config.n_steps,
        #     deterministic=True,
        #     render=False,
        #     callback_after_eval=StopTrainingOnNoModelImprovement(
        #         max_no_improvement_evals=50, min_evals=100, verbose=1
        #     ),
        # ),
        WandbCallback(
            verbose=2,
        ),
    ]

    venv = make_vec_env(
        ParkingEnvironment,
        env_kwargs=env_kwargs,
        n_envs=wandb_instance.config.num_parallel,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={
            "start_method": "spawn"
        },  # Jpype can only handle spawn processes without pipe errors
    )
    if wandb_instance.config["normalize"]:
        venv = VecNormalize(venv)

    model = PPO(
        "MlpPolicy",
        env=venv,
        verbose=1,
        tensorboard_log=str(log_path / "log" / "tensorboard"),
        # Add arguments from JSON file (learning rate, etc.)
        **ppo_config,
    )

    model.learn(
        total_timesteps=wandb_instance.config.num_episodes * 24,
        progress_bar=True,
        callback=callbacks,
    )
    if wandb_instance.config["normalize"]:
        venv.save(str(log_path / "log" / "norm_stats.pkl"))
    venv.close()

    env_kwargs["eval"] = True
    env_kwargs["model_size"] = "evaluation"
    eval_env = make_vec_env(
        ParkingEnvironment,
        env_kwargs=env_kwargs,
        n_envs=wandb_instance.config.num_parallel,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,
    )
    if wandb_instance.config["normalize"]:
        eval_env = VecNormalize.load(str(log_path / "log" / "norm_stats.pkl"), eval_env)
        #  do not update them at test time
        eval_env.training = False
        # reward normalization is not needed at test time
        eval_env.norm_reward = False

    # best_model_path = log_path / "log" / "eval" / "model" / "best_model.zip"
    #
    # if best_model_path.is_file():
    #     model.load(best_model_path)
    #     print("Best eval model loaded!")

    returns = evaluate_policy(
        model=model,
        env=eval_env,
        n_eval_episodes=100,
        deterministic=True,
        return_episode_rewards=True,
    )

    mean_average_reward = float(np.mean(returns[0], axis=0))
    wandb.log({"loss": -(mean_average_reward)})


if __name__ == "__main__":
    train()
