import json
import shutil
from datetime import datetime
from pathlib import Path
from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cmcrameri import cm
import wandb
from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import load_results

from wandb.integration.sb3 import WandbCallback

from src.parking_environment import ParkingEnvironment
from src.util import label_episodes, delete_unused_episodes

sns.set_style("dark")
sns.set_context("paper")

SEED = 2023


class Experiment:
    def __init__(
        self,
        agent: str,
        train_episodes: int,
        args,
        eval_episodes: int = 50,
        batch_agent_calls: bool = False,
        sync_episodes: bool = False,
        document: bool = True,
        wandb_project: str = None,
        wandb_entity: str = None,
        adjust_free: bool = False,
        group_pricing: bool = False,
        num_parallel: int = 1,
        reward_key: str = "occupancy",
        checkpoint: str = None,
        eval: bool = False,
        zip: bool = False,
        test: bool = False,
        model_size: str = "training",
        nl_path: str = None,
        render_mode: str = None,
    ):
        """
        Class to run individual experiments.
        :param agent: Agent specification (Path to JSON-file)
        :param train_episodes: Number of episodes for training.
        :param eval_episodes: Number of episodes for evaluating.
        :param args: Arguments experiment has been called with.
        :param document: Boolean if model outputs are to be saved.
        :param wandb_entity: Name of Weights and Biases Entity results are logged to.
        :param wandb_project: Name of Weights and Biases Project results are logged to.
        :param adjust_free: Whether the agent adjusts prices freely between 0 and 10€ or incrementally.
        :param group_pricing: Whether prices are set for different income groups individually (per CPZ).
        :param num_parallel: Number of environments to run in parallel.
        :param reward_key: Key to choose reward function.
        :param checkpoint: Timestamp of checkpoint to resume.
        :param eval: Whether to use one core for evaluation (necessary for evaluation phase).
        :param zip: Whether to zip the experiment directory.
        :param test: Whether to run in test mode.
        :param model_size: Model size to run experiments with, either "training" or "evaluation".
        :param nl_path: Path to NetLogo directory (for Linux users).
        :param render_mode: Whether NetLogo UI is shown during episodes (value "human" or None (default)).
        """

        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.eval = eval
        self.zip = zip
        self.document = document
        self.num_parallel = num_parallel

        # Check if checkpoint is given (resume if given)
        if checkpoint is not None:
            self.resume_checkpoint = True
            self.timestamp = checkpoint
        else:
            self.resume_checkpoint = False
            self.timestamp = datetime.now().strftime("%y%m-%d-%H%M")

        self.outpath = (
            Path(".").absolute().parent
            / "results"
            / reward_key
            / ("group" if group_pricing else "zone")
            / self.timestamp
        )
        # Create directory (if it does not exist yet)
        self.outpath.mkdir(parents=True, exist_ok=True)

        self.env_kwargs = {
            "timestamp": self.timestamp,
            "reward_key": reward_key,
            "document": self.document,
            "adjust_free": adjust_free,
            "group_pricing": group_pricing,
            "model_size": model_size,
            "nl_path": nl_path,
            "render_mode": render_mode,
            "test": test,
        }

        # register Gym Environment
        eval_env = make_vec_env(
            ParkingEnvironment,
            env_kwargs=self.env_kwargs,
            n_envs=1,
            seed=SEED,
            vec_env_cls=SubprocVecEnv,
        )

        self.callbacks = [
            EvalCallback(
                eval_env,
                best_model_save_path=str(self.outpath / "log" / "eval" / "model"),
                log_path=str(self.outpath / "log" / "eval"),
                eval_freq=48,
                deterministic=True,
                render=False,
            )
        ]

        if wandb_project is not None:
            self.wandb = wandb.init(
                dir=self.outpath,
                job_type="eval" if self.resume_checkpoint else "training",
                name=f"{self.timestamp}_{reward_key}_{'group' if group_pricing else 'zone'}_{'eval' if checkpoint is not None else 'training'}",
                project=wandb_project,
                entity=wandb_entity,
                config=args,
                sync_tensorboard=True,
            )
            self.callbacks.append(
                WandbCallback(
                    gradient_save_freq=1000,
                    model_save_path=str(self.outpath / "log" / "eval" / "model"),
                    verbose=2,
                )
            )
        else:
            self.wandb = None

        self.venv = make_vec_env(
            ParkingEnvironment,
            env_kwargs=self.env_kwargs,
            n_envs=self.num_parallel,
            seed=SEED,
            monitor_dir=str(self.outpath / "log"),
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={
                "start_method": "spawn"
            },  # Jpype can only handle spawn processes without pipe errors
        )
        self.model = PPO(
            "MlpPolicy",
            env=self.venv,
            verbose=1,
            n_steps=24,
            tensorboard_log=str(self.outpath / "log" / "tensorboard"),
        )

    def run(self):
        """
        Runs actual experiments and saves results.
        :return:
        """
        print(f"Training for {self.train_episodes} episodes")

        # Almost infinite number of timesteps, but the training will stop early as soon as the max number of episodes is reached
        self.model.learn(
            total_timesteps=self.train_episodes * 24,
            progress_bar=True,
            callback=self.callbacks,
        )
        self.venv.close()

        # # Saving results
        self.save_results()

        if self.eval:
            print(f"Evaluating for {self.eval_episodes} episodes")
            self.env_kwargs["eval"] = True
            self.env_kwargs["document"] = True
            eval_env = make_vec_env(
                ParkingEnvironment,
                env_kwargs=self.env_kwargs,
                n_envs=1,
                seed=SEED,
                vec_env_cls=SubprocVecEnv,
            )

            best_model_path = self.outpath / "log" / "eval" / "model" / "best_model.zip"

            if best_model_path.is_file():
                self.model.load(best_model_path)

            returns = evaluate_policy(
                model=self.model,
                env=eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
                return_episode_rewards=True,
            )
            print(returns)
            self.save_results(mode="eval", eval_results=returns)

        # # Delete unused episodes
        # if self.document:
        #     delete_unused_episodes(self.outpath)

        # Save Experiment output to Weights and Biases
        if self.wandb is not None:
            # self.wandb.save(f"{str(self.outpath)}/*", policy="now")
            self.wandb.finish()

        # Zip experiment directory
        if self.zip:
            shutil.make_archive(str(self.outpath), "zip", self.outpath)
            print("directory zipped")

    def save_results(self, mode="training", eval_results=None):
        """
        Saves results, result plots and, possibly, episode results of experiment.
        :param mode: Either "training" or "evaluation".
        :return:
        """
        if mode == "training":
            result_df = load_results(str(self.outpath / "log"))
            rewards = result_df["r"].values
            episode_length = result_df["l"].values
        else:
            rewards = np.array(eval_results[0])
            episode_length = np.array(eval_results[1])
        mean_reward = rewards / episode_length
        metrics_df = pd.DataFrame.from_dict(
            {
                "rewards": rewards,
                "episode_length": episode_length,
                "mean_reward": mean_reward,
            }
        )

        csv_path = (
            self.outpath
            / f"{mode}_result_{self.train_episodes if mode=='training' else self.eval_episodes}.csv"
        )
        i = 1
        # Check if results file already exists
        while csv_path.is_file():
            csv_path = (
                self.outpath
                / f"{mode}_result_{self.train_episodes if mode=='training' else self.eval_episodes} ({i}).csv"
            )
            i += 1

        metrics_df.to_csv(str(csv_path))

        # Rename best, worst and median performance
        if self.document and mode == "eval":
            label_episodes(self.outpath, metrics_df, mode)

        # Plotting mean-reward over episodes
        fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)
        ax.plot(
            range(len(rewards)), metrics_df.rewards, linewidth=5, color=cm.bamako(0)
        )
        rolling_average = metrics_df.rewards.rolling(35).mean()
        ax.plot(
            range(len(rolling_average)),
            rolling_average,
            linewidth=3,
            color=cm.bamako(1.0),
        )
        ax.set_ylabel("Reward per Episode", fontsize=30)
        ax.set_xlabel("Episodes", fontsize=30)
        ax.grid(True)
        ax.tick_params(axis="y", labelsize=25)
        ax.tick_params(axis="x", labelsize=25)

        pdf_path = (
            self.outpath
            / f"{mode}_result_reward_plot_{self.train_episodes if mode=='training' else self.eval_episodes}.pdf"
        )
        i = 1
        # Check if results file already exists
        while pdf_path.is_file():
            pdf_path = (
                self.outpath
                / f"{mode}_result_reward_plot_{self.train_episodes if mode=='training' else self.eval_episodes} ({i}).pdf"
            )
            i += 1

        fig.savefig(str(pdf_path), dpi=300)
