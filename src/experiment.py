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

from src.custom_environment import CustomEnvironment
from src.external.tensorforce.execution import Runner
from src.util import label_episodes, delete_unused_episodes

sns.set_style("dark")
sns.set_context("paper")


class Experiment:
    def __init__(
        self,
        agent: str,
        num_episodes: int,
        args,
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
        use_newest_checkpoint: bool = False,
        eval: bool = False,
        zip: bool = False,
        test: bool = False,
        model_size: str = "training",
        nl_path: str = None,
        gui: bool = False,
    ):
        """
        Class to run individual experiments.
        :param agent: Agent specification (Path to JSON-file)
        :param num_episodes: Number of episodes to run.
        :param args: Arguments experiment has been called with.
        :param batch_agent_calls: Whether agent calls are run in batches.
        :param sync_episodes: Whether all parallel environment are synced.
        :param document: Boolean if model outputs are to be saved.
        :param wandb_entity: Name of Weights and Biases Entity results are logged to.
        :param wandb_project: Name of Weights and Biases Project results are logged to.
        :param adjust_free: Whether the agent adjusts prices freely between 0 and 10€ or incrementally.
        :param group_pricing: Whether prices are set for different income groups individually (per CPZ).
        :param num_parallel: Number of environments to run in parallel.
        :param reward_key: Key to choose reward function.
        :param checkpoint: Timestamp of checkpoint to resume.
        :param use_newest_checkpoint: Whether to use the newest checkpoint for the given reward function.
        :param eval: Whether to use one core for evaluation (necessary for evaluation phase).
        :param zip: Whether to zip the experiment directory.
        :param test: Whether to run in test mode.
        :param model_size: Model size to run experiments with, either "training" or "evaluation".
        :param nl_path: Path to NetLogo directory (for Linux users).
        :param gui: Whether NetLogo UI is shown during episodes.
        """

        self.num_episodes = num_episodes
        self.batch_agent_calls = batch_agent_calls
        self.sync_episodes = sync_episodes
        self.eval = eval
        self.zip = zip
        self.document = document
        self.num_parallel = num_parallel
        assert (
            not eval or self.num_parallel > 1
        ), "Need parallel execution for evaluation mode."
        # Check if checkpoint is given (resume if given)
        if checkpoint is not None and not use_newest_checkpoint:
            self.resume_checkpoint = True
            self.timestamp = checkpoint
        elif checkpoint is None and use_newest_checkpoint:
            self.resume_checkpoint = True
            runs = glob(
                str(Path(".").absolute().parent / "results" / reward_key) + "/*"
            )
            latest_run = max(runs, key=os.path.getctime).split("\\")[-1]
            self.timestamp = latest_run
            print(f"Resuming latest checkpoint {self.timestamp}")
        else:
            self.resume_checkpoint = False
            self.timestamp = datetime.now().strftime("%y%m-%d-%H%M")

        self.outpath = (
            Path(".").absolute().parent / "results" / reward_key / self.timestamp
        )
        # Create directory (if it does not exist yet)
        self.outpath.mkdir(parents=True, exist_ok=True)

        if self.resume_checkpoint:
            agent = dict()
            agent["directory"] = str(self.outpath / "model-checkpoints")
            agent["format"] = "checkpoint"
        else:
            # Load json file for agent init
            with open(agent, "r") as fp:
                agent = json.load(fp=fp)
            # Update checkpoint path
            agent["saver"]["directory"] = str(self.outpath / "model-checkpoints")

            # Document Config
            args["agent"] = agent
            with open(str(self.outpath / "config.txt"), "w") as outfile:
                json.dump(args, outfile)

        if "summarizer" in agent.keys():
            agent["summarizer"]["directory"] = str(self.outpath / "summarizer")

        if wandb_project is not None:
            self.wandb = wandb.init(
                dir=self.outpath,
                job_type="eval" if self.resume_checkpoint else "training",
                name=f"{reward_key}_{'group' if group_pricing else 'zone'}_{'eval' if checkpoint is not None else 'training'}",
                project=wandb_project,
                entity=wandb_entity,
                config=args,
                sync_tensorboard=True if "summarizer" in agent.keys() else False,
            )
        else:
            self.wandb = None

        env_kwargs = {
            "timestamp": self.timestamp,
            "reward_key": reward_key,
            "document": self.document,
            "wandb": self.wandb,
            "adjust_free": adjust_free,
            "group_pricing": group_pricing,
            "model_size": model_size,
            "nl_path": nl_path,
            "gui": gui,
            "test": test,
        }
        # Create appropriate number of environments
        if num_parallel > 1:
            self.runner = Runner(
                agent=agent,
                environment=CustomEnvironment,
                remote="multiprocessing",
                evaluation=self.eval,
                num_parallel=num_parallel,
                max_episode_timesteps=24,
                **env_kwargs,
            )
        else:
            self.runner = Runner(
                agent=agent,
                environment=CustomEnvironment,
                max_episode_timesteps=24,
                **env_kwargs,
            )
            self.batch_agent_calls = False
            self.sync_episodes = False

    def run(self):
        """
        Runs actual experiments and saves results.
        :return:
        """
        print(f"Training for {self.num_episodes} episodes")
        self.runner.run(
            num_episodes=self.num_episodes,
            batch_agent_calls=self.batch_agent_calls,
            sync_episodes=self.sync_episodes,
            evaluation=self.eval if self.num_parallel == 1 else False,
            save_best_agent=str(self.outpath / "best_agent"),
        )

        # Saving results
        self.save_results()
        if self.eval:
            self.save_results(mode="eval")

        # Delete unused episodes
        if self.document:
            delete_unused_episodes(self.outpath)

        # Close runner
        self.runner.close()

        # Save Experiment output to Weights and Biases
        if self.wandb is not None:
            self.wandb.save(f"{str(self.outpath)}/*", policy="now")

        # Zip experiment directory
        if self.zip:
            shutil.make_archive(str(self.outpath), "zip", self.outpath)
            print("directory zipped")

    def save_results(self, mode="training"):
        """
        Saves results, result plots and, possibly, episode results of experiment.
        :param mode: Either "training" or "evaluation".
        :return:
        """
        # Accessing the appropriate metrics from runner
        if mode == "training":
            rewards = np.asarray(self.runner.episode_returns)
            episode_length = np.asarray(self.runner.episode_timesteps)
        elif mode == "eval":
            rewards = np.asarray(self.runner.evaluation_returns)
            episode_length = np.asarray(self.runner.evaluation_timesteps)

        mean_reward = rewards / episode_length
        metrics_df = pd.DataFrame.from_dict(
            {
                "rewards": rewards,
                "episode_length": episode_length,
                "mean_reward": mean_reward,
            }
        )

        csv_path = self.outpath / f"{mode}_result_{self.num_episodes}.csv"
        i = 1
        # Check if results file already exists
        while csv_path.is_file():
            csv_path = self.outpath / f"{mode}_result_{self.num_episodes} ({i}).csv"
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

        pdf_path = self.outpath / f"{mode}_result_reward_plot_{self.num_episodes}.pdf"
        i = 1
        # Check if results file already exists
        while pdf_path.is_file():
            pdf_path = (
                self.outpath
                / f"{mode}_result_reward_plot_{self.num_episodes} ({i}).pdf"
            )
            i += 1

        fig.savefig(str(pdf_path), dpi=300)
