import json
import platform
import uuid
from pathlib import Path

import numpy as np
import pyNetLogo

import gymnasium as gym
from gymnasium import spaces
from src.util import (
    occupancy_reward_function,
    n_cars_reward_function,
    social_reward_function,
    speed_reward_function,
    composite_reward_function,
    glob_outcome_reward_function,
    intergroup_outcome_reward_function,
    document_episode,
    compute_jenson_shannon,
)

COLOURS = ["yellow", "green", "teal", "blue"]
REWARD_FUNCTIONS = {
    "occupancy": occupancy_reward_function,
    "n_cars": n_cars_reward_function,
    "social": social_reward_function,
    "speed": speed_reward_function,
    "composite": composite_reward_function,
    "global_outcome": glob_outcome_reward_function,
    "intergroup_outcome": intergroup_outcome_reward_function,
}


class ParkingEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        timestamp: str,
        reward_key: str,
        document: bool = False,
        adjust_free: bool = False,
        group_pricing: bool = False,
        model_size: str = "training",
        nl_path: str = None,
        render_mode=None,
        eval: bool = False,
        test: bool = False,
    ):
        """
        Wrapper-Class to interact with NetLogo parking simulations.

        :param timestamp: Timestamp of episode.
        :param reward_key: Key to choose reward function
        :param document: Boolean to control whether individual episode results are saved.
        :param adjust_free: Boolean to control whether prices are adjusted freely or incrementally
        :param model_size: Model size to run experiments with, either "training" or "evaluation".
        :param nl_path: Path to NetLogo Installation (for Linux users)
        :param gui: Whether NetLogo UI is shown during episodes.
        :param group_pricing:
        :param eval:
        :param test:
        """
        super().__init__()
        self.timestamp = timestamp
        self.outpath = (
            Path(".").absolute().parent / "results" / reward_key / self.timestamp
        )
        # Unique id to identify Process
        self.uuid = uuid.uuid4()
        self.truncated = False
        self.episode_end = False
        self.document = document
        self.adjust_free = adjust_free
        self.group_pricing = group_pricing
        self.eval = eval
        self.test = test
        self.reward_function = REWARD_FUNCTIONS[reward_key]
        self.reward_sum = 0
        # Load model parameters
        config_path = Path(__file__).with_name("model_config.json")
        with open(config_path, "r") as fp:
            model_config = json.load(fp=fp)
        # Connect to NetLogo
        if platform.system() == "Linux":
            self.nl = pyNetLogo.NetLogoLink(
                gui=True if render_mode is not None else False,
                netlogo_home=nl_path,
                netlogo_version="6.2",
            )
        else:
            self.nl = pyNetLogo.NetLogoLink(
                gui=True if render_mode is not None else False
            )
        self.nl.load_model(str(Path(__file__).with_name("Model.nlogo")))
        # Set model size
        self.set_model_size(model_config, model_size)
        self.nl.command("setup")

        # create file and provide path of turtle.csv
        if self.eval and self.document:
            self.nl.command("set document-turtles true")
            open(self.outpath / f"turtles_{self.uuid}.csv", "w").close
            turtle_file_path = str(self.outpath / f"turtles_{self.uuid}.csv").replace(
                "\\", "/"
            )
            self.nl.command(f'set output-turtle-file-path "{turtle_file_path}"')
        else:
            self.nl.command("set document-turtles false")

        if self.group_pricing:
            self.nl.command("set group-pricing true")

        # Disable rendering of view
        if render_mode is None:
            self.nl.command("no-display")
        # Turn baseline pricing mechanism off
        self.nl.command("set dynamic-pricing-baseline false")
        # Record data
        self.nl.command("ask one-of cars [record-data]")
        # Save current state in dict
        self.current_state = dict()
        self.current_state["ticks"] = self.nl.report("ticks")
        self.current_state["n_cars"] = float(self.nl.report("n-cars"))
        self.current_state["overall_occupancy"] = self.nl.report("global-occupancy")

        # General information about model
        self.temporal_resolution = self.nl.report("temporal-resolution")
        self.n_garages = self.nl.report("num-garages")
        self.colours = COLOURS

        # Attributes of super class that have to be set

        self.action_space = self.define_action_space()
        self.observation_space = self.define_state_space()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def define_action_space(self):
        if self.adjust_free:
            num_values = 21
        else:
            num_values = 5
        actions = []
        if self.group_pricing:
            for _ in self.colours:
                for __ in ["low", "middle", "high"]:
                    actions.append(num_values)
        else:
            for _ in self.colours:
                actions.append(num_values)
        return spaces.MultiDiscrete(actions)

    def define_state_space(self):
        states_num = 11
        if self.n_garages > 0:
            states_num += 1

        return spaces.Box(low=0.0, high=1.0, shape=(states_num,), dtype=np.float32)

    def set_model_size(self, model_config, model_size):
        """
        Set NetLogo model to the appropriate size.
        :param model_config: Config dict containing grid size as well as number of cars and garages.
        :param model_size: Model size to run experiments with, either "training" or "evaluation".
        :return:
        """
        print(f"Configuring model size for {model_size}")
        max_x_cor = model_config[model_size]["max_x_cor"]
        max_y_cor = model_config[model_size]["max_y_cor"]
        self.nl.command(
            f"resize-world {-max_x_cor} {max_x_cor} {-max_y_cor} {max_y_cor}"
        )
        for key in model_config[model_size].keys():
            if "max" in key:
                continue
            self.nl.command(
                f'set {key.replace("_", "-")} {model_config[model_size][key]}'
            )
            if self.test:
                assert (
                    self.nl.report(f'{key.replace("_", "-")}')
                    == model_config[model_size][key]
                ), f"{key} was not correctly set."

    def _get_obs(self):
        """
        Query current state of simulation.
        """
        # Update globals
        self.nl.command("ask one-of cars [record-data]")
        self.current_state["ticks"] = self.nl.report("ticks")
        self.current_state["n_cars"] = self.nl.report("n-cars")
        self.current_state["overall_occupancy"] = self.nl.report("global-occupancy")
        # self.current_state['city_income'] = self.nl.report("city-income")
        self.current_state["mean_speed"] = self.nl.report("mean-speed")
        self.current_state["normalized_share_low"] = self.nl.report(
            "normalized-share-low"
        )

        # Append fees and current occupation to state
        for c in self.colours:
            self.current_state[f"{c}-lot occupancy"] = self.nl.report(
                f"{c}-lot-current-occup"
            )
            if self.group_pricing:
                self.current_state[f"{c}-lot fees"] = self.nl.report(
                    f"[group-fees] of one-of {c}-lot"
                )
            else:
                self.current_state[f"{c}-lot fee"] = self.nl.report(
                    f"mean [fee] of {c}-lot"
                )

        if self.n_garages > 0:
            self.current_state["garages occupancy"] = self.nl.report(
                "garages-current-occup"
            )

        # Add outcome divergences to state (compute current outcomes first)
        # self.nl.command("compute-outcome")
        self.current_state["global_outcome_divergence"] = compute_jenson_shannon(
            self.nl, intergroup=False
        )
        self.current_state["intergroup_outcome_divergence"] = compute_jenson_shannon(
            self.nl, intergroup=True
        )

        self.current_state["low_income_outcome"] = np.average(
            self.nl.report(f"get-outcomes 0")
        ) - self.nl.report("min-util")

        state = []
        state.append(float(self.current_state["ticks"] / 21600))
        state.append(np.around(self.current_state["n_cars"], 2))
        state.append(np.around(self.current_state["normalized_share_low"], 2))
        state.append(
            np.around(self.current_state["mean_speed"], 2)
            if self.current_state["mean_speed"] <= 1.0
            else 1.0
        )

        for key in sorted(self.current_state.keys()):
            if "occupancy" in key:
                state.append(np.around(self.current_state[key], 2))
            # elif "fees" in key:
            #     for fee in self.current_state[key]:
            #         state.append(np.around(fee, 2) / 10)
            # elif "fee" in key:
            #     state.append(np.around(self.current_state[key], 2) / 10)

        state.append(float(self.current_state["global_outcome_divergence"]))
        state.append(float(self.current_state["intergroup_outcome_divergence"]))

        return np.array(state, dtype=np.float32)

    def reset(self, seed=None):

        # super().reset()
        self.nl.command("setup")
        # Turn baseline pricing mechanism off
        self.nl.command("set dynamic-pricing-baseline false")
        # Record data
        self.nl.command("ask one-of cars [record-data]")
        self.truncated = False
        self.episode_end = False
        self.reward_sum = 0
        self.current_state["ticks"] = self.nl.report("ticks")
        self.current_state["n_cars"] = float(self.nl.report("n-cars"))
        self.current_state["overall_occupancy"] = self.nl.report("global-occupancy")

        state = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return state, {"info": "no-info"}

    def step(self, actions):
        next_state = self.compute_step(actions)
        terminated, truncated = self.terminal()
        reward = self.reward()
        self.reward_sum += reward

        if self.render_mode == "human":
            self._render_frame()

        if terminated and self.document and self.eval:
            print(self.current_state)
            self.nl.command(
                "ask cars [document-turtle]"
            )  # ask remaining turtles to document
            self.nl.command("file-close")  # close stream of turtle.csv
            document_episode(self.nl, self.outpath, self.reward_sum, self.uuid)

        return next_state, reward, terminated, truncated, {"info": "no-info"}

    def compute_step(self, actions):
        """
        Moves simulation one time step forward and records current state.
        :param actions: actions to be taken in next time step
        :return:
        """
        # Move simulation forward
        self.nl.repeat_command("go", self.temporal_resolution / 2)

        # Adjust prices and query state
        if self.adjust_free:
            new_state = self.adjust_prices_free(actions)
        # else:
        #     pass
        #     # new_state = self.adjust_prices_step(actions)

        return new_state

    def close(self):
        self.nl.kill_workspace()
        super().close()

    def terminal(self):
        """
        Determine whether episode ended (equivalent of 12 hours have passed) or finishing criteria
        (minimum number of cars) is reached
        :return:
        """
        self.episode_end = self.current_state["ticks"] >= self.temporal_resolution * 12
        self.truncated = self.current_state["n_cars"] < 0.1

        return self.episode_end, self.truncated

    def adjust_prices_free(self, actions):
        """
        Adjust prices freely in the interval from 0 to 10 in the simulation according to the actions taken by the agent.
        :param actions:
        :return:
        """
        i = 0
        for c in COLOURS:
            if self.group_pricing:
                new_fees = []
                for j in range(3):
                    new_fees.append(actions[i] / 2)
                    i += 1
                self.nl.command(
                    f"change-group-fees-free {c}-lot {new_fees}".replace(",", "")
                )
                if self.test:
                    assert all(
                        [
                            fee == action
                            for fee, action in zip(
                                self.nl.report(f"[group-fees] of one-of {c}-lot"),
                                new_fees,
                            )
                        ]
                    )
            else:
                new_fee = actions[i] / 2
                i += 1
                self.nl.command(f"change-fee-free {c}-lot {new_fee}")
                if self.test:
                    assert self.nl.report(f"mean [fee] of {c}-lot") == new_fee

        return self._get_obs()

    def render(self):
        """

        :return:
        """
        self._render_frame()

    def _render_frame(self):
        """

        :return:
        """
        # Update view in NetLogo once
        self.nl.command("display")
        self.nl.command("no-display")

    def reward(self):
        """
        Return the adequate reward function (defined in util.py)
        :return:
        """
        return self.reward_function(
            colours=self.colours, current_state=self.current_state
        )
