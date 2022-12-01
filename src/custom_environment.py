import json
import platform
import uuid
from pathlib import Path

import numpy as np
import pyNetLogo

from src.external.tensorforce.environments import Environment
from src.util import (
    occupancy_reward_function,
    n_cars_reward_function,
    social_reward_function,
    speed_reward_function,
    composite_reward_function,
    glob_outcome_reward_function,
    intergroup_outcome_reward_function,
    low_outcome_reward_function,
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
    "low_income_outcome": low_outcome_reward_function
}


class CustomEnvironment(Environment):
    def __init__(
        self,
        timestamp: str,
        reward_key: str,
        document: bool = False,
        adjust_free: bool = False,
        group_pricing: bool = False,
        model_size: str = "training",
        nl_path: str = None,
        gui: bool = False,
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
        :param gui: Whether or not NetLogo UI is shown during episodes.
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
        self.finished = False
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
                gui=gui, netlogo_home=nl_path, netlogo_version="6.2"
            )
        else:
            self.nl = pyNetLogo.NetLogoLink(gui=gui)
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
        if not gui:
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

    def states(self):
        states_num = 15
        if self.n_garages > 0:
            states_num += 1
        if self.group_pricing:
            states_num += 8

        return dict(type="float", shape=(states_num,), min_value=0.0, max_value=1.0)

    def actions(self):
        if self.adjust_free:
            num_values = 21
        else:
            num_values = 5
        actions = {}
        if self.group_pricing:
            for color in COLOURS:
                for income_group in ["low", "middle", "high"]:
                    actions[f"{color}_{income_group}"] = dict(
                        type="int", num_values=num_values
                    )
        else:
            for color in COLOURS:
                actions[color] = dict(type="int", num_values=num_values)
        return actions

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        self.nl.kill_workspace()
        super().close()

    def reset(self):
        self.nl.command("setup")
        # Turn baseline pricing mechanism off
        self.nl.command("set dynamic-pricing-baseline false")
        # Record data
        self.nl.command("ask one-of cars [record-data]")
        self.finished = False
        self.episode_end = False
        self.reward_sum = 0
        self.current_state["ticks"] = self.nl.report("ticks")
        self.current_state["n_cars"] = float(self.nl.report("n-cars"))
        self.current_state["overall_occupancy"] = self.nl.report("global-occupancy")

        state = self.get_state()
        return state

    def execute(self, actions):
        next_state = self.compute_step(actions)
        terminal = self.terminal()
        reward = self.reward()
        self.reward_sum += reward

        return next_state, terminal, reward

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
        else:
            new_state = self.adjust_prices_step(actions)

        return new_state

    def adjust_prices_free(self, actions):
        """
        Adjust prices freely in the interval from 0 to 10 in the simulation according to the actions taken by the agent.
        :param actions:
        :return:
        """
        for c in COLOURS:
            if self.group_pricing:
                new_fees = []
                for key in actions.keys():
                    if c in key:
                        new_fees.append(actions[key] / 2)
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
                new_fee = actions[c] / 2
                self.nl.command(f"change-fee-free {c}-lot {new_fee}")
                if self.test:
                    assert self.nl.report(f"mean [fee] of {c}-lot") == new_fee

        return self.get_state()

    def adjust_prices_step(self, actions):
        """
        Adjust prices incrementally in the simulation according to the actions taken by the agent.
        :param actions:
        :return:
        """
        action_translation = dict(zip(range(0, 5), [-0.5, -0.25, 0, 0.25, 0.5]))
        for c in COLOURS:
            if self.group_pricing:
                fee_updates = []
                for key in actions.keys():
                    if c in key:
                        fee_updates.append(action_translation[actions[key]])
                self.nl.command(
                    f"change-group-fees {c}-lot {fee_updates}".replace(",", "")
                )
                if self.test:
                    assert all(
                        [
                            fee == action
                            for fee, action in zip(
                                self.nl.report(f"[group-fees] of one-of {c}-lot"),
                                [
                                    sum(x)
                                    for x in zip(
                                        self.current_state[f"{c}-lot fees"], fee_updates
                                    )
                                ],
                            )
                        ]
                    )
            else:
                c_action = actions[c]
                self.nl.command(f"change-fee {c}-lot {action_translation[c_action]}")
                if self.test:
                    assert (
                        self.nl.report(f"mean [fee] of {c}-lot")
                        == self.current_state[f"{c}-lot fee"]
                        + action_translation[c_action]
                    )

        return self.get_state()

    def get_state(self):
        """
        Query current state of simulation.
        """
        # Update view in NetLogo once
        self.nl.command("display")
        self.nl.command("no-display")
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
            elif "fees" in key:
                for fee in self.current_state[key]:
                    state.append(np.around(fee, 2) / 10)
            elif "fee" in key:
                state.append(np.around(self.current_state[key], 2) / 10)

        state.append(np.float(self.current_state["global_outcome_divergence"]))
        state.append(np.float(self.current_state["intergroup_outcome_divergence"]))

        if not self.adjust_free:
            action_masks = {}
            updates = [-0.5, -0.25, 0, 0.25, 0.5]
            for c in COLOURS:
                if self.group_pricing:
                    for i, group in enumerate(["low", "middle", "high"]):
                        group_key = f"{group}_mask"
                        action_masks[f"{c}_{group_key}"] = np.ones(5, dtype=bool)
                        for j, up in enumerate(updates):
                            if (
                                self.current_state[f"{c}-lot fees"][i] + up < 0
                                or self.current_state[f"{c}-lot fees"][i] + up > 10
                            ):
                                action_masks[f"{c}_{group_key}"][j] = False
                else:
                    c_key = f"{c}_mask"
                    action_masks[c_key] = np.ones(5, dtype=bool)
                    for i, up in enumerate(updates):
                        if (
                            self.current_state[f"{c}-lot fee"] + up < 0
                            or self.current_state[f"{c}-lot fee"] + up > 10
                        ):
                            action_masks[c_key][i] = False
            return dict(state=state, **action_masks)
        else:
            return state

    def terminal(self):
        """
        Determine whether episode ended (equivalent of 12 hours have passed) or finishing criteria
        (minimum number of cars) is reached
        :return:
        """
        self.episode_end = self.current_state["ticks"] >= self.temporal_resolution * 12
        self.finished = self.current_state["n_cars"] < 0.1

        return self.finished or self.episode_end

    def reward(self):
        """
        Return the adequate reward function (defined in util.py)
        :return:
        """
        return self.reward_function(
            colours=self.colours, current_state=self.current_state
        )

    def document_eval_episode(self):
        """

        Returns:

        """
        self.nl.command(
            "ask cars [document-turtle]"
        )  # ask remaining turtles to document
        self.nl.command("file-close")  # close stream of turtle.csv
        document_episode(self.nl, self.outpath, self.reward_sum, self.uuid)
