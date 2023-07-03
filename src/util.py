import csv
import json
import os
import re
from glob import glob
from pathlib import Path
from typing import List, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import ParserError
import seaborn as sns
from cmcrameri import cm
from scipy.spatial import distance

sns.set_style("dark")
sns.set_context("paper")

X_LABEL = [
    f"{int(x)}:00 AM" if x < 12 else f"{int(x - [12 if x != 12 else 0])}:00 PM"
    for x in np.arange(8, 22, 2)
]


def occupancy_reward_function(
    colours: List[str], current_state: Dict[str, float], global_mode=False
):
    """
    Rewards occupancy rates between 75% and 90%. Punishes deviations exponentially.
    :param current_state: State dictionary.
    :param colours: Colours of different CPZs.
    :param global_mode: Whether to use the global occupancies or the one of the individual CPZs.
    :return: reward
    """
    reward = 0
    if global_mode:
        cpz_occupancies = [current_state["overall_occupancy"]]
    else:
        cpz_occupancies = [current_state[f"{c}-lot occupancy"] for c in colours]

    for val in cpz_occupancies:
        if 0.75 < val < 0.9:
            reward += 1
        elif val <= 0.75:
            value = 1 - (abs(val - 0.825) / 0.825) ** -1.2
            min_value = 1 - (abs(0 - 0.825) / 0.825) ** -1.2
            max_value = 1 - (abs(0.75 - 0.825) / 0.825) ** -1.2
            max_distance = max_value - min_value
            actual_distance = value - min_value
            reward += actual_distance / max_distance
        elif val >= 0.9:
            value = 1 - (abs(val - 0.825) / 0.825) ** -1.2
            min_value = 1 - (abs(1 - 0.825) / 0.825) ** -1.2
            max_value = 1 - (abs(0.9 - 0.825) / 0.825) ** -1.2
            max_distance = max_value - min_value
            actual_distance = value - min_value
            reward += actual_distance / max_distance
    return reward / len(cpz_occupancies)


def n_cars_reward_function(colours: List[str], current_state: Dict[str, float]):
    """
    Minimizes the number of cars in the simulation.
    :param colours: Colours of different CPZs (only present to be able to use one call in custom_environment.py).
    :param current_state:State dictionary.
    :return: reward
    """
    return optimize_attr(current_state, "n_cars", mode="min")


def social_reward_function(colours: List[str], current_state: Dict[str, float]):
    """
    Maximizes the normalized share of poor cars in the model.
    :param colours: Colours of different CPZs (only present to be able to use one call in custom_environment.py).
    :param current_state:State dictionary.
    :return: reward
    """
    return optimize_attr(current_state, "normalized_share_low")


def speed_reward_function(colours: List[str], current_state: Dict[str, float]):
    """
    Maximizes the average speed of the turtles in the model.
    :param colours: Colours of different CPZs (only present to be able to use one call in custom_environment.py).
    :param current_state:State dictionary.
    :return: reward
    """
    return optimize_attr(current_state, "mean_speed")


def composite_reward_function(colours: List[str], current_state: Dict[str, float]):
    """
    Maximizes 1/2 occupancy_reward_function + 1/4 n_cars_reward_function + 1/4 social_reward_function
    :param colours: Colours of different CPZs (only present to be able to use one call in custom_environment.py).
    :param current_state:State dictionary.
    :return: reward
    """
    return (
        0.5 * occupancy_reward_function(colours, current_state, global_mode=True)
        + 0.25 * n_cars_reward_function(colours, current_state)
        + 0.25 * social_reward_function(colours, current_state)
    )


def glob_outcome_reward_function(colours: List[str], current_state: Dict[str, float]):
    """
    Minimizes the global outcome divergences of the turtles in the model.
    :param colours: Colours of different CPZs (only present to be able to use one call in custom_environment.py).
    :param current_state:State dictionary.
    :return: reward
    """
    return optimize_attr(
        current_state, "global_outcome_divergence", mode="min", power=2
    )


def intergroup_outcome_reward_function(
    colours: List[str], current_state: Dict[str, float]
):
    """
    Minimizes the intergroup (income) outcome divergences of the turtles in the model.
     :param colours: Colours of different CPZs (only present to be able to use one call in custom_environment.py).
     :param current_state:State dictionary.
     :return: reward
    """
    return optimize_attr(
        current_state, "intergroup_outcome_divergence", mode="min", power=2
    )


def composite_outcome_reward_function(
    colours: List[str], current_state: Dict[str, float]
):
    """
    Minimizes the combination of intergroup and global (income) outcome divergences of the turtles in the model.
     :param colours: Colours of different CPZs (only present to be able to use one call in custom_environment.py).
     :param current_state:State dictionary.
     :return: reward
    """
    return optimize_attr(
        current_state, "intergroup_outcome_divergence", mode="min", power=2
    ) * optimize_attr(current_state, "average_outcome", mode="max")


def optimize_attr(
    current_state: Dict[str, float], attr: str, mode="max", power: int = 2
):
    """
    Abstract function to optimize attributes.
    :param mode: either "min" or "max" (default).
    :param current_state: State dictionary.
    :param attr: Attribute in state dictionary to optimize.
    :return: reward-value
    """
    if mode == "min":
        return abs(current_state[attr] - 1) ** power
    else:
        return current_state[attr] ** power


def add_bool_arg(parser, name, default=False):
    """
    Adds boolean arguments to parser by registering both the positive argument and the "no"-argument.
    :param parser:
    :param name: Name of argument.
    :param default:
    :return:
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true")
    group.add_argument("--no-" + name, dest=name, action="store_false")
    parser.set_defaults(**{name: default})


def compute_jenson_shannon(nl, intergroup=False):
    """
    :param nl: NetLogo-Session of environment.
    :param intergroup: Compute intergroup differences
    :return:
    """
    if intergroup:
        outcome_distro = []
        for group in [0, 1, 2]:
            group_average = np.average(nl.report(f"get-outcomes {group}"))
            outcome_distro.append(group_average)
        outcome_distro = np.asarray(outcome_distro)
        outcome_distro = (outcome_distro + 2 * abs(np.min(outcome_distro))) ** 5
    else:
        outcome_distro = np.array(nl.report('get-outcomes "all"'))
        outcome_distro = (outcome_distro + 2 * abs(np.min(outcome_distro))) ** 5
    # Create probability vectors
    outcome_vec = outcome_distro / np.sum(outcome_distro)
    uniform_vec = np.ones(len(outcome_vec)) / len(outcome_vec)
    max_outcome = np.zeros(len(outcome_vec))
    max_outcome[0] = 1.0
    dist = distance.jensenshannon(outcome_vec, uniform_vec)
    max_dist = distance.jensenshannon(max_outcome, uniform_vec)
    # Compare to uniform distribution and return Jensen-Shannon distance (will be squared in reward function)
    return dist / max_dist


def document_episode(nl, path: Path, reward_sum, uuid):
    """
    Create directory for current episode and command NetLogo to save model as csv.
    :param nl: NetLogo-Session of environment.
    :param path: Path of current episode.
    :param reward_sum: Sum of accumulated rewards for episode.
    :param uuid: Unique id to identify process and turtle.csv

    :return:
    """
    path.mkdir(parents=True, exist_ok=True)
    # Get all directories to check which episode this is
    dirs = glob(str(path) + "/E*.pkl")
    current_episode = 1
    if dirs:
        last_episode = max(
            [int(re.findall("E(\d+)", dirs[i])[0]) for i in range(len(dirs))]
        )
        current_episode = last_episode + 1
    episode_path = str(path / f"E{current_episode}_{np.around(reward_sum, 8)}").replace(
        "\\", "/"
    )

    nl.command(f'export-world "{episode_path}.csv"')
    nl.command(f'export-view "{episode_path}.png"')

    convert_to_pickle(path=episode_path, new_path=str(episode_path))
    turtle_csv_path = path / f"turtles_{uuid}.csv"

    if turtle_csv_path.is_file():
        convert_to_pickle(
            path=path / f"turtles_{uuid}", new_path=f"{episode_path}_turtles"
        )


def convert_to_pickle(path: Path, new_path: str):
    """

    :param path:
    :param new_path:
    :return:
    """
    # Save relevant data as pickle to save storage
    if "turtles" in str(path):
        df = pd.read_csv(f"{path}.csv")
    else:
        df = get_data_from_run(f"{path}.csv")
    df.to_pickle(f"{new_path}.pkl", compression="zip")

    # Delete csv
    os.remove(f"{path}.csv")


def label_episodes(path: Path, df: pd.DataFrame, mode: str):
    """
    Identifies worst, median and best episode of run. Renames them and saves plots.
    :param path: Path of current Experiment.
    :param df: DataFrame containing the results.
    :param mode: Usually either "training" or "evaluation".
    :return:
    """
    episode_files = [
        fn for fn in glob(str(path) + "/E*.pkl") if "turtles" not in str(fn)
    ]
    performances = dict()
    performances["max"] = np.around(df.rewards.max(), 8)
    performances["min"] = np.around(df.rewards.min(), 8)
    performances["median"] = np.around(
        df.rewards.sort_values(ignore_index=True)[np.ceil(len(df) / 2) - 1], 8
    )

    print(f"Performances for {mode}:")
    print(performances)

    for metric in performances.keys():
        if performances[metric] == 0.0:
            performances[metric] = 0
        found = False
        for episode in episode_files:
            # Baseline
            if mode not in ["training", "eval"]:
                if str(performances[metric]) == episode.split("_")[-1].split(".pkl")[0]:
                    found = True
            elif str(performances[metric]) in episode:
                found = True
            if found:
                new_path = path / mode / metric
                new_path.mkdir(parents=True, exist_ok=True)
                save_plots(new_path, episode)
                os.rename(
                    episode,
                    str(new_path / f"{mode}_{metric}_{performances[metric]}.pkl"),
                )
                os.rename(
                    episode.replace(".pkl", "_turtles.pkl"),
                    str(
                        new_path / f"{mode}_{metric}_{performances[metric]}_turtles.pkl"
                    ),
                )
                os.rename(
                    episode.replace("pkl", "png"),
                    str(new_path / f"view_{mode}_{metric}_{performances[metric]}.png"),
                )
                episode_files.remove(episode)
                break


def delete_unused_episodes(path: Path):
    """
    Deletes episodes that did not produce either min, median or max performances to save storage.
    :param path: Path of current Experiment
    :return:
    """
    # Get all episodes not moved due to being min, median or max
    episode_files = glob(str(path) + "/E*.*")

    # Remove files of episodes
    for file in episode_files:
        if os.path.exists(file):
            os.remove(file)

    print("Unused Files deleted!")


def create_turtle_df(episode_path: str):
    """

    :param episode_path:
    :return:
    """
    start_i = 0
    end_i = 0
    with open(episode_path, newline="") as csvfile:
        file_reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for i, row in enumerate(file_reader):
            if "TURTLES" in row:
                start_i = i + 1
            if "PATCHES" in row:
                end_i = i - 2
                break
    data_df = pd.read_csv(episode_path, skiprows=start_i, nrows=end_i - start_i)
    # data_df = data_df[data_df.breed == "{breed cars}"]
    data_df.to_csv(str(Path(episode_path).parent / "test.csv"), index=False)

    data_df = data_df[
        [
            "who",
            "income",
            "income-grade",
            "wtp",
            "parking-offender?",
            "distance-parking-target",
            "price-paid",
            "search-time",
            "park-time",
            "park",
        ]
    ]
    data_df.to_csv(str(Path(episode_path).parent / "filtered.csv"), index=False)


def get_data_from_run(episode_path):
    """
    Extracts data for plots from episode.csv saved by NetLogo.
    :param episode_path: Path of current episode.
    :return: DataFrame with data of current episode.
    """
    # Open JSON file containing the indexing information required to extract the information needed for plotting
    with open(Path(__file__).with_name("df_index.json"), "r") as fp:
        INDEX_DICT = json.load(fp=fp)

    with open(episode_path, newline="") as csvfile:
        file_reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for i, row in enumerate(file_reader):
            for key in INDEX_DICT.keys():
                if INDEX_DICT[key]["title"] in row:
                    INDEX_DICT[key]["i"] = i

    data_df = pd.read_csv(
        episode_path, skiprows=INDEX_DICT["fee"]["i"] + 11, nrows=21601
    )
    data_df = data_df.rename(
        columns={
            "y": "yellow_lot_fee",
            "y.1": "teal_lot_fee",
            "y.2": "green_lot_fee",
            "y.3": "blue_lot_fee",
        }
    )
    data_df = data_df[
        ["x", "yellow_lot_fee", "green_lot_fee", "teal_lot_fee", "blue_lot_fee"]
    ]

    data_df.x = data_df.x / 1800
    del INDEX_DICT["fee"]

    i = 0
    # Catch exceptions for different versions of NetLogo model run
    while i < len(INDEX_DICT.keys()):
        key = sorted(INDEX_DICT)[i]
        try:
            temp_df = pd.read_csv(
                episode_path,
                skiprows=INDEX_DICT[key]["i"] + INDEX_DICT[key]["offset"],
                nrows=21601,
            )
            for j, col in enumerate(INDEX_DICT[key]["cols"]):
                temp_df = temp_df.rename(columns={f"y.{j}" if j > 0 else "y": col})
            temp_df = temp_df[INDEX_DICT[key]["cols"]]
            data_df = data_df.join(temp_df)
            i += 1
        except KeyError:
            INDEX_DICT[key]["offset"] += 1
        except ParserError:
            print("No group fees recorded")
            i += 1

    return data_df


def save_plots(outpath: Path, episode_path: str):
    """
    Calls all plot functions for given episode.
    :param outpath: Path to save plots.
    :param episode_path: Path of current episode.
    :return:
    """
    try:
        data_df = pd.read_pickle(episode_path, compression="zip")
    except FileNotFoundError:
        data_df = get_data_from_run(episode_path)

    for func in [
        plot_fees,
        plot_group_fees,
        plot_occup,
        plot_social,
        plot_n_cars,
        plot_speed,
        plot_income_stats,
        plot_share_yellow,
        plot_share_parked,
        plot_share_vanished,
        plot_outcomes,
    ]:
        func(data_df, outpath)

    turtle_df_path = episode_path.replace(".pkl", "_turtles.pkl")
    try:
        turtle_df = pd.read_pickle(turtle_df_path, compression="zip")
        turtle_df.loc[:, "space-type"] = turtle_df.loc[:, "space-type"].replace(
            {"curb": 0, "garage": 1}
        )

        turtle_plot_path = outpath / "turtle_plots"
        turtle_plot_path.mkdir(parents=True, exist_ok=True)
        for group in ["income-group", "parking-strategy", "purpose"]:
            plot_space_attributes_grouped(
                turtle_df=turtle_df, group=group, outpath=turtle_plot_path
            )
            plot_average_attribute_grouped(
                turtle_df=turtle_df,
                group=group,
                attribute="price-paid",
                outpath=turtle_plot_path,
            )
            plot_average_attribute_grouped(
                turtle_df=turtle_df,
                group=group,
                attribute="outcome",
                outpath=turtle_plot_path,
            )
            plot_space_type_grouped(
                turtle_df=turtle_df, group=group, outpath=turtle_plot_path
            )

    except FileNotFoundError:
        print("No turtle DataFrame found!")


def plot_fees(data_df, outpath):
    """
    Plot fees for CPZs over run of episode.
    :param data_df: DataFrame with data from current episode.
    :param outpath: Path to save plot.
    :return:
    """
    color_list = [
        cm.imola_r(0),
        cm.imola_r(1.0 * 1 / 3),
        cm.imola_r(1.0 * 2 / 3),
        cm.imola_r(1.0),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300)
    ax.plot(
        data_df.x,
        data_df.yellow_lot_fee,
        linewidth=4,
        color=color_list[0],
        linestyle="solid",
    )
    ax.plot(
        data_df.x,
        data_df.green_lot_fee,
        linewidth=4,
        color=color_list[1],
        linestyle="dashed",
    )
    ax.plot(
        data_df.x,
        data_df.teal_lot_fee,
        linewidth=4,
        color=color_list[2],
        linestyle="dashed",
    )
    ax.plot(
        data_df.x,
        data_df.blue_lot_fee,
        linewidth=4,
        color=color_list[3],
        linestyle="dashed",
    )

    ax.set_ylim(bottom=0, top=10.1)

    ax.set_ylabel("Hourly Fee in €", fontsize=30)
    ax.set_xlabel("", fontsize=30)
    ax.grid(True)
    ax.tick_params(axis="both", labelsize=25)
    ax.set_xlabel("Time of Day", fontsize=30)
    ax.set_xticks(ticks=np.arange(0, max(data_df["x"]) + 1, 2))
    ax.set_xticklabels(labels=X_LABEL)

    create_colourbar(fig)
    fig.savefig(str(outpath / "fees.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_group_fees(data_df, outpath):
    """
    PLot shares of different income classes over run of episode.
    :param data_df: DataFrame with data from current episode.
    :param outpath: Path to save plot.
    :return:
    """
    for color in ["yellow", "green", "teal", "blue"]:
        if f"fee_{color}_middle" in data_df.columns:
            # Save plot with three variants of legend location
            for loc in ["lower right", "right", "upper right"]:
                fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300)
                color_list = [cm.bamako(0), cm.bamako(1.0 * 1 / 2), cm.bamako(1.0)]
                ax.plot(
                    data_df.x,
                    data_df[f"fee_{color}_low"],
                    label="Low Income",
                    linewidth=3,
                    color=color_list[0],
                )
                ax.plot(
                    data_df.x,
                    data_df[f"fee_{color}_middle"],
                    label="Middle Income",
                    linewidth=3,
                    color=color_list[1],
                    linestyle="dashed",
                )
                ax.plot(
                    data_df.x,
                    data_df[f"fee_{color}_high"],
                    label="High Income",
                    linewidth=3,
                    color=color_list[2],
                    linestyle="dashed",
                )
                ax.set_ylim(bottom=0, top=10.1)

                ax.set_ylabel("Hourly Fee in €", fontsize=30)
                ax.grid(True)
                ax.tick_params(axis="both", labelsize=25)
                ax.set_xlabel("Time of Day", fontsize=30)
                ax.set_xticks(ticks=np.arange(0, max(data_df["x"]) + 1, 2))
                ax.set_xticklabels(labels=X_LABEL)
                ax.legend(fontsize=25, loc=loc)

                fig.savefig(
                    str(outpath / f"{color}_group_fees_{loc}.pdf"), bbox_inches="tight"
                )
                plt.close(fig)


def plot_occup(data_df, outpath):
    """
    Plot occupation levels of different CPZs over run of episode.
    :param data_df: DataFrame with data from current episode.
    :param outpath: Path to save plot.
    :return:
    """
    # Save plot with three variants of legend location
    for loc in ["lower right", "right", "upper right"]:
        fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300)

        color_list = [
            cm.imola_r(0),
            cm.imola_r(1.0 * 1 / 3),
            cm.imola_r(1.0 * 2 / 3),
            cm.imola_r(1.0),
        ]
        ax.plot(
            data_df.x, data_df.yellow_lot_occup / 100, linewidth=2, color=color_list[0]
        )
        ax.plot(
            data_df.x, data_df.green_lot_occup / 100, linewidth=2, color=color_list[1]
        )
        ax.plot(
            data_df.x, data_df.teal_lot_occup / 100, linewidth=2, color=color_list[2]
        )
        ax.plot(
            data_df.x, data_df.blue_lot_occup / 100, linewidth=2, color=color_list[3]
        )
        ax.plot(
            data_df.x,
            data_df.garages_occup / 100,
            label="Garage(s)",
            linewidth=2,
            color="black",
        )
        ax.plot(
            data_df.x,
            data_df.overall_occup / 100,
            label="Kerbside Parking Overall",
            linewidth=4,
            color=cm.berlin(1.0),
            linestyle=(0, (1, 5)),
        ) if "composite" in str(outpath).lower() else None
        ax.plot(
            data_df.x,
            [0.75] * len(data_df.x),
            linewidth=2,
            color="red",
            linestyle="dashed",
        )
        ax.plot(
            data_df.x,
            [0.90] * len(data_df.x),
            linewidth=2,
            color="red",
            linestyle="dashed",
        )
        ax.set_ylim(bottom=0, top=1.01)

        ax.set_ylabel("Utilized Capacity", fontsize=30)
        ax.grid(True)
        ax.tick_params(axis="both", labelsize=25)
        ax.set_xlabel("Time of Day", fontsize=30)
        ax.set_xticks(ticks=np.arange(0, max(data_df["x"]) + 1, 2))
        ax.set_xticklabels(labels=X_LABEL)
        create_colourbar(fig)
        ax.legend(fontsize=25, loc=loc)

        fig.savefig(str(outpath / f"occupancy_{loc}.pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_social(data_df, outpath):
    """
    PLot shares of different income classes over run of episode.
    :param data_df: DataFrame with data from current episode.
    :param outpath: Path to save plot.
    :return:
    """
    # Save plot with three variants of legend location
    for loc in ["lower right", "right", "upper right"]:
        fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300)
        color_list = [cm.bamako(0), cm.bamako(1.0 * 1 / 2), cm.bamako(1.0)]
        ax.plot(
            data_df.x,
            data_df.low_income / 100,
            label="Low Income",
            linewidth=3,
            color=color_list[0],
        )
        ax.plot(
            data_df.x,
            data_df.middle_income / 100,
            label="Middle Income",
            linewidth=3,
            color=color_list[1],
        )
        ax.plot(
            data_df.x,
            data_df.high_income / 100,
            label="High Income",
            linewidth=3,
            color=color_list[2],
        )
        ax.set_ylim(bottom=0, top=1.01)

        ax.set_ylabel("Share of Cars per Income Class", fontsize=30)
        ax.grid(True)
        ax.tick_params(axis="both", labelsize=25)
        ax.set_xlabel("Time of Day", fontsize=30)
        ax.set_xticks(ticks=np.arange(0, max(data_df["x"]) + 1, 2))
        ax.set_xticklabels(labels=X_LABEL)
        ax.legend(fontsize=25, loc=loc)

        fig.savefig(str(outpath / f"social_{loc}.pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_speed(data_df, outpath):
    """
    Plot average speed over run of episode.
    :param data_df: DataFrame with data from current episode.
    :param outpath: Path to save plot.
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300)
    ax.plot(data_df.x, data_df.average_speed, linewidth=3, color=cm.bamako(0))
    ax.plot(
        data_df.x,
        data_df.average_speed.rolling(50).mean(),
        linewidth=3,
        color=cm.bamako(1.0),
    )

    ax.set_ylim(bottom=0, top=1.01)

    ax.set_ylabel("Average Normalized Speed", fontsize=30)
    ax.grid(True)
    ax.tick_params(axis="both", labelsize=25)
    ax.set_xlabel("Time of Day", fontsize=30)
    ax.set_xticks(ticks=np.arange(0, max(data_df["x"]) + 1, 2))
    ax.set_xticklabels(labels=X_LABEL)

    fig.savefig(str(outpath / "speed.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_n_cars(data_df, outpath):
    """
    Plot number of cars over run of episode.
    :param data_df: DataFrame with data from current episode.
    :param outpath: Path to save plot.
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300)
    ax.plot(data_df.x, data_df.cars_overall / 100, linewidth=3, color=cm.bamako(0))
    ax.set_ylim(bottom=0, top=1.01)

    ax.set_ylabel("Share of Initially Spawned Cars", fontsize=30)
    ax.grid(True)
    ax.tick_params(axis="both", labelsize=25)
    ax.set_xlabel("Time of Day", fontsize=30)
    ax.set_xticks(ticks=np.arange(0, max(data_df["x"]) + 1, 2))
    ax.set_xticklabels(labels=X_LABEL)

    fig.savefig(str(outpath / "n_cars.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_income_stats(data_df, outpath):
    """
    Plot mean, median and std. of income distribution of run of episode.
    :param data_df: DataFrame with data from current episode.
    :param outpath: Path to save plot.
    :return:
    """
    # Save plot with three variants of legend location
    for loc in ["lower right", "right", "upper right"]:
        fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300)
        color_list = [cm.berlin(0), cm.berlin(1.0 * 1 / 2), cm.berlin(1.0)]
        ax.plot(
            data_df.x, data_df["mean"], label="Mean", linewidth=3, color=color_list[0]
        )
        ax.plot(
            data_df.x,
            data_df["median"],
            label="Median",
            linewidth=3,
            color=color_list[1],
        )
        ax.plot(
            data_df.x,
            data_df["std"],
            label="Standard Deviation",
            linewidth=3,
            color=color_list[2],
        )
        ax.set_ylim(bottom=0, top=max(data_df[["mean", "median", "std"]].max()) + 1)

        ax.set_ylabel("Income in €", fontsize=30)
        ax.grid(True)
        ax.tick_params(axis="both", labelsize=25)
        ax.set_xlabel("Time of Day", fontsize=30)
        ax.set_xticks(ticks=np.arange(0, max(data_df["x"]) + 1, 2))
        ax.set_xticklabels(labels=X_LABEL)
        ax.legend(fontsize=25, loc=loc)

        fig.savefig(str(outpath / f"income_stats_{loc}.pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_share_yellow(data_df, outpath):
    """
    Plot share of different income classes on yellow CPZ.
    :param data_df: DataFrame with data from current episode.
    :param outpath: Path to save plot.
    :return:
    """
    # Save plot with three variants of legend location
    for loc in ["lower right", "right", "upper right"]:
        fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300)
        color_list = [cm.bamako(0), cm.bamako(1.0 * 1 / 2), cm.bamako(1.0)]
        ax.plot(
            data_df.x,
            data_df.share_y_low / 100,
            label="Low Income",
            linewidth=3,
            color=color_list[0],
        )
        ax.plot(
            data_df.x,
            data_df.share_y_middle / 100,
            label="Middle Income",
            linewidth=3,
            color=color_list[1],
        )
        ax.plot(
            data_df.x,
            data_df.share_y_high / 100,
            label="High Income",
            linewidth=3,
            color=color_list[2],
        )
        ax.set_ylim(bottom=0, top=1.01)

        ax.set_ylabel("Share of Cars in Yellow CPZ", fontsize=30)
        ax.grid(True)
        ax.tick_params(axis="both", labelsize=25)
        ax.set_xlabel("Time of Day", fontsize=30)
        ax.set_xticks(ticks=np.arange(0, max(data_df["x"]) + 1, 2))
        ax.set_xticklabels(labels=X_LABEL)
        ax.legend(fontsize=25, loc=loc)

        fig.savefig(str(outpath / f"share_yellow_{loc}.pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_share_parked(data_df, outpath):
    """
    Plot share of parked cars per income class.
    :param data_df: DataFrame with data from current episode.
    :param outpath: Path to save plot.
    :return:
    """
    # Save plot with three variants of legend location
    for loc in ["lower right", "right", "upper right"]:
        fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300)
        color_list = [cm.bamako(0), cm.bamako(1.0 * 1 / 2), cm.bamako(1.0)]
        ax.plot(
            data_df.x,
            data_df.share_p_low / 100,
            label="Low Income",
            linewidth=3,
            color=color_list[0],
        )
        ax.plot(
            data_df.x,
            data_df.share_p_middle / 100,
            label="Middle Income",
            linewidth=3,
            color=color_list[1],
        )
        ax.plot(
            data_df.x,
            data_df.share_p_high / 100,
            label="High Income",
            linewidth=3,
            color=color_list[2],
        )
        ax.set_ylim(bottom=0, top=1.01)

        ax.set_ylabel("Share of Cars Finding Parking", fontsize=30)
        ax.grid(True)
        ax.tick_params(axis="both", labelsize=25)
        ax.set_xlabel("Time of Day", fontsize=30)
        ax.set_xticks(ticks=np.arange(0, max(data_df["x"]) + 1, 2))
        ax.set_xticklabels(labels=X_LABEL)
        ax.legend(fontsize=25, loc=loc)

        fig.savefig(str(outpath / f"share_parked_{loc}.pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_share_vanished(data_df, outpath):
    """
    Plot share of vanished cars per income class.
    :param data_df: DataFrame with data from current episode.
    :param outpath: Path to save plot.
    :return:
    """
    # Save plot with three variants of legend location
    for loc in ["lower right", "right", "upper right"]:
        fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300)
        color_list = [cm.bamako(0), cm.bamako(1.0 * 1 / 2), cm.bamako(1.0)]
        ax.plot(
            data_df.x,
            data_df.share_v_low,
            label="Low Income",
            linewidth=3,
            color=color_list[0],
        )
        ax.plot(
            data_df.x,
            data_df.share_v_middle,
            label="Middle Income",
            linewidth=3,
            color=color_list[1],
        )
        ax.plot(
            data_df.x,
            data_df.share_v_high,
            label="High Income",
            linewidth=3,
            color=color_list[2],
        )
        #ax.set_ylim(bottom=0, top=1.01)

        ax.set_ylabel(" Share of Cars Vanished", fontsize=30)
        ax.grid(True)
        ax.tick_params(axis="both", labelsize=25)
        ax.set_xlabel("Time of Day", fontsize=30)
        ax.set_xticks(ticks=np.arange(0, max(data_df["x"]) + 1, 2))
        ax.set_xticklabels(labels=X_LABEL)
        ax.legend(fontsize=25, loc=loc)

        fig.savefig(str(outpath / f"share_vanished_{loc}.pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_outcomes(data_df, outpath):
    """
    PLot shares of different income classes over run of episode.
    :param data_df: DataFrame with data from current episode.
    :param outpath: Path to save plot.
    :return:
    """
    # Save plot with three variants of legend location
    for loc in ["lower right", "right", "upper right"]:
        fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300)
        color_list = [cm.bamako(0), cm.bamako(1.0 * 1 / 2), cm.bamako(1.0)]
        ax.plot(
            data_df.x,
            data_df.low_outcome,
            label="Low Income",
            linewidth=3,
            color=color_list[0],
        )
        ax.plot(
            data_df.x,
            data_df.middle_outcome,
            label="Middle Income",
            linewidth=3,
            color=color_list[1],
        )
        ax.plot(
            data_df.x,
            data_df.high_outcome,
            label="High Income",
            linewidth=3,
            color=color_list[2],
        )
        ax.plot(
            data_df.x,
            data_df.global_outcome,
            label="Global",
            linewidth=3,
            color="black",
        )
        # ax.set_ylim(bottom=0, top=1.01)

        ax.set_ylabel("Outcome per Income Class", fontsize=30)
        ax.grid(True)
        ax.tick_params(axis="both", labelsize=25)
        ax.set_xlabel("Time of Day", fontsize=30)
        ax.set_xticks(ticks=np.arange(0, max(data_df["x"]) + 1, 2))
        ax.set_xticklabels(labels=X_LABEL)
        ax.legend(fontsize=25, loc=loc)

        fig.savefig(str(outpath / f"outcomes_{loc}.pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_space_attributes_grouped(turtle_df, group: str, outpath: Path):
    """
    Plots attributes of found parking spaces (access, search, egress) across different groups.
    :param turtle_df: DataFrame with data of turtles from current episode.
    :param group: Group to use for grouping the data.
    :param outpath: Path to save plot.
    :return:
    """
    turtle_df = turtle_df.loc[
        (turtle_df["wants-to-park"]) & (turtle_df["reinitialize?"])
    ]
    x = np.arange(3)  # the label locations
    width = 0.25 if group == "income-group" else 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(1, 1, figsize=(25, 8), dpi=300)

    grouped_df = turtle_df.groupby(group).mean()
    if group == "income-group":
        labels = ["Low Income", "Middle Income", "High Income"]
        color_list = [cm.bamako(0), cm.bamako(1.0 * 1 / 2), cm.bamako(1.0)]
    else:
        if group == 'parking-strategy':
            labels = ["Close to Goal", "Garage", "En Route", "Other"]
        else:
            labels = ["Job / Education", "Doctor", "Meeting a Friend", "Shopping"]
        color_list = []
        n = len(turtle_df[group].unique())
        for step in range(n):
            color_list.append(cm.batlow(step / n))
    for group_name, label, color in zip(
        sorted(turtle_df[group].unique()), labels, color_list
    ):
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            np.round(
                grouped_df.loc[group_name, ["egress", "access", "search-time"]]
                .values.flatten()
                .tolist(),
                2,
            ),
            width,
            label=label,
            color=color,
        )
        ax.bar_label(rects, padding=3, fontsize=15)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Duration in Minutes", fontsize=30)
    if group == "income-group":
        ax.set_xticks(ticks=(x + width))
    else:
        ax.set_xticks(ticks=(x + 1.5 * width))
    ax.set_xticklabels(["Egress", "Access", "Search"], fontsize=30)
    ax.tick_params(axis="both", labelsize=25)

    ax.legend(loc="best", fontsize=25)
    fig.savefig(
        str(outpath / f"space_attributes_{group}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_average_attribute_grouped(
    turtle_df, group: str, attribute: str, outpath: Path
):
    """
    Plots average attribute (paid-fee or outcome) across different groups.
    :param turtle_df: DataFrame with data of turtles from current episode.
    :param group: Group to use for grouping the data.
    :param outpath: Path to save plot.
    :return:
    """
    x = np.arange(len(turtle_df[group].unique()))  # the label locations

    fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300)

    if attribute == "price-paid":
        turtle_df = turtle_df.loc[
            (turtle_df["wants-to-park"])
            & (turtle_df["reinitialize?"])
            & (~turtle_df["parking-offender?"])
        ]
    else:
        turtle_df = turtle_df.loc[
            (turtle_df["wants-to-park"]) & (turtle_df["outcome"] != -99)
        ]

    grouped_df = turtle_df.groupby(group).mean()

    if group == "income-group":
        labels = ["Low Income", "Middle Income", "High Income"]
        color_list = [cm.bamako(0), cm.bamako(1.0 * 1 / 2), cm.bamako(1.0)]
    else:
        if group == 'parking-strategy':
            labels = ["Close to Goal", "Garage", "En Route", "Other"]
        else:
            labels = ["Job / Education", "Doctor", "Meeting a Friend", "Shopping"]
        color_list = []
        n = len(turtle_df[group].unique())
        for step in range(n):
            color_list.append(cm.batlow(step / n))

    rect = ax.bar(x, np.round(grouped_df[attribute], 2), color=color_list)
    ax.bar_label(rect, padding=3, fontsize=15)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(
        "Average Fee Paid in €" if attribute == "price-paid" else "Average Utility",
        fontsize=30,
    )
    ax.set_xticks(ticks=(x))
    ax.set_xticklabels(labels, fontsize=30)
    ax.tick_params(axis="both", labelsize=25)

    # ax.legend(loc='upper left', ncols=3)
    # ax.set_ylim(0, 250)

    fig.savefig(
        str(outpath / f"{attribute}_{group}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_space_type_grouped(turtle_df, group: str, outpath: Path):
    """

    :param turtle_df:
    :param group:
    :param outpath:
    :return:
    """
    x = np.arange(len(turtle_df[group].unique()))  # the label locations

    fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300)

    turtle_df = turtle_df.loc[
        (turtle_df["wants-to-park"])
        & (turtle_df["reinitialize?"])
        & (~turtle_df["parking-offender?"])
    ]

    grouped_df = turtle_df.groupby(group).mean()

    rect = ax.bar(
        x, np.round(grouped_df["space-type"], 2), color=cm.acton(0), label="Garage"
    )
    ax.bar(
        x,
        1 - np.round(grouped_df["space-type"], 2),
        color=cm.acton(0.5),
        bottom=grouped_df["space-type"],
        label="Curb",
    )

    ax.bar_label(rect, padding=3, fontsize=15)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Average Fee Paid in €', fontsize=30)
    ax.set_xticks(ticks=(x))

    if group == "income-group":
        labels = ["Low Income", "Middle Income", "High Income"]
    else:
        if group == 'parking-strategy':
            labels = ["Close to Goal", "Garage", "En Route", "Other"]
        else:
            labels = ["Job / Education", "Doctor", "Meeting a Friend", "Shopping"]

    ax.set_xticklabels(labels, fontsize=30)
    ax.tick_params(axis="both", labelsize=25)

    ax.legend(loc="best", fontsize=25)
    # ax.set_ylim(0, 250)

    fig.savefig(
        str(outpath / f"space_type_{group}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def create_colourbar(fig):
    """
    Draws colourbar with colour of different CPZs on given figure.
    :param fig: Figure to draw colourbar on.
    :return:
    """
    cmap = cm.imola

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.01)
    cb_ax = fig.add_axes([0.8, 0.1, 0.015, 0.8])

    bounds = [0, 1, 2, 3, 4]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cb_ax,
        orientation="vertical",
    )

    cbar.set_ticks([])
    cbar.ax.set_ylabel(
        r"$\Leftarrow$ Distance of CPZ to City Center", fontsize=25, loc="top"
    )
