import json
import platform
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import shutil


import pandas as pd
import numpy as np
import pynetlogo
from tqdm import trange

from src.util import (
    add_bool_arg,
    document_episode,
    label_episodes,
    delete_unused_episodes,
)

COLOURS = ["yellow", "green", "teal", "blue"]


def run_baseline(
    num_episodes: int,
    model_size: str = "evaluation",
    nl_path: str = None,
    gui: bool = False,
    static: bool = False,
    zip: bool = False
):
    """
    Runs baseline experiments and save results.
    :param num_episodes: Number of episodes to run.
    :param model_size: Model size to run experiments with, either "training" or "evaluation".
    :param nl_path: Path to NetLogo Installation (for Linux users)
    :param gui: Whether NetLogo UI is shown during episodes.
    :param static: Use static baseline.
    :param zip: Whether directory should be zipped after running the script.
    :return:
    """

    timestamp = datetime.now().strftime("%y%m-%d-%H%M")
    outpath = (
        Path(".").absolute().parent
        / f"results/baseline {'static' if static else 'dynamic'}"
        / timestamp
    )
    # Create directory
    outpath.mkdir(parents=True, exist_ok=True)
    # Connect to NetLogo
    if platform.system() == "Linux":
        nl = pynetlogo.NetLogoLink(gui=gui, netlogo_home=nl_path)
    else:
        nl = pynetlogo.NetLogoLink(gui=gui)
    nl.load_model("../src/Model.nlogo")
    # Load model parameters
    with open("../src/model_config.json", "r") as fp:
        model_config = json.load(fp=fp)

    print(f"Configuring model size for {model_size}")
    max_x_cor = model_config[model_size]["max_x_cor"]
    max_y_cor = model_config[model_size]["max_y_cor"]
    nl.command(f"resize-world {-max_x_cor} {max_x_cor} {-max_y_cor} {max_y_cor}")
    nl.command(f'set num-cars-mean {model_config[model_size]["num_cars_mean"]}')
    nl.command(f'set num-garages {model_config[model_size]["num_garages"]}')
    nl.command(
        f'set demand-curve-intercept {model_config[model_size]["demand_curve_intercept"]}'
    )
    nl.command(
        f'set lot-distribution-percentage {model_config[model_size]["lot_distribution_percentage"]}'
    )
    nl.command(
        f'set target-start-occupancy {model_config[model_size]["target_start_occupancy"]}'
    )

    traffic_counter = []
    share_cruising_counter = []
    scores = [0] * num_episodes


    for i in trange(num_episodes):
        episode_cruising = []
        nl.command("set document-turtles true")
        process_id = i
        open(outpath / f"turtles_{process_id}.csv", "w").close
        turtle_file_path = str(outpath / f"turtles_{process_id}.csv").replace("\\", "/")
        nl.command(f'set output-turtle-file-path "{turtle_file_path}"')
        nl.command("setup")
        # nl.command(f'set parking-cars-percentage {p(8) * 100}')
        # Disable rendering of view
        if not gui:
            nl.command("no-display")
        if static:
            # Turn dynamic baseline pricing mechanism off
            nl.command("set dynamic-pricing-baseline false")
        for c in COLOURS:
            nl.command(f"change-fee-free {c}-lot 3.5")
        nl.command("ask one-of cars [record-data]")
        for _ in range(24):
            nl.repeat_command("go", 900)
            episode_cruising.append(nl.report("share-cruising"))
            for c in COLOURS:
                occup = nl.report(f"{c}-lot-current-occup")
                if 0.75 < occup < 0.9:
                    scores[i] += 0.25
        # nl.command(
        #     "ask cars [document-turtle]"
        # )  # ask remaining turtles to document
        nl.command("file-close")  # close stream of turtle.csv
        document_episode(nl=nl, path=outpath, reward_sum=scores[i], uuid=process_id)
        traffic_counter.append(nl.report("traffic-counter"))
        share_cruising_counter.append(np.mean(episode_cruising))
        print(i)
        print(share_cruising_counter)
        print(traffic_counter)

    nl.kill_workspace()
    metrics_df = pd.DataFrame(scores, columns=["rewards"])
    label_episodes(outpath, metrics_df, "standard")
    delete_unused_episodes(outpath)
    print(np.mean(share_cruising_counter))
    print(np.var(share_cruising_counter))
    print(f"Traffic Counter: {np.mean(traffic_counter)}")

    if zip:
        shutil.make_archive(str(outpath), "zip", outpath)
        print("directory zipped")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("episodes", type=int, help="Number of episodes")
    parser.add_argument(
        "-m",
        "--model_size",
        type=str,
        default="evaluation",
        choices=["training", "evaluation"],
        help="Control which model size to use",
    )
    parser.add_argument(
        "-np",
        "--nl_path",
        type=str,
        default=None,
        help="Path to NetLogo directory (for Linux Users)",
    )
    add_bool_arg(parser, "gui", default=False)
    add_bool_arg(parser, "static", default=False)
    add_bool_arg(parser, "zip", default=False)


    args = parser.parse_args()
    print(f" Baseline called with arguments: {vars(args)}")

    run_baseline(
        num_episodes=args.episodes,
        model_size=args.model_size,
        nl_path=args.nl_path,
        gui=args.gui,
        static=args.static,
        zip=args.zip
    )
