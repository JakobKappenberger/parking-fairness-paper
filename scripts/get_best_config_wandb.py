from argparse import ArgumentParser
import os
import json
import wandb


def get_best_run(user: str, project: str, sweep_id: str):
    """

    :param user:
    :param project:
    :param sweep_id:
    :return:
    """
    api = wandb.Api()
    sweep = api.sweep(f"{user}/{project}/{sweep_id}")
    runs = sorted(sweep.runs, key=lambda run: run.summary.get("loss", 0))
    best_run = runs[0]
    print(f"Best run {best_run.name} with {best_run.summary['loss']} loss")

    if best_run.config["entropy_regularization"] < 1e-5:
        best_run.config["entropy_regularization"] = 0.0
        best_run.update()

    agent = {
        "agent": "ppo",
        "saver": {
            "directory": "model-checkpoint",
            "frequency": 1,
            "max_checkpoints": 5,
        },
        "summarizer": {"flush": 10, "directory": "."},
    }

    for key in [
        "batch_size",
        "discount",
        "entropy_regularization",
        "exploration",
        "learning_rate",
    ]:
        agent[key] = best_run.config[key]

    with open(
        os.path.join(
            "./",
            f"ppo_agent_{best_run.config['reward_key']}_{'group-pricing' if best_run.config['group_pricing'] else 'zone_pricing'}.json",
        ),
        "w",
    ) as outfile:
        json.dump(agent, outfile, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--user",
        type=str,
        default="jakob-kappenberger",
        help="Weights & Biases user",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="parking-fairness",
        help="Weights & Biases project",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        help="Sweep ID to get best run from",
    )

    args = parser.parse_args()
    print(f" Baseline called with arguments: {vars(args)}")

    get_best_run(user=args.user, project=args.project, sweep_id=args.sweep_id)
