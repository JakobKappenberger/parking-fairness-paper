from src.experiment import Experiment


def test_simulation():
    """

    :return:
    """
    # Standard Config
    args = init_args()
    experiment = Experiment(**args, args=args)
    experiment.run()
    args = init_args(adjust_free=False)
    second_experiment = Experiment(**args, args=args)
    second_experiment.run()
    args = init_args(adjust_free=True, group_pricing=True)
    third_experiment = Experiment(**args, args=args)
    third_experiment.run()
    args = init_args(adjust_free=False, group_pricing=True)
    fourth_experiment = Experiment(**args, args=args)
    fourth_experiment.run()


def init_args(adjust_free=True, group_pricing=False):
    return {
        "agent": "C:\\Users\\Jakob\\Projekte\\parking-fairness-paper\\scripts\\ppo_agent_local.json",
        "num_episodes": 1,
        "num_parallel": 2,
        "document": True,
        "adjust_free": adjust_free,
        "group_pricing": group_pricing,
        "eval": True,
        "model_size": "training",
        "gui": False,
        "test": True,
    }
