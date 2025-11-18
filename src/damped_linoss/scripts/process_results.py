import glob
import os
import statistics
from collections import defaultdict
from sys import argv

import yaml


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def make_group_key(hparams, keys):
    """Create a group key string from selected hyperparameter keys."""
    parts = []
    for k in keys:
        # Support nested keys like "training.seed"
        value = hparams
        for subkey in k.split("."):
            value = value.get(subkey, None)
            if value is None:
                break
        parts.append(f"{k}={value}")
    return ", ".join(parts)


def main(exp_root):
    groups = defaultdict(list)
    group_keys_to_use = [
        "model_name",
        "dataset_name",
        "seed",
        "lr",
        "state_dim",
        "hidden_dim",
        "num_blocks",
        "include_time",
    ]

    # Find all results.txt in run_XXX folders under exp_root
    pattern = os.path.join(exp_root, "run_*/results.txt")
    for result_path in glob.glob(pattern, recursive=True):
        dir_path = os.path.dirname(result_path)
        hyper_path = os.path.join(dir_path, "hyperparameters.yaml")

        # Load result.txt
        try:
            with open(result_path, "r") as f:
                lines = f.readlines()
                result_value = float(lines[0])
        except Exception as e:
            print(f"Failed to read {result_path}: {e}")
            continue

        # Load hyperparameters.json
        try:
            with open(hyper_path, "r") as file:
                hyperparameters = yaml.safe_load(file)
        except Exception as e:
            print(f"Failed to read {hyper_path}: {e}")
            continue

        group_key = make_group_key(hyperparameters, group_keys_to_use)

        groups[group_key].append(result_value)

    summaries = []

    for group_key, scores in groups.items():
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
        num = len(scores)
        summaries.append((mean_score, std_score, group_key, num))

    summaries.sort(key=lambda x: x[0])

    for mean_score, std_score, group_key, num in summaries:
        print(f"[{mean_score:.6f} ± {std_score:.6f}] {group_key}")
        print(f"# {num}")
        print()


if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: python process_results.py <experiment_root_folder>")
    else:
        main(argv[1])
