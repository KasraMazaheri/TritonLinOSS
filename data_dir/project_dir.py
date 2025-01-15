from pathlib import Path


def get_linoss_directory():
    return Path(__file__).parent.parent.resolve()