import os
import shlex
import subprocess
import sys
from typing import List

import hydra
from flatten_dict import flatten
from omegaconf import DictConfig


def run_subprocess(command: str):
    print(f"Running command: \n\n{command}\n")
    process = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=2,
    )
    with process.stdout as p_out:
        for line in iter(p_out.readline, b""):  # b'\n'-separated lines
            print(line.decode().strip())

    process.wait()  # to fetch returncode
    return process.returncode


def get_all_params_overrides(cfg: DictConfig) -> List[str]:
    config_flat = flatten(cfg)
    params_overrides = [
        ".".join(param_keys) + ":" + str(param_value)
        for param_keys, param_value in config_flat.items()
    ]
    return params_overrides


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    pipeline = cfg.pipeline if "pipeline" in cfg.keys() else "__default__"
    params_overrides = get_all_params_overrides(cfg)
    kedro_bin = os.path.join(os.path.split(sys.executable)[0], "kedro")
    command = " ".join(
        [
            kedro_bin,
            "run",
            f"--pipeline={pipeline}",
            f'--params="{",".join(params_overrides)}"',
        ]
    )

    returncode = run_subprocess(command)

    if returncode:
        raise Exception


if __name__ == "__main__":
    main()
