import os
import traceback
from datetime import datetime
from torch import nn
from typing import Tuple, List


def count_parameters(model: nn.Module) -> str:
    total_params = sum(p.numel() for p in model.parameters())

    if total_params >= 1e6:
        params_num = f"{total_params / 1e6:.2f}M"
    elif total_params >= 1e3:
        params_num = f"{total_params / 1e3:.2f}K"
    else:
        params_num = str(total_params)

    return params_num


class RLTrainDirs:
    def __init__(
        self,
        root_path,
        log_sub_path,
        debug_log_sub_path,
        ckpt_sub_path,
        results_sub_path,
        records_sub_path,
    ):
        self.root_path = root_path
        self.name = self.get_current_datetime()
        self.log_path = str(os.path.join(root_path, self.name, log_sub_path))
        self.debug_log_path = str(
            os.path.join(root_path, self.name, debug_log_sub_path)
        )
        self.ckpt_path = str(os.path.join(root_path, self.name, ckpt_sub_path))
        self.results_path = str(os.path.join(root_path, self.name, results_sub_path))
        self.records_path = str(os.path.join(root_path, self.name, records_sub_path))
        for path in [
            self.log_path,
            self.debug_log_path,
            self.ckpt_path,
            self.results_path,
            self.records_path,
        ]:
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def get_current_datetime():
        now = datetime.now()
        formatted_datetime = now.strftime("%Y_%m_%d_%H_%M")
        return formatted_datetime


class ExceptionCatcher:
    def __init__(self):
        self.exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            print(f"Exception type: {exc_type}", flush=True)
            print(f"Exception value: {exc_value}", flush=True)
            print("Traceback:", flush=True)
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            # Return False to propagate the exception, True to suppress it
            return True
        return True
