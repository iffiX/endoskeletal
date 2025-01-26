import os
import ray
import pickle
from sim.env import RiseStochasticEnv


def find_debug_data(directory, epoch):
    files = os.listdir(directory)
    for file in files:
        if file == f"debug_e_{epoch}_performance.log":
            with open(os.path.join(directory, file), "rb") as f:
                performance_data = pickle.load(f)
                debug_log_ranks = []
                debug_log_paths = []
                for rank in performance_data["collector_metrics"].keys():
                    debug_log_path = os.path.join(
                        directory, f"debug_e_{epoch}_ra_{rank}.log"
                    )
                    if not os.path.exists(debug_log_path):
                        raise FileNotFoundError(f"{debug_log_path} Not found")
                    debug_log_ranks.append(rank)
                    debug_log_paths.append(str(debug_log_path))
                return performance_data, debug_log_ranks, debug_log_paths

    raise FileNotFoundError(f"Not found {epoch}")

@ray.remote(num_gpus=1)
def process_debug_log(debug_log_path):
    RiseStochasticEnv.replay(
        [0],
        debug_log_path,
        save_record=True,
    )


if __name__ == "__main__":
    ray.init()

    # Example debug log replay
    debug_logs_dir = "data/rl-result/.../debug-logs"
    replay_epoch = 0

    _, _, debug_log_paths = find_debug_data(debug_logs_dir, replay_epoch)

    tasks = []
    for debug_log_path in debug_log_paths:
        # Submit the task to Ray
        task = process_debug_log.remote(debug_log_path)
        tasks.append(task)

    ray.get(tasks)
