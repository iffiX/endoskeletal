from sim.env import RiseStochasticEnv
from visualize.generate_robot_observations import find_debug_data
import ray

def find_design_data_by_epoch(directory, epoch):
    pattern = re.compile(rf"design_e_{epoch}_it_\d+\.data")
    files = os.listdir(directory)
    for file in files:
        if pattern.match(file):
            with open(os.path.join(directory, file), "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Not found {epoch}")


def find_score_data_by_epoch(directory, epoch):
    pattern = re.compile(rf"score_e_{epoch}_it_\d+\.data")
    files = os.listdir(directory)
    for file in files:
        if pattern.match(file):
            with open(os.path.join(directory, file), "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Not found {epoch}")


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
    debug_logs_dir = (
        "/home/<your-root-rl-dir>/debug-logs"
    )
    replay_epoch = 0 # The epoch you would like to replay

    _, _, debug_log_paths = find_debug_data(debug_logs_dir, replay_epoch)

    tasks = []
    for debug_log_path in debug_log_paths:
        # Submit the task to Ray
        task = process_debug_log.remote(debug_log_path)
        tasks.append(task)

    ray.get(tasks)
