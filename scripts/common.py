import ray
import torch as t
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from io import BytesIO
from tqdm import tqdm
from typing import List
from sim.builder import SimBuilder
from sim.terrain import *
from utils.train_utils import RLTrainDirs, ExceptionCatcher

from rl.algorithms import PPO
from model.rl.robot import RobotActor, RobotCritic
from sim.env import RiseStochasticEnv


@ray.remote
def build_robot(
    epoch: int,
    robot_id: int,
    logits: np.ndarray,
    rl_dirs: RLTrainDirs,
    max_torque: float,
    plot_robots: bool,
    voxel_size: float,
):
    with ExceptionCatcher() as _:
        builder = SimBuilder(
            voxel_size=voxel_size,
            valid_min_rigid_ratio=0.2,
            valid_min_joint_num=2,
            valid_max_connected_components=1,
            min_rigid_volume=100,
            hinge_torque=max_torque,
        )
        rsc = builder.build(
            logits,
            f"bot_e_{epoch}_ro_{robot_id}",
            rl_dirs.results_path,
            rl_dirs.records_path,
            save_history=True,
            save_h5_history=True,
            print_summary=True,
        )

        fig = plt.figure(figsize=(18, 6))
        if plot_robots:
            axs = fig.subplots(1, 3, subplot_kw={"projection": "3d"})
            builder.visualize(
                is_not_empty_ax=axs[0],
                is_rigid_ax=axs[1],
                segmentation_ax=axs[2],
            )
        buf = BytesIO()
        fig.savefig(buf)
        plt.close()
        buf.seek(0)
        return rsc, buf


@ray.remote(num_gpus=0.9)
class RobotSimulationCollector:
    def __init__(
        self,
        rank: int,
        rl_dirs: RLTrainDirs,
        actor: RobotActor,
        critic: RobotCritic,
        min_actions: int,
        voxel_size: float,
    ):
        with ExceptionCatcher() as _:
            self.rank = rank
            self.rl_dirs = rl_dirs
            self.actor = actor.to("cuda:0")
            self.critic = critic.to("cuda:0")
            self.min_actions = min_actions
            self.voxel_size = voxel_size
            self.ppo = PPO(actor, critic, optim.AdamW, nn.MSELoss())

    def update_parameters(self, actor_state_dict, critic_state_dict):
        with ExceptionCatcher() as _:
            self.actor.load_state_dict(actor_state_dict)
            self.critic.load_state_dict(critic_state_dict)
            print(f"Collector {self.rank} parameters updated")
            return self.rank

    def collect(
        self,
        epoch,
        robot_ids: List[int],
        env_configs: List[str],
        robot_configs: List[str],
    ):
        with ExceptionCatcher() as _:
            if len(robot_configs) == 0:
                return []
            if len({len(robot_ids), len(env_configs), len(robot_configs)}) != 1:
                raise ValueError(
                    f"id, env config, robot config sizes "
                    f"[{len(robot_ids)}, {len(env_configs)}, {len(robot_configs)}] "
                    f"doesn't match"
                )
            env = RiseStochasticEnv(
                self.ppo,
                [0],
                env_configs,
                robot_configs,
                angles=[-1.4, -0.7, 0, 0.7, 1.4],
                debug_log_name=f"{self.rl_dirs.debug_log_path}"
                f"/debug_e_{epoch}_ra_{self.rank}.log",
            )
            result, observation, action, reward_state = env.run(epoch)

            episodes = []
            for robot_idx, success in tqdm(enumerate(result), total=len(result)):
                if success and len(observation[robot_idx]) > self.min_actions:
                    episode = []
                    episode_rewards = []
                    episode_costs = []
                    obs = observation[robot_idx]
                    act = action[robot_idx]
                    last_a = None

                    start_com = obs[0]["com"]
                    end_com = obs[-1]["com"]

                    for i, (o, a) in enumerate(zip(obs, act)):
                        com = o["com"]
                        current_movement = com - start_com
                        reward_dir = current_movement.squeeze()
                        reward_dir[2] = 0
                        reward_norm = t.norm(reward_dir)

                        if reward_norm > 0:
                            reward_dir /= t.norm(reward_dir)
                        else:
                            reward_dir = t.tensor([1.0, 0.0, 0.0]).to(reward_dir.device)
                        reward = 0.0
                        if i < len(obs) - 1:
                            next_com = obs[i + 1]["com"]
                            next_movement = (next_com - com).squeeze()
                            reward = t.dot(next_movement, reward_dir) / self.voxel_size
                            reward = max(reward, -0.1)
                            if t.norm(next_movement) < 1e-3:
                                reward = -10.0

                        total = len(a[0])
                        changed = 0
                        if last_a is None:
                            for sub_a in a[0]:
                                if sub_a != 1:
                                    changed += 1
                        else:
                            for last_sub_a, sub_a in zip(last_a, a[0]):
                                if sub_a != last_sub_a:
                                    changed += 1
                        last_a = a[0]
                        action_cost = changed / total

                        episode_rewards.append(reward)
                        episode_costs.append(action_cost)
                        episode.append(
                            {
                                "state": o,
                                "action": {"action": a[0]},
                                "log_prob": a[1],
                                "reward": reward,
                                "terminal": i == len(obs) - 1,
                            }
                        )

                    start_com = start_com.cpu().numpy()
                    end_com = end_com.cpu().numpy()
                    movement = (end_com - start_com).squeeze()
                    plane_distance = float(np.linalg.norm(movement[:2]))

                    episodes.append(
                        {
                            "id": robot_ids[robot_idx],
                            "reward": np.sum(episode_rewards),
                            "cost": np.sum(episode_costs),
                            "distance": plane_distance,
                            "episode": episode,
                        }
                    )
                else:
                    episodes.append(
                        {
                            "id": robot_ids[robot_idx],
                            "reward": 0,
                            "cost": 0,
                            "distance": 0,
                            "episode": [],
                        }
                    )
            t.cuda.empty_cache()
            return episodes
