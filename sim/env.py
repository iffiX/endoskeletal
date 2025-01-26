import time
import threading
import queue
import torch as t
import numpy as np
import pickle
from rise import Rise, RiseFrame
from typing import List, Any, Union
from rl.algorithms import A2C, PPO
from utils.misc import list_of_dict_to_dict_of_list


def build_voxel_observations(frame: RiseFrame, random_seed: int, voxel_sample_num: int):
    random = np.random.RandomState(random_seed)
    voxel_num = len(frame.voxels())

    voxel_mask = random.choice(
        np.arange(voxel_num), voxel_sample_num, replace=voxel_num < voxel_sample_num
    )

    # shape [voxel_num, 3]
    positions = t.from_numpy(
        frame.voxel_positions()[voxel_mask]
        if voxel_mask is not None
        else frame.voxel_positions()
    )

    # shape [3]
    center_of_mass = t.mean(t.from_numpy(frame.voxel_positions()), dim=0)

    # shape [voxel_num, 3]
    velocities = t.from_numpy(
        frame.voxel_linear_velocities()[voxel_mask]
        if voxel_mask is not None
        else frame.voxel_linear_velocities()
    )

    # shape [voxel_num, 1]
    pressures = t.mean(
        t.from_numpy(
            frame.voxel_poissons_strains()[voxel_mask]
            if voxel_mask is not None
            else frame.voxel_poissons_strains()
        ),
        dim=1,
        keepdim=True,
    )
    return positions, center_of_mass, velocities, pressures


def build_kinematic_graph(frame: RiseFrame):
    rigid_body_mass = frame.rigid_body_mass()
    rigid_body_com = frame.rigid_body_center_of_mass()
    rigid_body_orientations = frame.rigid_body_orientations()
    rigid_body_linear_velocities = frame.rigid_body_linear_velocities()
    rigid_body_angular_velocities = frame.rigid_body_angular_velocities()
    node_features = t.from_numpy(
        np.concatenate(
            (
                rigid_body_com,
                rigid_body_orientations,
                rigid_body_linear_velocities,
                rigid_body_angular_velocities,
                rigid_body_mass[:, None],
            ),
            axis=1,
        )
    )
    joints = frame.joints()
    joint_num = len(joints)
    edges = t.zeros([2, joint_num * 2], dtype=t.long)
    edge_features = t.zeros([joint_num * 2, 9], dtype=t.float32)
    for idx, joint in enumerate(joints):
        rb_a, rb_b, j_pos, j_axis, j_angle_min, j_angle_max, j_angle = (
            joint[1],
            joint[2],
            np.array(joint[5].tolist()),
            np.array(joint[6].tolist()),
            joint[7],
            joint[8],
            joint[9],
        )
        edges[0][idx] = rb_a
        edges[1][idx] = rb_b
        edge_features[idx] = t.from_numpy(
            np.concatenate(
                [
                    j_pos - rigid_body_com[rb_a],
                    j_axis,
                    np.array([j_angle_min, j_angle_max, j_angle]),
                ]
            )
        )
        edges[0][idx + joint_num] = rb_b
        edges[1][idx + joint_num] = rb_a
        edge_features[idx + joint_num] = t.from_numpy(
            np.concatenate(
                [
                    j_pos - rigid_body_com[rb_b],
                    -j_axis,
                    np.array([j_angle_min, j_angle_max, j_angle]),
                ]
            )
        )
    return (
        t.from_numpy(rigid_body_com),
        node_features,
        edges,
        edge_features,
    )


def build_reward_state(frame: RiseFrame):
    """
    This function is for reward computation, for efficiency only necessary
    info is used
    """
    voxel_positions = frame.voxel_positions()
    # com = frame.com()
    com = np.mean(voxel_positions, axis=0)
    return voxel_positions, com


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.interval = self.end - self.start
        # print(f"Elapsed {self.name} time: {self.interval:.4f} seconds")


class AccumulateTimer:
    def __init__(self):
        self.elapsed = 0.0
        self.start = 0.0
        self.end = 0.0

    def resume(self):
        self.start = time.time()

    def pause(self):
        self.end = time.time()
        self.elapsed += self.end - self.start

    def get(self):
        return self.elapsed


class ThreadPool:
    def __init__(self, num_threads):
        self.tasks = queue.Queue()
        self.results = queue.Queue()
        self.threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=self.worker)
            t.start()
            self.threads.append(t)

    def worker(self):
        while True:
            task = self.tasks.get()
            if task is None:  # None is the signal to shut down
                self.tasks.task_done()
                break
            func, args = task
            result = func(*args)
            self.results.put(result)
            self.tasks.task_done()

    def submit(self, func, *args):
        """Submit a task to the thread pool"""
        self.tasks.put((func, args))

    def map(self, func, iterable):
        """Apply 'func' to every item in 'iterable' and collect the results,
        Note: order is unspecified"""
        for args in iterable:
            self.submit(func, args)

        self.close()
        results = []
        while not self.results.empty():
            results.append(self.results.get())
        return results

    def close(self):
        """Stop the thread pool and wait for all tasks to complete"""
        for _ in self.threads:
            self.tasks.put(None)  # Each None will shut down one thread
        for thread in self.threads:
            thread.join()


class RiseStochasticEnv:
    def __init__(
        self,
        framework: Union[A2C, PPO, List[A2C], List[PPO]],
        devices: List[int],
        env_configs: List[str],
        robot_configs: List[str],
        angles: Union[np.ndarray, List[float]],
        voxel_sample_num: int = 5000,
        random_seed: int = 42,
        debug_log_name: str = None,
    ):
        if isinstance(framework, list):
            assert len(framework) == len(env_configs)
        self.framework = framework
        self.rise = Rise(devices=devices)
        self.env_configs = env_configs
        self.robot_configs = robot_configs
        self.angles = np.array(angles).astype(np.float32)
        self.observations = [
            [] for _ in range(len(robot_configs))
        ]  # type: List[List[dict]]
        self.actions = [
            [] for _ in range(len(robot_configs))
        ]  # type: List[List[List[t.Tensor]]]
        self.reward_states = [
            [] for _ in range(len(robot_configs))
        ]  # type: List[List[dict]]
        self.voxel_sample_num = voxel_sample_num
        self.random_seed = random_seed
        self.debug_log = (
            {
                "input": {
                    id_: {
                        "rotation_signals": [],
                        "time_points": [],
                    }
                    for id_ in range(len(robot_configs))
                },
                "configs": [],
                "file_name": debug_log_name,
            }
            if debug_log_name is not None
            else None
        )

    @staticmethod
    def process_observation(args):
        (
            id_,
            frame,
            rotation_signals,
            voxel_sample_num,
            random_seed,
        ) = args

        (
            positions,
            center_of_mass,
            velocities,
            pressures,
        ) = build_voxel_observations(frame, random_seed, voxel_sample_num)

        (
            node_positions,
            node_features,
            edges,
            edge_features,
        ) = build_kinematic_graph(frame)

        reward_voxel_positions, reward_com = build_reward_state(frame)

        return (
            id_,
            {
                "voxel_positions": positions,
                "voxel_features": t.cat((velocities, pressures), dim=1),
                "node_positions": node_positions,
                "node_features": node_features,
                "edges": edges,
                "edge_features": edge_features,
                "com": center_of_mass.unsqueeze(0),
            },
            {"voxel_positions": reward_voxel_positions, "com": reward_com},
        )

    @staticmethod
    def get_callback(
        framework: Union[A2C, PPO],
        observations: List[List[dict]],
        actions: List[List[List[t.Tensor]]],
        reward_states: List[List[dict]],
        angles: np.ndarray,
        voxel_sample_num: int = 5000,
        random_seed: int = 42,
        debug_log: dict = None,
        callback_timer: AccumulateTimer = None,
    ):

        def callback(
            ids: List[int],
            time_points: List[float],
            frames: List[RiseFrame],
            expansion_signals: List[Any],
            rotation_signals: List[Any],
        ):
            if callback_timer is not None:
                callback_timer.resume()
            with Timer("Preprocess"):
                pool = ThreadPool(4)
                results = list(
                    pool.map(
                        RiseStochasticEnv.process_observation,
                        zip(
                            ids,
                            frames,
                            [np.array(sig) for sig in rotation_signals],
                            [voxel_sample_num] * len(ids),
                            [random_seed + id_ for id_ in ids],
                        ),
                    )
                )
                for id_, observation, reward_state in results:
                    observations[id_].append(observation)
                    reward_states[id_].append(reward_state)
                mapped_results = {x[0]: x[1] for x in results}
                raw_all_observations = [mapped_results[id_] for id_ in ids]

            if isinstance(framework, list):
                all_action = []
                with Timer("Eval"):
                    with t.no_grad():
                        for idx, (id_, observation) in enumerate(
                            zip(ids, raw_all_observations)
                        ):
                            output = framework[id_].act(
                                list_of_dict_to_dict_of_list([observation]),
                                call_dp_or_ddp_internal_module=True,
                            )
                            all_action.append(angles[output[0][0].cpu().numpy()])
                            actions[id_].append([output[0][0], output[1][0]])
                            rotation_signals[idx][:] = all_action[-1]

            else:
                all_observations = list_of_dict_to_dict_of_list(raw_all_observations)
                with Timer("Eval"):
                    with t.no_grad():
                        all_output = framework.act(
                            all_observations,
                            call_dp_or_ddp_internal_module=True,
                        )
                all_action = [angles[a.cpu().numpy()] for a in all_output[0]]

                for idx, id_ in enumerate(ids):
                    actions[id_].append([all_output[0][idx], all_output[1][idx]])
                    rotation_signals[idx][:] = all_action[idx]

            if debug_log is not None:
                for idx, id_ in enumerate(ids):
                    debug_log["input"][id_]["rotation_signals"].append(
                        np.array(rotation_signals[idx])
                    )
                    debug_log["input"][id_]["time_points"].append(time_points[idx])
                with open(debug_log["file_name"], "wb") as file:
                    pickle.dump(debug_log, file)

            if callback_timer is not None:
                callback_timer.pause()

        return callback

    @staticmethod
    def replay(
        devices: List[int],
        debug_log_path: str,
        save_record: bool = True,
        record_buffer_size: int = 100,
        return_observations: bool = False,
        voxel_sample_num: int = 5000,
        random_seed: int = 42,
        override_env_config: str = None,
        replay_configs: List[int] = None,
    ):
        with open(debug_log_path, "rb") as file:
            debug_log = pickle.load(file)
        if replay_configs is not None:
            for real_id in replay_configs:
                if len(debug_log["configs"]) <= real_id:
                    raise ValueError("Invalid replay index")
        else:
            replay_configs = list(range(len(debug_log["configs"])))

        current_index = {real_id: 0 for real_id in replay_configs}
        observations = {real_id: [] for real_id in replay_configs}

        def replay_callback(
            ids: List[int],
            time_points: List[float],
            frames: List[RiseFrame],
            expansion_signals: List[Any],
            rotation_signals: List[Any],
        ):
            nonlocal current_index

            if return_observations:
                pool = ThreadPool(4)
                for id_, observation, _ in pool.map(
                    RiseStochasticEnv.process_observation,
                    zip(
                        ids,
                        frames,
                        [np.array(sig) for sig in rotation_signals],
                        [voxel_sample_num] * len(ids),
                        [random_seed + id_ for id_ in ids],
                    ),
                ):
                    real_id = replay_configs[id_]
                    observations[real_id].append(observation)

            for idx, id_ in enumerate(ids):
                real_id = replay_configs[id_]
                if current_index[real_id] < len(
                    debug_log["input"][real_id]["rotation_signals"]
                ):
                    rotation_signals[idx][:] = debug_log["input"][real_id][
                        "rotation_signals"
                    ][current_index[real_id]]
                else:
                    rotation_signals[idx][:] = debug_log["input"][real_id][
                        "rotation_signals"
                    ][-1]
                current_index[real_id] += 1
            print(current_index)

        if override_env_config is not None:
            for configs in debug_log["configs"]:
                configs[0] = override_env_config

        rise = Rise(devices=devices)
        result = rise.run_sims(
            [debug_log["configs"][id_] for id_ in replay_configs],
            list(range(len(replay_configs))),
            replay_callback,
            record_buffer_size=record_buffer_size,
            save_result=True,
            save_record=save_record,
            log_level="info",
        )
        if return_observations:
            return result, observations
        else:
            return result

    def run(
        self,
        generation: int,
        save_record: bool = False,
        record_buffer_size: int = 100,
    ):
        """
        Args:
            generation: A unique generation number
            save_record: Whether to save the record for every simulation.
            record_buffer_size: Record buffer size for every simulation, on GPU.
        Returns:
            A four element tuple,
                first: List[bool], length N, indicating whether simulation is
                    successful, unsuccessful experiments will not have a result file,
                    the record file may exist and contains information up until the error
                    in simulation occurs.
                second: List[List[Dict[str, t.Tensor]]], shape [N, F*, ...], observations.
                third: List[List[Tuple[t.Tensor, t.Tensor, t.Tensor]]], actions.
                fourth: List[List[Dict[str, np.ndarray]]], shape [N, F*, ...], reward_states.
        Note:
            N: robot num (batch size)

            F*: frame num, may differ between robots
        """
        configs = []
        callback_timer = AccumulateTimer()
        callback = self.get_callback(
            self.framework,
            self.observations,
            self.actions,
            self.reward_states,
            self.angles,
            self.voxel_sample_num,
            self.random_seed + generation,
            self.debug_log,
            callback_timer,
        )
        for env_config, robot_config in zip(self.env_configs, self.robot_configs):
            configs.append([env_config, robot_config])

        if self.debug_log is not None:
            self.debug_log["configs"] = configs

        result = self.rise.run_sims(
            configs,
            list(range(len(configs))),
            callback,
            record_buffer_size=1 if not save_record else record_buffer_size,
            save_result=True,
            save_record=save_record,
            log_level="info",
        )
        print(f"Callback time: {callback_timer.elapsed}")
        return result, self.observations, self.actions, self.reward_states
