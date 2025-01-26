from .a2c import *


class PPO(A2C):
    """
    PPO framework.
    """

    def __init__(
        self,
        actor: Union[NeuralNetworkModule, nn.Module],
        critic: Union[NeuralNetworkModule, nn.Module],
        optimizer: Callable,
        criterion: Callable,
        *_,
        shared_parameters_belong_to_optimizer: str = "both",
        lr_scheduler: Callable = None,
        lr_scheduler_args: Tuple[Tuple, Tuple] = None,
        lr_scheduler_kwargs: Tuple[Dict, Dict] = None,
        optimizer_args: Tuple[Tuple, Tuple] = None,
        optimizer_kwargs: Tuple[Dict, Dict] = None,
        batch_size: int = 100,
        actor_learning_rate: float = 0.001,
        critic_learning_rate: float = 0.001,
        entropy_weight: float = None,
        surrogate_loss_clip: float = 0.2,
        gae_lambda: Union[float, None] = 1.0,
        discount: float = 0.99,
        normalize_advantage: bool = True,
        replay_size: int = 500000,
        replay_device: Union[str, t.device] = "cpu",
        replay_buffer: Buffer = None,
        visualize: bool = False,
        visualize_dir: str = "",
        **__,
    ):
        """
        See Also:
            :class:`.A2C`

        Args:
            actor: Actor network module.
            critic: Critic network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            criterion: Criterion used to evaluate the value loss.
            shared_parameters_belong_to_optimizer: Which optimizer the
                shared parameters belongs to, "actor" or "critic" or "both.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            optimizer_args: Arguments of the optimizer.
            optimizer_kwargs: Keyword arguments of the optimizer.
            batch_size: Batch size used during training.
            actor_learning_rate: Learning rate of the actor optimizer,
                not compatible with ``lr_scheduler``.
            critic_learning_rate: Learning rate of the critic optimizer,
                not compatible with ``lr_scheduler``.
            entropy_weight: Weight of entropy in your loss function, a positive
                entropy weight will minimize entropy, while a negative one will
                maximize entropy.
            surrogate_loss_clip: Surrogate loss clipping parameter in PPO.
            gae_lambda: :math:`\\lambda` used in generalized advantage
                estimation.
            discount: :math:`\\gamma` used in the bellman function.
            normalize_advantage: Whether to normalize sampled advantage values in
                the batch.
            replay_size: Replay buffer size. Not compatible with
                ``replay_buffer``.
            replay_device: Device where the replay buffer locates on, Not
                compatible with ``replay_buffer``.
            replay_buffer: Custom replay buffer.
            visualize: Whether visualize the network flow in the first pass.
            visualize_dir: Visualized graph save directory.
        """
        super().__init__(
            actor,
            critic,
            optimizer,
            criterion,
            shared_parameters_belong_to_optimizer=shared_parameters_belong_to_optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            optimizer_args=optimizer_args,
            optimizer_kwargs=optimizer_kwargs,
            batch_size=batch_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            entropy_weight=entropy_weight,
            gae_lambda=gae_lambda,
            discount=discount,
            normalize_advantage=normalize_advantage,
            replay_size=replay_size,
            replay_device=replay_device,
            replay_buffer=replay_buffer,
            visualize=visualize,
            visualize_dir=visualize_dir,
        )
        self.surr_clip = surrogate_loss_clip

    def _process_episode(self, episode: List[Dict], concatenate_samples=True):
        # DOC INHERITED
        super()._process_episode(episode, concatenate_samples=concatenate_samples)

    def store_episode(self, episode: List[Dict], concatenate_samples=True):
        """
        Add a full episode of transition samples to the replay buffer.

        "value" and "gae" are automatically calculated.
        """
        self._process_episode(episode, concatenate_samples=concatenate_samples)
        self.replay_buffer.store_episode(
            episode,
            required_attrs=(
                "state",
                "action",
                "log_prob",
                "reward",
                "value",
                "gae",
                "terminal",
            ),
        )

    def get_loss(
        self, get_actor_loss=True, get_critic_loss=True, concatenate_samples=True, **__
    ):
        # DOC INHERITED

        losses = []
        if get_actor_loss:
            # sample a batch
            batch_size, (state, action, log_prob, advantage) = (
                self.replay_buffer.sample_batch(
                    self.batch_size,
                    sample_method="random_unique",
                    concatenate=concatenate_samples,
                    sample_attrs=["state", "action", "log_prob", "gae"],
                    additional_concat_custom_attrs=["gae"],
                )
            )

            if self.entropy_weight is not None:
                __, new_log_prob, new_action_entropy, *_ = self._eval_act(state, action)
            else:
                __, new_log_prob, *_ = self._eval_act(state, action)
                new_action_entropy = None

            if not concatenate_samples:
                log_prob = t.tensor(log_prob)
                advantage = t.tensor(advantage)

            # normalize advantage
            if self.normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

            new_action_entropy = (
                new_action_entropy.view(batch_size, 1)
                if new_action_entropy is not None
                else None
            )
            log_prob = log_prob.view(batch_size, 1)
            new_log_prob = new_log_prob.view(batch_size, 1)
            advantage = advantage.view(batch_size, 1)

            new_log_prob = t.clamp(new_log_prob, -1e2, 1e2)

            # calculate surrogate loss
            # The function of this process is ignoring actions that are not
            # likely to be produced in current actor policy distribution,
            # Because in each update, the old policy distribution diverges
            # from the current distribution more and more.
            sim_ratio = t.exp(new_log_prob - log_prob.to(new_log_prob.device))
            advantage = advantage.type_as(sim_ratio)
            surr_loss_1 = sim_ratio * advantage
            surr_loss_2 = (
                t.clamp(sim_ratio, 1 - self.surr_clip, 1 + self.surr_clip) * advantage
            )

            # calculate policy loss using surrogate loss
            act_policy_loss = -t.min(surr_loss_1, surr_loss_2)

            if new_action_entropy is not None:
                act_policy_loss += self.entropy_weight * new_action_entropy.mean()

            act_policy_loss = act_policy_loss.mean()

            if self.visualize:
                self.visualize_model(act_policy_loss, "actor", self.visualize_dir)

            losses.append(act_policy_loss)

        if get_critic_loss:
            # sample a batch
            batch_size, (state, target_value) = self.replay_buffer.sample_batch(
                self.batch_size,
                sample_method="random_unique",
                concatenate=concatenate_samples,
                sample_attrs=["state", "value"],
                additional_concat_custom_attrs=["value"],
            )

            if not concatenate_samples:
                target_value = t.tensor(target_value)

            # calculate value loss
            value = self._criticize(state)
            value_loss = self.criterion(
                target_value.type_as(value).view(batch_size, 1),
                value.view(batch_size, 1),
            )

            if self.visualize:
                self.visualize_model(value_loss, "critic", self.visualize_dir)

            losses.append(value_loss)

        return losses
