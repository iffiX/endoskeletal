from typing import Union, Dict, List, Tuple, Callable, Any
import torch as t
import torch.nn as nn

from rl.buffers.buffer import Buffer
from rl.net import NeuralNetworkModule
from rl.transition import TransitionStochastic
from .base import TorchFramework
from .utils import safe_call, batch_tensor_dicts, find_shared_parameters


class A2C(TorchFramework):
    """
    A2C framework.
    """

    _is_top = ["actor", "critic"]
    _is_restorable = ["actor", "critic"]

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
        Important:
            When given a state, and an optional action, actor must
            at least return two values:

            **1. Action**

              For **contiguous environments**, action must be of shape
              ``[batch_size, action_dim]`` and *clamped by action space*.
              For **discrete environments**, action could be of shape
              ``[batch_size, action_dim]`` if it is a one hot vector, or
              ``[batch_size, 1]`` or [batch_size] if it is a categorically
              encoded integer.

              When the given action is not None, actor must return the given
              action.

            **2. Log likelihood of action (action probability)**

              For either type of environment, log likelihood is of shape
              ``[batch_size, 1]`` or ``[batch_size]``.

              Action probability must be differentiable, Gradient of actor
              is calculated from the gradient of action probability.

              When the given action is not None, actor must return the log
              likelihood of the given action.

            The third entropy value is optional:

            **3. Entropy of action distribution**

              Entropy is usually calculated using dist.entropy(), its shape
              is ``[batch_size, 1]`` or ``[batch_size]``. You must specify
              ``entropy_weight`` to make it effective.

        Hint:
            For contiguous environments, action's are not directly output by
            your actor, otherwise it would be rather inconvenient to calculate
            the log probability of action. Instead, your actor network should
            output parameters for a certain distribution
            (eg: :class:`~torch.distributions.categorical.Normal`)
            and then draw action from it.

            For discrete environments,
            :class:`~torch.distributions.categorical.Categorical` is sufficient,
            since differentiable ``rsample()`` is not needed.

            This trick is also known as **reparameterization**.

        Hint:
            Actions are from samples during training in the actor critic
            family (A2C, PPO).

            When your actor model is given a batch of actions and states, it
            must evaluate the states, and return the log likelihood of the
            given actions instead of re-sampling actions.

            An example of your actor in contiguous environments::

                class ActorNet(nn.Module):
                    def __init__(self):
                        super(ActorNet, self).__init__()
                        self.fc = nn.Linear(3, 100)
                        self.mu_head = nn.Linear(100, 1)
                        self.sigma_head = nn.Linear(100, 1)

                    def forward(self, state, action=None):
                        x = t.relu(self.fc(state))
                        mu = 2.0 * t.tanh(self.mu_head(x))
                        sigma = F.softplus(self.sigma_head(x))
                        dist = Normal(mu, sigma)
                        action = (action
                                  if action is not None
                                  else dist.sample())
                        action_entropy = dist.entropy()
                        action = action.clamp(-2.0, 2.0)

                        # Since we are representing a multivariate gaussian
                        # distribution in terms of independent univariate gaussians:
                        action_log_prob = dist.log_prob(action).sum(
                            dim=1, keepdim=True
                        )
                        return action, action_log_prob, action_entropy

        Hint:
            Entropy weight is usually negative, to increase exploration.

            Value weight is usually 0.5. So critic network converges less
            slowly than the actor network and learns more conditions.

            Update equation is equivalent to:

            :math:`Loss= w_e * Entropy + w_v * Loss_v + w_a * Loss_a`
            :math:`Loss_a = -log\\_likelihood * advantage`
            :math:`Loss_v = criterion(target\\_bellman\\_value - V(s))`

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
            w
            actor_learning_rate: Learning rate of the actor optimizer,
                not compatible with ``lr_scheduler``.
            critic_learning_rate: Learning rate of the critic optimizer,
                not compatible with ``lr_scheduler``.
            entropy_weight: Weight of entropy in your loss function, a positive
                entropy weight will minimize entropy, while a negative one will
                maximize entropy.
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
        self.batch_size = batch_size
        self.discount = discount
        self.entropy_weight = entropy_weight
        self.gae_lambda = gae_lambda
        self.normalize_advantage = normalize_advantage
        self.visualize = visualize
        self.visualize_dir = visualize_dir
        super().__init__()
        self.actor = actor
        self.critic = critic
        if optimizer_args is None:
            optimizer_args = ((), ())
        if optimizer_kwargs is None:
            optimizer_kwargs = ({}, {})

        shared_parameters, (actor_private_parameters, critic_private_parameters) = (
            find_shared_parameters([actor, critic])
        )
        if len(shared_parameters) > 0:
            print(f"Found shared parameter num: {len(shared_parameters)}")
        self.actor_optim = optimizer(
            (
                shared_parameters + actor_private_parameters
                if shared_parameters_belong_to_optimizer in ("both", "actor")
                else actor_private_parameters
            ),
            *optimizer_args[0],
            lr=actor_learning_rate,
            **optimizer_kwargs[0],
        )
        self.critic_optim = optimizer(
            (
                shared_parameters + critic_private_parameters
                if shared_parameters_belong_to_optimizer in ("both", "critic")
                else critic_private_parameters
            ),
            *optimizer_args[1],
            lr=critic_learning_rate,
            **optimizer_kwargs[1],
        )
        self.replay_buffer = (
            Buffer(replay_size, replay_device)
            if replay_buffer is None
            else replay_buffer
        )

        if lr_scheduler is not None:
            if lr_scheduler_args is None:
                lr_scheduler_args = ((), ())
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = ({}, {})
            self.actor_lr_sch = lr_scheduler(
                self.actor_optim,
                *lr_scheduler_args[0],
                **lr_scheduler_kwargs[0],
            )
            self.critic_lr_sch = lr_scheduler(
                self.critic_optim, *lr_scheduler_args[1], **lr_scheduler_kwargs[1]
            )

        self.criterion = criterion

    @property
    def optimizers(self):
        return [self.actor_optim, self.critic_optim]

    @optimizers.setter
    def optimizers(self, optimizers):
        self.actor_optim, self.critic_optim = optimizers

    @property
    def lr_schedulers(self):
        if hasattr(self, "actor_lr_sch") and hasattr(self, "critic_lr_sch"):
            return [self.actor_lr_sch, self.critic_lr_sch]
        return []

    def act(
        self, state: Dict[str, Any], call_dp_or_ddp_internal_module=False, *_, **__
    ):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Anything produced by actor.
        """
        # No need to safe_return because the number of
        # returned values is always more than one
        return safe_call(
            self.actor,
            state,
            call_dp_or_ddp_internal_module=call_dp_or_ddp_internal_module,
        )

    def _eval_act(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        call_dp_or_ddp_internal_module=False,
        *_,
        **__,
    ):
        """
        Use actor network to evaluate the log-likelihood of a given
        action in the current state.

        Returns:
            Anything produced by actor.
        """
        return safe_call(
            self.actor,
            state,
            action,
            call_dp_or_ddp_internal_module=call_dp_or_ddp_internal_module,
        )

    def _criticize(
        self, state: Dict[str, Any], call_dp_or_ddp_internal_module=False, *_, **__
    ):
        """
        Use critic network to evaluate current value.

        Returns:
            Value of shape ``[batch_size, 1]``
        """
        return safe_call(
            self.critic,
            state,
            call_dp_or_ddp_internal_module=call_dp_or_ddp_internal_module,
        )[0]

    def _process_episode(self, episode: List[Dict], concatenate_samples=True):
        """
        Process transitions in episode.
        """
        episode[-1]["value"] = episode[-1]["reward"]
        # get critic value for every transition
        states = batch_tensor_dicts(
            [transition["state"] for transition in episode],
            concatenate_samples=concatenate_samples,
        )

        if self.gae_lambda is not None:
            with t.no_grad():
                critic_values = (
                    self._criticize(states, call_dp_or_ddp_internal_module=True)
                    .view(len(episode))
                    .tolist()
                )

        # calculate value for each transition
        for i in reversed(range(1, len(episode))):
            episode[i - 1]["value"] = (
                episode[i]["value"] * self.discount + episode[i - 1]["reward"]
            )

        # calculate advantage
        if self.gae_lambda == 1.0:
            for idx, transition in enumerate(episode):
                transition["gae"] = transition["value"] - critic_values[idx]
        elif self.gae_lambda == 0.0:
            for idx, transition in enumerate(episode):
                if transition["terminal"] or idx == len(episode) - 1:
                    transition["gae"] = transition["reward"] - critic_values[idx]
                else:
                    transition["gae"] = (
                        transition["reward"]
                        + self.discount
                        * (1 - float(transition["terminal"]))
                        * critic_values[idx + 1]
                        - critic_values[idx]
                    )
        elif self.gae_lambda is not None:
            last_critic_value = 0
            last_gae = 0
            for idx, transition in reversed(list(enumerate(episode))):
                critic_value = critic_values[idx]
                gae_delta = (
                    transition["reward"]
                    + self.discount
                    * last_critic_value
                    * (1 - float(transition["terminal"]))
                    - critic_value
                )
                last_critic_value = critic_value
                last_gae = transition["gae"] = (
                    last_gae
                    * self.discount
                    * (1 - float(transition["terminal"]))
                    * self.gae_lambda
                    + gae_delta
                )
        else:
            for idx, transition in enumerate(episode):
                transition["gae"] = transition["reward"]

        for i in range(len(episode)):
            if "entropy" not in episode[i]:
                episode[i]["entropy"] = 0
            episode[i] = TransitionStochastic(**episode[i])

    def store_episode(self, episode: List[Dict], concatenate_samples=True):
        """
        Add a full episode of transition samples to the replay buffer.

        Required attributes: "state", "action", "log_prob", "reward", "terminal"

        Optional attributes: "entropy" (if entropy_weight is specified in initialization)

        "value" and "gae" are automatically calculated.
        """
        self._process_episode(episode, concatenate_samples=concatenate_samples)
        self.replay_buffer.store_episode(
            episode,
            required_attrs=(
                "state",
                "action",
                "log_prob",
                "entropy",
                "reward",
                "value",
                "gae",
                "terminal",
            ),
        )

    def get_loss(
        self, get_actor_loss=True, get_critic_loss=True, concatenate_samples=True, **__
    ):
        """
        Update network weights by sampling from buffer. Buffer
        will be cleared after update is finished.

        Args:
            get_actor_loss: Whether compute loss for actor.
            get_critic_loss: Whether compute loss for critic.
            concatenate_samples: Whether concatenate the samples.

        Returns:
            [(optional)actor loss, (optional)critic loss]
        """
        losses = []
        if get_actor_loss:
            # sample a batch
            batch_size, (state, action, advantage) = self.replay_buffer.sample_batch(
                self.batch_size,
                sample_method="random_unique",
                concatenate=concatenate_samples,
                sample_attrs=["state", "action", "gae"],
                additional_concat_custom_attrs=["gae"],
            )

            if self.entropy_weight is not None:
                __, log_prob, entropy, *_ = self._eval_act(state, action)
            else:
                __, log_prob, *_ = self._eval_act(state, action)
                entropy = None

            if not concatenate_samples:
                advantage = t.tensor(advantage)

            # normalize advantage
            if self.normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

            entropy = entropy.view(batch_size, 1) if entropy is not None else None
            log_prob = log_prob.view(batch_size, 1)
            advantage = advantage.view(batch_size, 1)

            log_prob = t.clamp(log_prob, -1e2, 1e2)

            # calculate policy loss
            act_policy_loss = -(log_prob * advantage.type_as(log_prob))

            if self.entropy_weight is not None:
                act_policy_loss += self.entropy_weight * entropy.mean()

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

    def finish_update(self):
        self.replay_buffer.clear()

    def update_lr_scheduler(self):
        """
        Update learning rate schedulers.
        """
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()
