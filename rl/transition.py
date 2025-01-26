import torch as t
import numpy as np
from typing import Union, Dict, Iterable, Any, NewType
from itertools import chain

Scalar = NewType("Scalar", Union[int, float, bool])


class TransitionBase:
    """
    Base class for all transitions
    """

    _inited = False

    def __init__(
        self,
        major_attr: Iterable[str],
        sub_attr: Iterable[str],
        custom_attr: Iterable[str],
        major_data: Iterable[Dict[str, t.Tensor]],
        sub_data: Iterable[Union[Scalar, t.Tensor]],
        custom_data: Iterable[Any],
    ):
        """
        Note:
            Major attributes store things like state, action, etc.
            They are usually dictionaries of tensors, **concatenated by keys**
            during sampling, and passed as keyword arguments to actors,
            critics, etc.

            Sub attributes store things like terminal states, reward, etc.
            They are usually scalars, but tensors are supported as well,
            **concatenated directly** during sampling, and used in different
            algorithms.

            Custom attributes store not concatenatable values, usually user
            specified states, used in models or as special arguments in
            different algorithms. They will be collected together as a list
            during sampling, **no further concatenation is performed**.

        Args:
            major_attr: A list of major attribute names.
            sub_attr: A list of sub attribute names.
            custom_attr: A list of custom attribute names.
            major_data: Data of major attributes.
            sub_data: Data of sub attributes.
            custom_data: Data of custom attributes.
        """
        self.episode_id = None
        self._major_attr = list(major_attr)
        self._sub_attr = list(sub_attr)
        self._custom_attr = list(custom_attr)
        self._keys = self._major_attr + self._sub_attr + self._custom_attr
        self._length = len(self._keys)

        for attr, data in zip(
            chain(major_attr, sub_attr, custom_attr),
            chain(major_data, sub_data, custom_data),
        ):
            object.__setattr__(self, attr, data)
        # will trigger _check_validity in __setattr__
        self._inited = True
        self._detach()

    def __len__(self):
        return self._length

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        if key not in self._keys:
            raise RuntimeError(
                "You cannot dynamically set new attributes in a Transition object!"
            )
        object.__setattr__(self, key, value)
        self._check_validity()

    def __setattr__(self, key, value):
        if not self._inited:
            object.__setattr__(self, key, value)
        else:
            if key != "episode_id" and key not in self._keys:
                raise RuntimeError(
                    "You cannot dynamically set new attributes in"
                    " a Transition object!"
                )
        if key != "episode_id" and self._inited:
            self._check_validity()

    @property
    def major_attr(self):
        return self._major_attr

    @property
    def sub_attr(self):
        return self._sub_attr

    @property
    def custom_attr(self):
        return self._custom_attr

    def keys(self):
        """
        Returns:
            All attribute names in current transition object.
            Ordered in: "major_attrs, sub_attrs, custom_attrs"
        """
        return self._keys

    def items(self):
        """
        Returns:
            All attribute values in current transition object.
        """
        for k in self._keys:
            yield k, getattr(self, k)

    def has_keys(self, keys: Iterable[str]):
        """
        Args:
            keys: A list of keys

        Returns:
            A bool indicating whether current transition object
            contains all specified keys.
        """
        return all([k in self._keys for k in keys])

    def to(self, device: Union[str, t.device]):
        """
        Move current transition object to another device. will be
        a no-op if it already locates on that device.

        Args:
            device: A valid pytorch device.

        Returns:
            Self.
        """
        for ma in self._major_attr:
            ma_data = getattr(self, ma)
            for k, v in ma_data.items():
                ma_data[k] = v.to(device)
        for sa in self._sub_attr:
            sa_data = getattr(self, sa)
            if t.is_tensor(sa_data):
                object.__setattr__(self, sa, sa_data.to(device))
        return self

    def _detach(self):
        """
        Detach all tensors in major attributes and sub attributes, put
        data of all attributes in place, but do not copy them.

        Returns:
            Self.
        """
        for ma in self._major_attr:
            ma_data = getattr(self, ma)
            for k, v in ma_data.items():
                ma_data[k] = v.detach()
        for sa in self._sub_attr:
            sa_data = getattr(self, sa)
            if t.is_tensor(sa_data):
                object.__setattr__(self, sa, sa_data.detach())
        for ca in self._custom_attr:
            ca_data = getattr(self, ca)
            object.__setattr__(self, ca, ca_data)
        return self

    def _check_validity(self):
        """
        Check validity of current transition object.

        Raises:
            ``ValueError`` if anything is invalid.
        """
        for ma in self._major_attr:
            ma_data = getattr(self, ma)
            for k, v in ma_data.items():
                if not t.is_tensor(v):
                    raise ValueError(
                        f'Key "{k}" of transition major attribute "{ma}" '
                        "is an invalid tensor"
                    )
        for sa in self._sub_attr:
            sa_data = getattr(self, sa)
            if np.isscalar(sa_data):
                # will return true for inbuilt scalar types
                # like int, bool, float
                pass
            elif t.is_tensor(sa_data):
                if sa_data.dim() > 1:
                    raise ValueError(
                        f'Transition sub attribute "{sa}" is an invalid tensor.'
                    )
            else:
                raise ValueError(
                    f'Transition sub attribute "{sa}" has invalid '
                    f"value {sa_data}, requires scalar or tensor."
                )


class Transition(TransitionBase):
    """
    The default Transition class.

    Have two main attributes: ``state``, ``action``.

    Have two sub attributes: ``reward`` and ``terminal``.

    Have one custom attribute: ``episode_id``.

    Store one transition step of one agent.
    """

    # for auto-suggestion in IDEs

    state = None  # type: Dict[str, t.Tensor]
    action = None  # type: Dict[str, t.Tensor]
    reward = None  # type: Union[float, t.Tensor]
    terminal = None  # type: bool

    def __init__(
        self,
        state: Dict[str, t.Tensor],
        action: Dict[str, t.Tensor],
        reward: Union[float, t.Tensor],
        terminal: bool,
        **kwargs,
    ):
        """
        Args:
            state: Previous observed state.
            action: Action of agent.
            reward: Reward of agent.
            terminal: Whether environment has reached terminal state.
            **kwargs: Custom attributes. They are ordered in the alphabetic
                order (provided by ``sort()``) when you call ``keys()``.

        Note:
            You should not store any tensor inside ``**kwargs`` as they will
            not be moved to the sample output device.
        """
        custom_keys = sorted(list(kwargs.keys()))
        super().__init__(
            major_attr=["state", "action"],
            sub_attr=["reward", "terminal"],
            custom_attr=custom_keys,
            major_data=[state, action],
            sub_data=[reward, terminal],
            custom_data=[kwargs[k] for k in custom_keys],
        )


class TransitionStochastic(TransitionBase):
    """
    Transition class for stochastic policies.

    Have two main attributes: ``state``, ``action``.

    Have four sub attributes: ``log_prob``, ``entropy``, ``reward`` and ``terminal``.

    Have one custom attribute: ``episode_id``.

    Store one transition step of one agent.
    """

    # for auto-suggestion in IDEs

    state = None  # type: Dict[str, t.Tensor]
    action = None  # type: Dict[str, t.Tensor]
    reward = None  # type: Union[float, t.Tensor]
    log_prob = None  # type: t.Tensor
    entropy = None  # type: t.Tensor
    terminal = None  # type: bool

    def __init__(
        self,
        state: Dict[str, t.Tensor],
        action: Dict[str, t.Tensor],
        reward: Union[float, t.Tensor],
        log_prob: t.Tensor,
        entropy: t.Tensor,
        terminal: bool,
        **kwargs,
    ):
        """
        Args:
            state: Previous observed state.
            action: Action of agent.
            reward: Reward of agent.
            terminal: Whether environment has reached terminal state.
            **kwargs: Custom attributes. They are ordered in the alphabetic
                order (provided by ``sort()``) when you call ``keys()``.

        Note:
            You should not store any tensor inside ``**kwargs`` as they will
            not be moved to the sample output device.
        """
        custom_keys = sorted(list(kwargs.keys()))
        super().__init__(
            major_attr=["state", "action"],
            sub_attr=["log_prob", "entropy", "reward", "terminal"],
            custom_attr=custom_keys,
            major_data=[state, action],
            sub_data=[log_prob, entropy, reward, terminal],
            custom_data=[kwargs[k] for k in custom_keys],
        )
