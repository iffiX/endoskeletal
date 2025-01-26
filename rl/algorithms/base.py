import torch as t
import torch.nn as nn
from typing import Callable
from torchviz import make_dot


class TorchFramework(nn.Module):
    """
    Base framework for all algorithms
    """

    _is_top = []
    _is_restorable = []

    def __init__(self):
        super(TorchFramework, self).__init__()
        self._visualized = set()
        self._backward = t.autograd.backward

    @property
    def optimizers(self):
        raise NotImplementedError

    @optimizers.setter
    def optimizers(self, optimizers):
        raise NotImplementedError

    @property
    def lr_schedulers(self):
        raise NotImplementedError

    @property
    def top_models(self):
        models = []
        for m in self._is_top:
            models.append(getattr(self, m))
        return models

    @property
    def restorable_models(self):
        models = []
        for m in self._is_restorable:
            models.append(getattr(self, m))
        return models

    @property
    def backward_function(self):
        return self._backward

    @classmethod
    def get_top_model_names(cls):
        """
        Get attribute name of top level nn models.
        """
        return cls._is_top

    @classmethod
    def get_restorable_model_names(cls):
        """
        Get attribute name of restorable nn models.
        """
        return cls._is_restorable

    def set_backward_function(self, backward_func: Callable):
        """
        Replace the default backward function with a custom function.
        The default loss backward function is ``torch.autograd.backward``
        """
        assert callable(backward_func), "Backward function must be callable."
        self._backward = backward_func

    def enable_multiprocessing(self):
        """
        Enable multiprocessing for all modules.
        """
        for top in self._is_top:
            model = getattr(self, top)
            model.share_memory()

    def visualize_model(self, final_tensor: t.Tensor, name: str, directory: str):
        if name in self._visualized:
            return
        else:
            self._visualized.add(name)
            g = make_dot(final_tensor)
            g.render(
                filename=name,
                directory=directory,
                view=False,
                cleanup=False,
                quiet=True,
            )
