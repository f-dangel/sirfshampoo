"""Implementation of structured inverse-, root-free Shampoo."""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer


class SIRFShampoo(Optimizer):
    """Structured inverse-free and root-free Shampoo optimizer."""

    def __init__(
        self,
        model: Module,
        params: Optional[Union[List[Parameter], Dict[str, Any]]] = None,
        beta1: float = 0.001,
    ):
        """Set up the optimizer.

        Args:
            model: The model to optimize. The optimizer needs access to the model
                to figure out weights/biases of one layer.
            params: The parameters to optimize. If `None`, all parameters of the
                model are optimized. Default: `None`.
            beta1: Learning rate for the parameter update. Default: `0.001`.
        """
        defaults = dict(beta1=beta1)

        if params is None:
            params = [p for p in model.parameters() if p.requires_grad]

        super().__init__(params, defaults)

        self.model = model
        # _params_in_layer maps layer to parameter names which are trained
        # _layer_to_param_group maps layer names to parameter group indices
        self._params_in_layer, self._layer_to_param_group = self._create_mappings()

    def _create_mappings(self) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """Create mappings from layers to parameters and parameter groups.

        Raises:
            ValueError: If parameters in the same layer are in different parameter
                groups.

        Returns:
            A dictionary mapping layer names to lists of parameter names and
            a dictionary mapping layer names to parameter group indices.
        """
        params = sum((group["params"] for group in self.param_groups), [])
        param_ids = {p.data_ptr() for p in params}

        # keys are layer names, values are lists containing the parameter names
        params_in_layer = defaultdict(list)
        for name, p in self.model.named_parameters():
            if p.data_ptr() in param_ids:
                sep_idx = name.rfind(".")  # position of param/layer name separator
                layer_name, p_name = name[:sep_idx], name[sep_idx + 1 :]
                params_in_layer[layer_name].append(p_name)

        # keys are layer names, values are parameter group indices
        layer_to_param_group = {}
        for layer_name, param_names in params_in_layer.items():
            layer = self.model.get_submodule(layer_name)
            params = [layer.get_parameter(p_name) for p_name in param_names]

            param_group_idx = set()
            for p in params:
                for group_idx, group in enumerate(self.param_groups):
                    group_param_ids = {p.data_ptr() for p in group["params"]}
                    if p.data_ptr() in group_param_ids:
                        param_group_idx.add(group_idx)
            if len(param_group_idx) > 1:
                raise ValueError(
                    f"{layer_name}' params are in multiple groups: {param_group_idx}."
                )
            layer_to_param_group[layer_name] = param_group_idx.pop()

        return params_in_layer, layer_to_param_group

    def _get_param_group_entry(self, layer_name: str, key: str) -> Any:
        """Get an entry from the parameter group of a layer.

        Args:
            layer_name: The name of the layer.
            key: The key of the entry to get.

        Returns:
            The entry from the parameter group of the layer.
        """
        return self.param_groups[self._layer_to_param_group[layer_name]][key]

    def _update_preconditioner(self, layer_name: str) -> None:
        """Update the preconditioner of a layer.

        Args:
            layer_name: The name of the layer.
        """
        warn("_update_preconditioner is a dummy implementation.")

    def _precondition_gradient(self, layer_name: str) -> Dict[str, Tensor]:
        """Precondition the gradient of a layer.

        Args:
            layer_name: The name of the layer.

        Returns:
            The preconditioned gradient.
        """
        warn("_precondition_gradient is a dummy implementation.")
        param_names = self._params_in_layer[layer_name]
        return {
            p_name: self.model.get_parameter(f"{layer_name}.{p_name}").grad
            for p_name in param_names
        }

    def step(self, closure: Optional[Callable] = None) -> None:
        """Perform a single optimization step.

        Args:
            closure: Not supported. Default: `None`.

        Raises:
            NotImplementedError: If `closure` is not `None`.
        """
        if closure is not None:
            raise NotImplementedError("Closure is not supported.")

        for layer_name in self._params_in_layer:
            self._step(layer_name)

    def _step(self, layer_name: str):
        """Perform a single optimization step for a layer.

        Args:
            layer_name: The name of the layer.
        """
        param_names = self._params_in_layer[layer_name]
        self._update_preconditioner(layer_name)
        update = self._precondition_gradient(layer_name)
        params = [
            self.model.get_parameter(f"{layer_name}.{p_name}") for p_name in param_names
        ]
        beta1 = self._get_param_group_entry(layer_name, "beta1")

        for p_name, p in zip(param_names, params):
            p_step = update[p_name]
            p.data.add_(p_step, alpha=-beta1)
