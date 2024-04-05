"""Implementation of structured inverse-, root-free Shampoo."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

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
