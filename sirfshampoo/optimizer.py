"""Implementation of structured inverse-, root-free Shampoo."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch import Tensor, eye, zeros_like
from torch.nn import Module, Parameter
from torch.optim import Optimizer

from sirfshampoo.combiner import TensorCombiner


def get_batch_size(inputs: Tuple[Tensor, ...]) -> int:
    """Determine the batch size from input tensors to a neural network.

    Args:
        inputs: The input tensors passed to the `forward` of a neural network.

    Returns:
        The batch size.
    """
    return inputs[0].shape[0]


class SIRFShampoo(Optimizer):
    """Structured inverse-free and root-free Shampoo optimizer."""

    def __init__(
        self,
        model: Module,
        params: Optional[Union[List[Parameter], List[Dict[str, Any]]]] = None,
        beta1: float = 0.001,
        beta2: float = 0.01,
        alpha1: float = 0.9,
        kappa: float = 0.0,
        batch_size: Union[int, Callable[[Tuple[Tensor, ...]], int]] = get_batch_size,
        T: Union[int, Callable[[int], bool]] = 1,
        verbose_init: bool = False,
    ):
        """Set up the optimizer.

        Notation based on [Can We Remove the Square-Root in Adaptive Gradient
        Methods?](https://openreview.net/pdf?id=vuMD71R20q).

        Note:
            We overwrite the parameter groups such that parameters sharing a pre-
            conditioner (e.g. weight and bias of a linear layer if both parameters are
            in the same original parameter group). This simplifies the internal book-
            keeping when updating the pre-conditioner and parameters.

        Args:
            model: The model to optimize. The optimizer needs access to the model
                to figure out weights/biases of one layer.
            params: The parameters to optimize. If `None`, all parameters of the
                model are optimized. Default: `None`.
            beta1: Learning rate for the parameter update. Default: `0.001`.
            beta2: Learning rate for the preconditioner update. Default: `0.01`.
            alpha1: Momentum for the parameter update. Default: `0.9`.
            kappa: Weight decay. Default: `0.0`.
            batch_size: The batch size as integer or a callable from the input tensors
                of the neural network to the batch size (will be installed as pre-
                forward hook). If not specified, we detect the batch size by using the
                first input tensors leading dimension.
            T: The pre-conditioner update frequency as integer or callable from the
                optimizer's global step to a boolean that is `True` if the pre-
                conditioner should be updated at that iteration. Default: `1`.
            verbose_init: Whether to print information at initialization, i.e. how
                parameters are grouped and what pre-conditioners are used.
                Default: `False`.
        """
        defaults = dict(beta1=beta1, beta2=beta2, alpha1=alpha1, kappa=kappa, T=T)

        if params is None:
            params = [p for p in model.parameters() if p.requires_grad]
        super().__init__(params, defaults)

        self.model = model
        self.global_step = 0

        # batch size detection
        if callable(batch_size):
            # install as module hook that updates the batch size in every forward pass
            self.batch_size = None

            def hook(module: Module, inputs: Tuple[Tensor, ...]):
                """Forward hook to store the current batch size in the optimizer.

                Args:
                    module: The module that is called.
                    inputs: The input tensors to the module.
                """
                self.batch_size = batch_size(inputs)

            self.batch_size_handle = model.register_forward_pre_hook(hook)
        else:
            self.batch_size = batch_size
            self.batch_size_handle = None

        # we rewrite the original parameter groups and create new ones such that each
        # parameter group contains the parameters that are treated jointly with one
        # pre-conditioner. This simplifies book-keeping when updating the
        # pre-conditioner and taking a step.
        self._one_param_group_per_preconditioner()

        # The pre-conditioner for one group is a list of matrices (the Kronecker
        # factors). For a layer with 2d weight of shape `(D_out, D_in)`, the entries are
        # (C, K) from the paper where C is `(D_out, D_out)` and K is `(D_in, D_in)`.
        self.preconditioner: List[List[Tensor]] = self._initialize_preconditioner()

        if verbose_init:
            self.print_group_info()

    def step(self, closure: Optional[Callable] = None) -> None:
        """Perform a single optimization step.

        Args:
            closure: Not supported. Default: `None`.

        Raises:
            NotImplementedError: If `closure` is not `None`.
        """
        if closure is not None:
            raise NotImplementedError("Closure is not supported.")

        for group_idx, _ in enumerate(self.param_groups):
            self._step(group_idx)

        self.global_step += 1

    def print_group_info(self) -> None:
        """Print information about the parameter groups and pre-conditioners."""
        param_to_names = {p.data_ptr(): n for n, p in self.model.named_parameters()}
        print("Parameter groups:")
        for i, group in enumerate(self.param_groups):
            param_names = [param_to_names[p.data_ptr()] for p in group["params"]]
            other = {k: v for k, v in group.items() if k != "params"}
            prec_shapes = [(str(s) for s in p.shape) for p in self.preconditioner[i]]
            prec_structure = ["dense" for _ in prec_shapes]
            prec_desc = [
                f"{'x'.join(shape)} ({structure})"
                for shape, structure in zip(prec_shapes, prec_structure)
            ]
            print(
                f"Group {i}\n\t- Parameter names: {param_names}"
                f"\n\t-Pre-conditioner: {prec_desc}\n\t- Other: {other}"
            )

    def _get_current_batch_size(self) -> int:
        """Get the current batch size.

        Returns:
            The current batch size.

        Raises:
            RuntimeError: If the batch size is not an integer.
        """
        if isinstance(self.batch_size, int):
            return self.batch_size

        raise RuntimeError(f"Batch size is not an integer: {self.batch_size}")

    def _one_param_group_per_preconditioner(self) -> None:
        """Overwrite parameter groups so that a group's params share a pre-conditioner.

        Raises:
            ValueError: If the re-arranging process lost parameters.
        """
        all_params = sum((group["params"] for group in self.param_groups), [])
        all_ids = {p.data_ptr() for p in all_params}

        # find all layers that contain trainable parameters
        layers = [
            layer
            for layer in self.model.modules()
            if not list(layer.children())
            and list(layer.parameters())
            and any(
                p.requires_grad and p.data_ptr() in all_ids for p in layer.parameters()
            )
        ]

        # create list entries where each entry lists parameters that are treated jointly
        treat_jointly = []
        processed_ids = set()
        param_to_group = {
            p.data_ptr(): i
            for i, group in enumerate(self.param_groups)
            for p in group["params"]
        }

        for layer in layers:
            params = [
                p
                for p in layer.parameters()
                if p.requires_grad and p.data_ptr() in all_ids
            ]
            in_param_groups = len({param_to_group[p.data_ptr()] for p in params})

            # treat jointly if all have the same shape (e.g. weight+bias of norm layer)
            if {p.shape for p in params} == 1 and in_param_groups == 1:
                treat_jointly.append(params)
                processed_ids.update(p.data_ptr() for p in params)
            # treat jointly if first is a weight, second a bias of a linear/conv. layer
            elif (
                len(params) == 2
                and params[0].ndim in [2, 3, 4, 5]
                and params[1].ndim == 1
                and params[0].shape[0] == params[1].shape[0]
                and in_param_groups == 1
            ):
                treat_jointly.append(params)
                processed_ids.update(p.data_ptr() for p in params)
            # otherwise, treat each parameter separately
            else:
                for p in params:
                    treat_jointly.append([p])
                    processed_ids.add(p.data_ptr())

        if processed_ids != all_ids:
            raise ValueError("Parameter group rewriting lost parameters.")

        # create new parameter groups, one per pre-conditioner
        new_param_groups = []
        for params in treat_jointly:
            old_group = self.param_groups[param_to_group[params[0].data_ptr()]]
            new_param_groups.append({**old_group, "params": params})

        self.param_groups = new_param_groups

    def _initialize_preconditioner(self) -> List[List[Tensor]]:
        """Return preconditioner matrices initialized to identity.

        Data type and devices are inferred from the parameters.

        Returns:
            A list of preconditioner matrices, one list per parameter group.
        """
        preconditioners = []
        for group in self.param_groups:
            params = group["params"]
            (dtype,) = {p.dtype for p in params}
            (device,) = {p.device for p in params}
            kwargs = {"dtype": dtype, "device": device}
            dims = TensorCombiner.group(params).shape
            preconditioners.append([eye(d, **kwargs) for d in dims])

        return preconditioners

    def _step(self, group_idx):
        """Perform a single optimization step for a group.

        Args:
            group_idx: The index of the group in `self.param_groups`.
        """
        self._update_preconditioner(group_idx)
        updates = self._precondition_gradient(group_idx)

        group = self.param_groups[group_idx]
        params = group["params"]
        beta1 = group["beta1"]
        alpha1 = group["alpha1"]
        kappa = group["kappa"]

        for p, p_step in zip(params, updates):
            # add weight decay
            if kappa != 0.0:
                p_step.add_(p.data, alpha=kappa)

            # momentum on previous updates
            if alpha1 != 0:
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = zeros_like(p.data)

                param_state["momentum_buffer"].mul_(alpha1).add_(p_step)
                p_step = param_state["momentum_buffer"]

            p.data.add_(p_step, alpha=-beta1)

    def _update_preconditioner(self, group_idx: int) -> None:
        """Update the preconditioner of a group.

        Args:
            group_idx: The index of the group in `self.param_groups`.

        Raises:
            NotImplementedError: If the preconditioner does not have 2 factors.
        """
        group = self.param_groups[group_idx]

        # maybe skip the update depending on the update interval/schedule
        T = group["T"]
        skip = not T(self.global_step) if callable(T) else self.global_step % T != 0
        if skip:
            return

        prec = self.preconditioner[group_idx]
        if len(prec) != 2:
            raise NotImplementedError("Only pre-conditioners with 2 factors supported.")
        C, K = prec
        dim_K, dim_C = K.shape[0], C.shape[0]
        G = TensorCombiner.group([p.grad for p in group["params"]])

        gamma = 1  # moving average, not sum
        beta2 = group["beta2"]
        B = self._get_current_batch_size()

        CT_G_K = C.T @ G @ K  # shared between updates

        # update C (first-order truncation of matrix exponential)
        C_exp_arg = B * CT_G_K @ CT_G_K.T - dim_K * gamma * eye(
            dim_C, device=C.device, dtype=C.dtype
        )
        C.add_(C_exp_arg, alpha=-beta2 / (2 * dim_K))

        # update K (first-order truncation of matrix exponential)
        K_exp_arg = B * CT_G_K.T @ CT_G_K - dim_C * gamma * eye(
            dim_K, device=K.device, dtype=K.dtype
        )
        K.add_(K_exp_arg, alpha=-beta2 / (2 * dim_C))

    def _precondition_gradient(self, group_idx: int) -> List[Tensor]:
        """Multiply the pre-conditioner onto the gradient for a parameter group.

        Args:
            group_idx: The index of the group in `self.param_groups`.

        Returns:
            The preconditioned gradient. Has the same structure as the `'params'`
            entry of the parameter group.

        Raises:
            NotImplementedError: If the preconditioner does not have 2 factors.
        """
        group = self.param_groups[group_idx]
        params = group["params"]

        prec = self.preconditioner[group_idx]
        if len(prec) != 2:
            raise NotImplementedError("Only pre-conditioners with 2 factors supported.")

        C, K = prec
        G = TensorCombiner.group([p.grad for p in params])
        G_preconditioned = (C @ C.T) @ G @ (K @ K.T)

        return TensorCombiner.ungroup(G_preconditioned, [p.shape for p in params])
