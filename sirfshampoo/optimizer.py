"""Implementation of structured inverse-, root-free Shampoo."""

from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from singd.structures.base import StructuredMatrix
from singd.structures.dense import DenseMatrix
from torch import Tensor, zeros_like
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
    """Structured inverse-free and root-free Shampoo optimizer.

    Attributes:
        SUPPORTED_STRUCTURES: A dictionary mapping structure names to the respective
            classes of structured matrices that can be used for the pre-conditioner.
            Currently, only `'dense'` is supported.

    TODO Add preconditioner_dtype, default value is the same as the parameter
    TODO Implement update rule for Nd tensors
    """

    SUPPORTED_STRUCTURES: Dict[str, Type[StructuredMatrix]] = {"dense": DenseMatrix}

    def __init__(
        self,
        model: Module,
        params: Optional[Union[List[Parameter], List[Dict[str, Any]]]] = None,
        lr: float = 0.001,  # beta1 in the paper
        beta2: float = 0.01,
        alpha1: float = 0.9,
        alpha2: float = 0.5,
        lam: float = 0.001,
        kappa: float = 0.0,
        batch_size: Union[int, Callable[[Tuple[Tensor, ...]], int]] = get_batch_size,
        T: Union[int, Callable[[int], bool]] = 1,
        structures: Union[str, Tuple[str, ...]] = "dense",
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
            lr: Learning rate for the parameter update. Default: `0.001`.
            beta2: Learning rate for the preconditioner update. Default: `0.01`.
            alpha1: Momentum for the parameter update. Default: `0.9`.
            alpha2: Riemannian momentum on the pre-conditioners. Default `0.5`.
            lam: Damping for the pre-conditioner update. Default: `0.001`.
            kappa: Weight decay. Default: `0.0`.
            batch_size: The batch size as integer or a callable from the input tensors
                of the neural network to the batch size (will be installed as pre-
                forward hook). If not specified, we detect the batch size by using the
                first input tensors leading dimension.
            T: The pre-conditioner update frequency as integer or callable from the
                optimizer's global step to a boolean that is `True` if the pre-
                conditioner should be updated at that iteration. Default: `1`.
            structures: A string specifying the structure to use for the
                pre-conditioner's Kronecker factors. If a tuple of strings is specified,
                it must be of the same length as the number of Kronecker factors.
                Supported choices are `'dense'`.
            verbose_init: Whether to print information at initialization, i.e. how
                parameters are grouped and what pre-conditioners are used.
                Default: `False`.
        """
        defaults = dict(
            lr=lr,
            beta2=beta2,
            alpha1=alpha1,
            alpha2=alpha2,
            lam=lam,
            kappa=kappa,
            T=T,
            structures=structures,
        )

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
        self._verify_hyperparameters()

        # The pre-conditioner for one group is a list of matrices (the Kronecker
        # factors). For a layer with 2d weight of shape `(D_out, D_in)`, the entries are
        # (C, K) from the paper where C is `(D_out, D_out)` and K is `(D_in, D_in)`.
        self.preconditioner: List[List[StructuredMatrix]] = (
            self._initialize_preconditioner("identity")
        )
        # same for the momenta, i.e (m_C, m_K) from the paper for a 2d weight
        self.preconditioner_momenta: List[List[StructuredMatrix]] = (
            self._initialize_preconditioner("zero")
        )

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
            precs = self.preconditioner[i]
            shapes = [(str(s) for s in p.to_dense().shape) for p in precs]
            structures = [p.__class__.__name__ for p in precs]
            prec_desc = [
                f"{'x'.join(shape)} ({structure})"
                for shape, structure in zip(shapes, structures)
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
            RuntimeError: If the batch size is negative or not an integer.
        """
        if isinstance(self.batch_size, int) and self.batch_size > 0:
            return self.batch_size

        raise RuntimeError(f"Batch size is not a positive integer: {self.batch_size}.")

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

    def _verify_hyperparameters(self):  # noqa: C901
        """Verify that the hyperparameters are valid.

        Raises:
            ValueError: If a hyperparameter is invalid.
        """
        for beta in ["lr", "beta2"]:
            values = {group[beta] for group in self.param_groups}
            if any(val <= 0 for val in values):
                raise ValueError(f"{beta}-s must be non-negative. Got: {values}.")

        for alpha in ["alpha1", "alpha2"]:
            values = {group[alpha] for group in self.param_groups}
            if not all(0 <= val < 1 for val in values):
                raise ValueError(f"{alpha}-s must be in [0; 1). Got: {values}.")

        lambdas = {group["lam"] for group in self.param_groups}
        if any(lam < 0 for lam in lambdas):
            raise ValueError(f"lam-s must be non-negative. Got: {lambdas}.")

        kappa = {group["kappa"] for group in self.param_groups}
        if any(k < 0 for k in kappa):
            raise ValueError(f"kappa-s must be non-negative. Got: {kappa}.")

        T = {group["T"] for group in self.param_groups}
        if not all((isinstance(t, int) and t > 0) or callable(t) for t in T):
            raise ValueError(f"T-s must be positive integers or callables. Got: {T}.")

        structures = set()
        for group in self.param_groups:
            struct = group["structures"]
            if isinstance(struct, str):
                structures.add(struct)
            else:
                structures.update(set(struct))
        if any(struct not in self.SUPPORTED_STRUCTURES for struct in structures):
            raise ValueError(
                "Unsupported structure. Supported: "
                + f"{list(self.SUPPORTED_STRUCTURES.keys())}. Got {structures}."
            )

    def _initialize_preconditioner(self, method: str) -> List[List[StructuredMatrix]]:
        """Return preconditioner matrices initialized to identity.

        Data type and devices are inferred from the parameters.

        Args:
            method: The method to use for preconditioning.
                Must either be `'identity` or `'zero'`.

        Returns:
            A list of preconditioner matrices, one list per parameter group.

        Raises:
            ValueError: If the method is not supported.
            ValueError: If number of structures does not match number of factors.
        """
        preconditioners = []
        for group in self.param_groups:
            params = group["params"]
            (dtype,) = {p.dtype for p in params}
            (device,) = {p.device for p in params}
            kwargs = {"dtype": dtype, "device": device}
            dims = TensorCombiner.group(params).shape

            # obtain constructors of structured matrices
            structures = group["structures"]
            if isinstance(structures, str):
                structures = len(dims) * [structures]
            elif len(structures) != len(dims):
                raise ValueError(
                    f"Number of structures ({len(structures)}) must match number"
                    + f"of Kronecker factors ({len(dims)})."
                )
            classes = [self.SUPPORTED_STRUCTURES[struct] for struct in structures]

            if method == "identity":
                preconditioners.append(
                    [cls.eye(d, **kwargs) for cls, d in zip(classes, dims)]
                )
            elif method == "zero":
                preconditioners.append(
                    [cls.zeros(d, **kwargs) for cls, d in zip(classes, dims)]
                )
            else:
                raise ValueError(
                    f"Unsupported preconditioning method: {method}."
                    + " Supported methods are 'identity' and 'zero'."
                )

        return preconditioners

    def _step(self, group_idx):
        """Perform a single optimization step for a group.

        Args:
            group_idx: The index of the group in `self.param_groups`.
        """
        self._update_preconditioner(group_idx)
        # TODO We could incorporate a scaling trick here, and then return
        # the scaling and incorporate it into the final update step
        updates = self._precondition_gradient(group_idx)

        group = self.param_groups[group_idx]
        params = group["params"]
        lr = group["lr"]
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

            p.data.add_(p_step, alpha=-lr)

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
        prec_mom = self.preconditioner_momenta[group_idx]
        if len(prec) != 2 or len(prec_mom) != 2:
            raise NotImplementedError("Only pre-conditioners with 2 factors supported.")
        C, K = prec
        m_C, m_K = prec_mom
        G = TensorCombiner.group([p.grad for p in group["params"]])
        dim_C, dim_K = G.shape

        gamma = 1  # moving average, not sum
        alpha2 = group["alpha2"]
        beta2 = group["beta2"]
        lam = group["lam"]
        B = self._get_current_batch_size()

        # scale the preconditioner matrices to make the operations numerically stable
        C_scaled = C * (1 / sqrt(dim_C))
        K_scaled = K * (1 / sqrt(dim_K))

        CT_G_K = C_scaled.rmatmat(K_scaled.rmatmat(G.T).T)  # shared between updates
        CT_C = C_scaled.from_inner()
        KT_K = K_scaled.from_inner()

        # Update Riemannian momentum on C and K
        m_C_step = C.from_dense(CT_G_K @ CT_G_K.T).mul_(B)
        m_C_step.add_(CT_C, alpha=lam * KT_K.average_trace() * dim_K)
        m_C_step.diag_add_(-gamma / dim_C)

        m_C.mul_(alpha2)
        m_C.add_(m_C_step, alpha=(1 - alpha2) * dim_C / 2)

        m_K_step = K.from_dense(CT_G_K.T @ CT_G_K).mul_(B)
        m_K_step.add_(KT_K, alpha=lam * CT_C.average_trace() * dim_C)
        m_K_step.diag_add_(-gamma / dim_K)

        m_K.mul_(alpha2)
        m_K.add_(m_K_step, alpha=(1 - alpha2) * dim_K / 2)

        # matrix norm scaling
        norm_C = C.frobenius_norm().clamp(min=1.0)
        norm_K = K.frobenius_norm().clamp(min=1.0)

        # update C, K (first-order truncation of matrix exponential)
        C.add_(m_C, alpha=-beta2 / norm_C)
        K.add_(m_K, alpha=-beta2 / norm_K)

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
        # multiply from the left with C @ C.T
        G = C @ C.rmatmat(G)
        # multiply from the right with K.T @ K
        G = (K.rmatmat(K @ G.T)).T

        return TensorCombiner.ungroup(G, [p.shape for p in params])
