"""Implementation of structured inverse-, root-free Shampoo."""

from copy import deepcopy
from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from numpy import array
from singd.structures.base import StructuredMatrix
from singd.structures.dense import DenseMatrix
from torch import Tensor, dtype, tensordot, zeros_like
from torch.nn import Module, Parameter
from torch.optim import Optimizer

from sirfshampoo.combiner import TensorCombiner
from sirfshampoo.utils import tensormatdot


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
        STATE_ATTRIBUTES: Attributes that belong to the optimizer's state but are
            not stored inside the `self.state` attribute. They will be saved
            and restored when the optimizer is check-pointed (by calling
            [`.state_dict()`](\
https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.html) and
            [`.load_state_dict()`](\
https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html)).

    TODO Support more structures
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
        structures: Union[str, Dict[int, Union[str, Tuple[str, ...]]]] = "dense",
        preconditioner_dtypes: Optional[
            Union[dtype, Dict[int, Union[None, dtype, Tuple[Union[None, dtype], ...]]]]
        ] = None,
        verbose_init: bool = False,
    ):
        """Set up the optimizer.

        Notation based on [Can We Remove the Square-Root in Adaptive Gradient
        Methods?](https://openreview.net/pdf?id=vuMD71R20q).

        Note:
            We currently treat weights and biases of a layer independently, because this
            this the approach in the paper. It will be more memory- and time-efficient
            to combine parameters, e.g. to append the bias of a linear layer as last
            column to the weight matrix. We will make such parameter groupings fully
            customizable in the future.

        Note:
            We rewrite the parameter groups such that parameters sharing a pre-
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
            structures: Specification of which structures the preconditioner matrices
                should use. There are multiple ways to specify this:
                - If a single string, every of the `N` factors of an `N`d tensor's
                  preconditioner will use the same structure specified by the string.
                - If specified as dictionary, each key represents the dimension of a
                  preconditioned tensor and its value specifies the structure as string
                  or tuple. E.g. `{1: 'dense', 2: ('dense', 'diagonal'), 3: 'diagonal'}`
                  means that 1d tensors will be predonditioned with a single dense
                  Kronecker factor, 2d tensors with a dense and a diagonal factor, and
                  3d tensors with three diagonal factors.
                Supported choices are `'dense'`.
            preconditioner_dtypes: The data type to use for the pre-conditioner. There
                are multiple ways to specify this and the format is identical to that of
                `structures`. E.g. `{1: bfloat16, 2: (float32, float16), 3: float32}`
                means that 1d tensors will use `bfloat16`, 2d tensors will use `float32`
                for the first and `float16` for the second factor, and 3d tensors will
                use `float32` for all factors. If `None`, the parameter's data type will
                be used. Default: `None`.
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
            preconditioner_dtypes=preconditioner_dtypes,
        )

        if params is None:
            params = [p for p in model.parameters() if p.requires_grad]
        super().__init__(params, defaults)

        self.param_to_names = {p.data_ptr(): n for n, p in model.named_parameters()}
        self.global_step = 0

        # batch size detection
        if callable(batch_size):
            # install as module hook that updates the batch size in every forward pass
            self.batch_size_valid = self.global_step
            self.batch_size = 0

            def hook(module: Module, inputs: Tuple[Tensor, ...]):
                """Forward hook to accumulate the batch size in the optimizer.

                Args:
                    module: The module that is called.
                    inputs: The input tensors to the module.
                """
                # batch size is outdated because optimizer has stepped
                if self.batch_size_valid != self.global_step:
                    self.batch_size_valid = self.global_step
                    self.batch_size = 0

                # do not accumulate batch size during evaluation
                if module.training:
                    self.batch_size += batch_size(inputs)

            model.register_forward_pre_hook(hook)
        else:
            self.batch_size = batch_size
            self.batch_size_valid = "always"

        # we rewrite the original parameter groups and create new ones such that each
        # parameter group contains the parameters that are treated jointly with one
        # pre-conditioner. This simplifies book-keeping when updating the
        # pre-conditioner and taking a step.
        self._one_param_group_per_preconditioner()
        # convert structure and dtype arguments into tuples
        self._standardize_structures()
        self._standardize_preconditioner_dtypes()
        self._verify_hyperparameters()

        # The pre-conditioner for one group is a list of matrices (the Kronecker
        # factors). For a layer with 2d weight of shape `(D_out, D_in)`, the entries are
        # (C, K) from the paper where C is `(D_out, D_out)` and K is `(D_in, D_in)`.
        self.preconditioner: List[List[StructuredMatrix]] = (
            self._initialize_preconditioner("identity")
        )
        # same for the momenta, i.e. (m_C, m_K) from the paper for a 2d weight
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
        print("Parameter groups:")
        for i, group in enumerate(self.param_groups):
            param_names = [self.param_to_names[p.data_ptr()] for p in group["params"]]
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
                f"\n\t- Pre-conditioner: {prec_desc}\n\t- Other: {other}"
            )

    def _one_param_group_per_preconditioner(self) -> None:
        """Overwrite param groups so that one group shares a pre-conditioner."""
        params = sum((group["params"] for group in self.param_groups), [])

        # create list entries where each entry lists parameters that are treated jointly
        # treat each parameter with an independent pre-conditioner
        treat_jointly = [[p] for p in params]

        # create new parameter groups, one per pre-conditioner
        param_to_group = {
            p.data_ptr(): i
            for i, group in enumerate(self.param_groups)
            for p in group["params"]
        }
        new_param_groups = []
        for params in treat_jointly:
            old_group = self.param_groups[param_to_group[params[0].data_ptr()]]

            # If T is a class with internal state we do not want this state to be shared
            # between parameter groups because otherwise calling T of one parameter
            # group might have side effects on other groups that use the same T.
            # Hence, create an independent copy if T is callable.
            T = deepcopy(old_group["T"]) if callable(old_group["T"]) else old_group["T"]

            new_param_groups.append({**old_group, "params": params, "T": T})

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

        for group in self.param_groups:
            params = group["params"]
            N = TensorCombiner().group(params).ndim

            structures = group["structures"]
            if len(structures) != N:
                raise ValueError(
                    f"Number of structures ({len(structures)}) does not match "
                    f"number of Kronecker matrices ({N})."
                )

            dtypes = group["preconditioner_dtypes"]
            if len(dtypes) != N:
                raise ValueError(
                    f"Number of data types ({len(dtypes)}) does not match "
                    f"number of Kronecker matrices ({N})."
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
            RuntimeError: If the number of structures, data types, and dimensions do not
                match.
        """
        preconditioners = []
        for group in self.param_groups:
            classes = [
                self.SUPPORTED_STRUCTURES[struct] for struct in group["structures"]
            ]

            params = group["params"]
            (dev,) = {p.device for p in params}
            dtypes = group["preconditioner_dtypes"]
            kwargs = [{"dtype": dt, "device": dev} for dt in dtypes]

            dims = TensorCombiner.group(params).shape

            if not len(dtypes) == len(classes) == len(dims):
                raise RuntimeError(
                    "Number of structures, data dtypes, and dimensions do not match."
                )

            if method == "identity":
                preconditioners.append(
                    [cls.eye(d, **kw) for cls, d, kw in zip(classes, dims, kwargs)]
                )
            elif method == "zero":
                preconditioners.append(
                    [cls.zeros(d, **kw) for cls, d, kw in zip(classes, dims, kwargs)]
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
        """
        group = self.param_groups[group_idx]

        # maybe skip the update depending on the update interval/schedule
        T = group["T"]
        skip = not T(self.global_step) if callable(T) else self.global_step % T != 0
        if skip:
            return

        # hyper-parameters for the update
        gamma = 1  # moving average, not sum
        alpha2 = group["alpha2"]
        beta2 = group["beta2"]
        lam = group["lam"]

        # arrange the gradients into the tensor that is pre-conditioned
        G = TensorCombiner.group([p.grad for p in group["params"]])
        dims = G.shape
        dtypes = group["preconditioner_dtypes"]

        Ks = self.preconditioner[group_idx]
        m_Ks = self.preconditioner_momenta[group_idx]
        (N,) = {len(Ks), len(m_Ks), len(dims), len(dtypes)}

        # 1) PRE-COMPUTE QUANTITIES THAT ARE SHARED BY ALL UPDATES
        # Multiply each Kronecker factor onto the gradient axis it preconditions
        GK = G * sqrt(self.batch_size)
        KTKs, Tr_KTKs = [], []
        for n, K, dt, dim in zip(range(N), Ks, dtypes, dims):
            GK = GK.to(dt)
            # NOTE To make the operations more numerically stable, we scale K for
            # (i) multiplication onto G, (ii) computing its self-outer product KᵀK,
            # and (iii) its trace Tr(KᵀK)
            K_scaled = K * (1 / sqrt(dim))
            GK = tensormatdot(GK, K_scaled, n, transpose=True)
            KTK = K_scaled.from_inner()
            KTKs.append(KTK)
            # NOTE Deliberately convert to python float here to simplify computing the
            # trace products in the update. This costs GPU-CPU synchronization.
            Tr_KTKs.append(KTK.average_trace().item())

        # convert to numpy array so we can use list slicing syntax
        Tr_KTKs, dims = array(Tr_KTKs), array(dims)

        # 2) UPDATE THE KRONECKER FACTORS
        # NOTE `GK`, `KT_K`, and `Tr_KTK` have scalings to improve numerical stability.
        # Therefore, the update reads differently to the version in the paper.
        for n, dt, m_K, K in zip(range(N), dtypes, m_Ks, Ks):
            not_n = list(range(n)) + list(range(n + 1, N))
            GK = GK.to(dt)
            m_K_step = KTKs.pop(0).mul_(lam / 2 * Tr_KTKs[not_n].prod() * dims.prod())
            m_K_step.add_(
                K.from_dense(tensordot(GK, GK, dims=(not_n, not_n))), alpha=dims[n] / 2
            )
            m_K_step.diag_add_(-gamma / 2)

            # Update Riemannian momentum on K_n
            m_K.mul_(alpha2)
            m_K.add_(m_K_step, alpha=1 - alpha2)

            # update K_n (first-order truncation of matrix exponential)
            K.add_(K @ m_K, alpha=-beta2 / m_K.frobenius_norm().clamp(min=1.0))

    def _precondition_gradient(self, group_idx: int) -> List[Tensor]:
        """Multiply the pre-conditioner onto the gradient for a parameter group.

        Args:
            group_idx: The index of the group in `self.param_groups`.

        Returns:
            The preconditioned gradient. Has the same structure as the `'params'`
            entry of the parameter group.
        """
        group = self.param_groups[group_idx]
        params = group["params"]
        G = TensorCombiner.group([p.grad for p in params])
        dtypes = group["preconditioner_dtypes"]
        Ks = self.preconditioner[group_idx]
        (N,) = {len(Ks), len(dtypes), G.ndim}

        # NOTE To improve numerical stability, we scale each Kronecker factor
        # before multiplying it onto the gradient. We deliberately use `item`
        # here because each `K` might have its individual data type
        scales = array(
            [K.infinity_vector_norm().sqrt().clamp(min=1.0).item() for K in Ks]
        )

        for n, dt, K, scale in zip(range(N), dtypes, Ks, scales):
            K_scaled = K * (1 / scale)
            # multiply K Kᵀ onto axis n
            G = tensormatdot(G.to(dt), K_scaled, n, transpose=True)
            G = tensormatdot(G, K_scaled, n)

        # correct the scaling
        G.mul_((scales**2).prod())

        return TensorCombiner.ungroup(G, [p.shape for p in params])

    def _standardize_structures(self):
        """Standardize the values for structures in parameter groups.

        Rewrites the `'structures'` entries to be tuples, each entry specifying the
        structures of a Kronecker factor.

        Raises:
            ValueError: If the structures were specified incorrectly.
        """
        for group in self.param_groups:
            structures = group["structures"]
            params = group["params"]

            # number of Kronecker factors
            N = TensorCombiner.group(params).ndim

            # overwrite structures in parameter group
            if isinstance(structures, str):
                structures = N * (structures,)
            elif isinstance(structures, Dict) and N in structures:
                structure = structures[N]
                if isinstance(structure, str):
                    structures = N * (structure,)
                elif len(structure) == N and all(isinstance(s, str) for s in structure):
                    structures = tuple(structure)
                else:
                    raise ValueError(
                        f"Invalid structure specification for N={N}: {structure}."
                    )
            else:
                raise ValueError(
                    f"Invalid structure specification for N={N}: {structures}."
                )

            group["structures"] = structures

    def _standardize_preconditioner_dtypes(self):
        """Standardize the values for preconditioner data types in parameter groups.

        Rewrites the `'preconditioner_dtypes'` entries to be tuples, each entry
        specifying the data type of a Kronecker factor.

        Raises:
            ValueError: If the data types were specified incorrectly.
        """
        for group in self.param_groups:
            dtypes = group["preconditioner_dtypes"]
            params = group["params"]

            # number of Kronecker factors
            N = TensorCombiner.group(params).ndim

            # detect data types and overwrite entry in parameter groups
            if isinstance(dtypes, dtype) or dtypes is None:
                dtypes = N * (dtypes,)
            elif isinstance(dtypes, Dict) and N in dtypes:
                dt = dtypes[N]
                if isinstance(dt, dtype) or dt is None:
                    dtypes = N * (dt,)
                elif len(dt) == N and all(
                    d is None or isinstance(d, dtype) for d in dt
                ):
                    dtypes = tuple(dt)
                else:
                    raise ValueError(
                        f"Invalid dtype specification inside dict for N={N}: {dt}."
                    )
            else:
                raise ValueError(f"Invalid dtype specification for N={N}: {dtypes}.")

            if None in dtypes:
                (default_dt,) = {p.dtype for p in params}
                dtypes = tuple(default_dt if dt is None else dt for dt in dtypes)

            group["preconditioner_dtypes"] = dtypes

    STATE_ATTRIBUTES: List[str] = [
        "global_step",
        "batch_size",
        "batch_size_valid",
        "preconditioner",
        "preconditioner_momenta",
    ]

    def state_dict(self) -> Dict[str, Any]:
        """Return a save-able state of the optimizer.

        Returns:
            A dictionary containing the optimizer state.
        """
        state_dict = super().state_dict()

        for name in self.STATE_ATTRIBUTES:
            assert name not in state_dict.keys()
            state_dict[name] = getattr(self, name)

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load an optimizer state.

        Args:
            state_dict: A dictionary containing a valid state obtained from this
                class's `.state_dict()` method.
        """
        attributes = {name: state_dict.pop(name) for name in self.STATE_ATTRIBUTES}
        super().load_state_dict(state_dict)

        for name, value in attributes.items():
            setattr(self, name, value)
