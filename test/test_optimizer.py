"""Test `sirfshampoo.optimizer` module."""

from collections import OrderedDict
from test.utils import DEVICE_IDS, DEVICES
from typing import Callable, Dict, Optional, Tuple, Union

from pytest import mark, raises
from torch import (
    Tensor,
    bfloat16,
    device,
    dtype,
    float16,
    float32,
    manual_seed,
    rand,
    zeros,
)
from torch.nn import Linear, Module, MSELoss, Parameter, ReLU, Sequential, Sigmoid
from torch.optim.lr_scheduler import StepLR

from sirfshampoo.combiner import LinearWeightBias, PerParameter, PreconditionerGroup
from sirfshampoo.optimizer import DEFAULT_COMBINE_PARAMS, SIRFShampoo


def nested_network(D_in: int, D_hidden: int, D_out: int) -> Sequential:
    """Create a nested network for testing.

    Args:
        D_in: Input dimension.
        D_hidden: Hidden dimension.
        D_out: Output dimension.

    Returns:
        Sequential: A nested network.
    """
    inner = Sequential(
        OrderedDict(
            {
                "linear": Linear(D_hidden, D_hidden, bias=False),
                "sigmoid": Sigmoid(),
            }
        )
    )
    return Sequential(
        OrderedDict(
            {
                "linear1": Linear(D_in, D_hidden),
                "relu1": ReLU(),
                "inner": inner,
                "linear2": Linear(D_hidden, D_out),
            }
        )
    )


def test__one_param_group_per_preconditioner():
    """Test creation of parameter groups (one per pre-conditioner)."""
    manual_seed(0)
    D_in, D_hidden, D_out = 5, 4, 3
    model = nested_network(D_in, D_hidden, D_out)
    defaults = {
        "lr": 0.001,
        "beta2": 0.01,
        "alpha1": 0.9,
        "alpha2": 0.5,
        "kappa": 0.0,
        "T": 1,
        "lam": 0.001,
    }

    # one parameter group
    optimizer = SIRFShampoo(model, verbose_init=True)
    assert len(optimizer.param_groups) == len(list(model.parameters()))
    assert optimizer.param_groups == [
        {
            "params": [model.linear1.weight],
            **defaults,
            "preconditioner_dtypes": 2 * (float32,),
            "structures": 2 * ("dense",),
            "combine_params": optimizer.param_groups[0]["combine_params"],
        },
        {
            "params": [model.linear1.bias],
            **defaults,
            "preconditioner_dtypes": (float32,),
            "structures": ("dense",),
            "combine_params": optimizer.param_groups[1]["combine_params"],
        },
        {
            "params": [model.inner.linear.weight],
            **defaults,
            "preconditioner_dtypes": 2 * (float32,),
            "structures": 2 * ("dense",),
            "combine_params": optimizer.param_groups[2]["combine_params"],
        },
        {
            "params": [model.linear2.weight],
            **defaults,
            "preconditioner_dtypes": 2 * (float32,),
            "structures": 2 * ("dense",),
            "combine_params": optimizer.param_groups[3]["combine_params"],
        },
        {
            "params": [model.linear2.bias],
            **defaults,
            "preconditioner_dtypes": (float32,),
            "structures": ("dense",),
            "combine_params": optimizer.param_groups[4]["combine_params"],
        },
    ]

    # two parameter groups (sub-set of parameters), one with modified defaults
    param_groups = [
        {
            "params": [model.linear1.weight, model.linear1.bias, model.linear2.bias],
            **defaults,
            "alpha1": 0.5,  # change alpha1 to verify it is passed through the overwrite
        },
        {"params": [model.inner.linear.weight]},
    ]
    optimizer = SIRFShampoo(model, params=param_groups, verbose_init=True)
    assert len(optimizer.param_groups) == 4
    assert optimizer.param_groups == [
        {
            "params": [model.linear1.weight],
            **defaults,
            "preconditioner_dtypes": 2 * (float32,),
            "structures": 2 * ("dense",),
            "alpha1": 0.5,
            "combine_params": optimizer.param_groups[0]["combine_params"],
        },
        {
            "params": [model.linear1.bias],
            **defaults,
            "preconditioner_dtypes": (float32,),
            "structures": ("dense",),
            "alpha1": 0.5,
            "combine_params": optimizer.param_groups[1]["combine_params"],
        },
        {
            "params": [model.linear2.bias],
            **defaults,
            "alpha1": 0.5,
            "preconditioner_dtypes": (float32,),
            "structures": ("dense",),
            "combine_params": optimizer.param_groups[2]["combine_params"],
        },
        {
            "params": [model.inner.linear.weight],
            **defaults,
            "preconditioner_dtypes": 2 * (float32,),
            "structures": 2 * ("dense",),
            "combine_params": optimizer.param_groups[3]["combine_params"],
        },
    ]

    # two parameter groups (sub-set of parameters), one with modified defaults,
    # and parameters in one layer split into different groups such that they cannot
    # be treated jointly
    param_groups = [
        {
            "params": [model.linear1.weight, model.linear2.bias],
            **defaults,
            "alpha1": 0.5,  # change alpha1 to verify it is passed through the overwrite
        },
        {"params": [model.linear1.bias, model.inner.linear.weight]},
    ]
    optimizer = SIRFShampoo(model, params=param_groups, verbose_init=True)
    assert len(optimizer.param_groups) == 4
    assert optimizer.param_groups == [
        {
            "params": [model.linear1.weight],
            **defaults,
            "alpha1": 0.5,
            "preconditioner_dtypes": 2 * (float32,),
            "structures": 2 * ("dense",),
            "combine_params": optimizer.param_groups[0]["combine_params"],
        },
        {
            "params": [model.linear2.bias],
            **defaults,
            "alpha1": 0.5,
            "structures": ("dense",),
            "preconditioner_dtypes": (float32,),
            "combine_params": optimizer.param_groups[1]["combine_params"],
        },
        {
            "params": [model.linear1.bias],
            **defaults,
            "structures": ("dense",),
            "preconditioner_dtypes": (float32,),
            "combine_params": optimizer.param_groups[2]["combine_params"],
        },
        {
            "params": [model.inner.linear.weight],
            **defaults,
            "preconditioner_dtypes": 2 * (float32,),
            "structures": 2 * ("dense",),
            "combine_params": optimizer.param_groups[3]["combine_params"],
        },
    ]

    # different data types for pre-conditioner
    param_groups = [
        {
            "params": [model.linear1.weight],
            **defaults,
            "preconditioner_dtypes": {2: (float16, bfloat16)},
        },
    ]
    optimizer = SIRFShampoo(model, params=param_groups, verbose_init=True)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups == [
        {
            "params": [model.linear1.weight],
            **defaults,
            "preconditioner_dtypes": (float16, bfloat16),
            "structures": 2 * ("dense",),
            "combine_params": optimizer.param_groups[0]["combine_params"],
        }
    ]

    # one parameter group, treat weights and biases of linear layers jointly
    # different data types for pre-conditioner
    param_groups = [
        {"params": list(model.parameters()), **defaults},
    ]
    optimizer = SIRFShampoo(
        model,
        params=param_groups,
        combine_params=(LinearWeightBias(), PerParameter()),
        verbose_init=True,
    )
    assert len(optimizer.param_groups) == 3
    assert optimizer.param_groups == [
        {
            "params": [model.linear1.weight, model.linear1.bias],
            **defaults,
            "preconditioner_dtypes": 2 * (float32,),
            "structures": 2 * ("dense",),
            "combine_params": optimizer.param_groups[0]["combine_params"],
        },
        {
            "params": [model.linear2.weight, model.linear2.bias],
            **defaults,
            "structures": 2 * ("dense",),
            "preconditioner_dtypes": 2 * (float32,),
            "combine_params": optimizer.param_groups[1]["combine_params"],
        },
        {
            "params": [model.inner.linear.weight],
            **defaults,
            "preconditioner_dtypes": 2 * (float32,),
            "structures": 2 * ("dense",),
            "combine_params": optimizer.param_groups[2]["combine_params"],
        },
    ]


T_CASES = [1, lambda step: step in [0, 3]]
T_CASE_IDS = ["every", "custom"]

SCHEDULE_LRS = [False, True]
SCHEDULE_LR_IDS = ["constant-lr", "scheduled-lr"]

PRECONDITIONER_DTYPES = [
    None,
    bfloat16,
    {1: float32, 2: None},
    {1: (float32,), 2: (float16, float32)},
    {1: float16, 2: (float16, bfloat16)},
]
PRECONDITIONER_DTYPE_IDS = [
    f"dtypes={dt}".replace("\n", "") for dt in PRECONDITIONER_DTYPES
]

STRUCTURES = list(SIRFShampoo.SUPPORTED_STRUCTURES.keys()) + [
    {1: "triltoeplitz", 2: ("dense", "diagonal")},  # dictionary format
]
STRUCTURE_IDS = [
    f"structures={structures}".replace("\n", "").replace(" ", "")
    for structures in STRUCTURES
]

ALPHA2S = [0.5, 0.0]
ALPHA2_IDS = [f"alpha2={alpha2}" for alpha2 in ALPHA2S]

COMBINE_PARAMS = [
    DEFAULT_COMBINE_PARAMS,
    (PerParameter(), PerParameter()),  # same rule supplied multiple times
    (LinearWeightBias(), PerParameter()),  # same rule supplied multiple times
]
COMBINE_PARAMS_IDS = [
    f"combine_params=({', '.join(str(c.__class__.__name__) for c in combine_params)})"
    for combine_params in COMBINE_PARAMS
]


@mark.parametrize("combine_params", COMBINE_PARAMS, ids=COMBINE_PARAMS_IDS)
@mark.parametrize("alpha2", ALPHA2S, ids=ALPHA2_IDS)
@mark.parametrize("structures", STRUCTURES, ids=STRUCTURE_IDS)
@mark.parametrize(
    "preconditioner_dtypes", PRECONDITIONER_DTYPES, ids=PRECONDITIONER_DTYPE_IDS
)
@mark.parametrize("schedule_lr", SCHEDULE_LRS, ids=SCHEDULE_LR_IDS)
@mark.parametrize("T", T_CASES, ids=T_CASE_IDS)
@mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
def test_step_integration(
    T: Union[int, Callable[[int], bool]],
    schedule_lr: bool,
    preconditioner_dtypes: Optional[
        Union[dtype, Dict[int, Union[None, dtype, Tuple[Union[None, dtype], ...]]]]
    ],
    structures: Union[str, Dict[int, Union[str, Tuple[str, ...]]]],
    alpha2: float,
    combine_params: Tuple[PreconditionerGroup, ...],
    dev: device,
):
    """Check the optimizer is able to take a couple of steps without erroring.

    Args:
        T: Pre-conditioner update schedule.
        schedule_lr: Whether to use a learning rate scheduler.
        preconditioner_dtypes: Pre-conditioner data types.
        structures: Pre-conditioner structures.
        alpha2: Riemannian momentum on the pre-conditioner matrices.
        combine_params: Rules for detecting parameters that are combined, then treated
            with one pre-conditioner.
        dev: Device to run the test on.
    """
    manual_seed(0)
    batch_size = 6
    D_in, D_hidden, D_out = 5, 4, 3
    model = nested_network(D_in, D_hidden, D_out).to(dev)
    loss_func = MSELoss().to(dev)
    X, y = rand(batch_size, D_in, device=dev), rand(batch_size, D_out, device=dev)

    optimizer = SIRFShampoo(
        model,
        lr=0.1,
        alpha2=alpha2,
        kappa=0.001,
        T=T,
        preconditioner_dtypes=preconditioner_dtypes,
        structures=structures,
        combine_params=combine_params,
    )
    num_steps = 5
    if schedule_lr:
        scheduler = StepLR(optimizer, num_steps // 2, gamma=0.9)

    losses = []
    for step in range(num_steps):
        verify_preconditioner_dtypes(optimizer)
        verify_preconditioner_structures(optimizer)
        optimizer.zero_grad()
        loss = loss_func(model(X), y)
        loss.backward()
        losses.append(loss.item())
        print(f"Step {step:03g}, Loss: {losses[-1]:.5f}")
        optimizer.step()
        if schedule_lr:
            scheduler.step()

    assert losses[0] > losses[-1]


def verify_preconditioner_dtypes(optimizer: SIRFShampoo):
    """Check that the preconditioner dtypes are as expected.

    Args:
        optimizer: The optimizer to check.
    """
    for group, preconditioner, preconditioner_momenta in zip(
        optimizer.param_groups,
        optimizer.preconditioner,
        optimizer.preconditioner_momenta,
    ):
        (default_dt,) = {p.dtype for p in group["params"]}
        dtypes = [
            default_dt if dt is None else dt for dt in group["preconditioner_dtypes"]
        ]
        assert len(preconditioner) == len(preconditioner_momenta) == len(dtypes)
        for prec, mom, dt in zip(preconditioner, preconditioner_momenta, dtypes):
            (prec_dt,) = {t.dtype for _, t in prec.named_tensors()}
            assert prec_dt == dt

            if group["alpha2"] == 0.0:
                assert mom is None
            else:
                (mom_dt,) = {t.dtype for _, t in mom.named_tensors()}
                assert mom_dt == dt


def verify_preconditioner_structures(optimizer: SIRFShampoo):
    """Check that the preconditioner structures are as expected.

    Args:
        optimizer: The optimizer to check.
    """
    for group, preconditioner, preconditioner_momenta in zip(
        optimizer.param_groups,
        optimizer.preconditioner,
        optimizer.preconditioner_momenta,
    ):
        structures = group["structures"]
        assert len(preconditioner) == len(preconditioner_momenta) == len(structures)
        for prec, mom, s in zip(preconditioner, preconditioner_momenta, structures):
            assert isinstance(prec, optimizer.SUPPORTED_STRUCTURES[s])

            if group["alpha2"] == 0.0:
                assert mom is None
            else:
                assert isinstance(mom, optimizer.SUPPORTED_STRUCTURES[s])


def test__verify_hyperparameters():
    """Test verification of hyperparameters."""
    manual_seed(0)
    D_in, D_hidden, D_out = 5, 4, 3
    model = nested_network(D_in, D_hidden, D_out)

    with raises(ValueError):
        SIRFShampoo(model, lr=-0.1)
    with raises(ValueError):
        SIRFShampoo(model, beta2=-0.1)
    with raises(ValueError):
        SIRFShampoo(model, alpha1=1)
    with raises(ValueError):
        SIRFShampoo(model, lam=-0.1)
    with raises(ValueError):
        SIRFShampoo(model, kappa=-0.1)
    with raises(ValueError):
        SIRFShampoo(model, T=-1)
    with raises(ValueError):
        SIRFShampoo(model, structures="not_a_supported_structure")

    # test specification of structures
    cases = [
        {1: "dense"},  # incomplete
        "some_structure",  # unsupported
        {1: ("dense", "dense", "dense")},  # wrong format
    ]
    for case in cases:
        with raises(ValueError):
            SIRFShampoo(model, structures=case)

    # test specification of structures
    cases = [
        {1: float32},  # incomplete
        "some_dtype",  # unsupported
        {1: float32, 2: (float32, float32, float32)},  # wrong format
    ]
    for case in cases:
        with raises(ValueError):
            SIRFShampoo(model, preconditioner_dtypes=case)


def test_batch_size_accumulation():
    """Check batch size accumulation through hooks inside the optimizer."""
    manual_seed(0)
    micro_batch_sizes = [6, 6, 6]
    # scale the micro-batch loss to obtain correct scale of accumulated gradient
    micro_batch_scale = 1 / 3
    D_in, D_hidden, D_out = 5, 4, 3
    model = nested_network(D_in, D_hidden, D_out)
    loss_func = MSELoss()
    batches = [(rand(B, D_in), rand(B, D_out)) for B in micro_batch_sizes]

    optimizer = SIRFShampoo(model, lr=0.1, kappa=0.001)
    num_steps = 5

    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        verify_preconditioner_dtypes(optimizer)
        verify_preconditioner_structures(optimizer)

        # gradient accumulation over micro-batches
        batch_loss = 0
        for i, (X, y) in enumerate(batches):
            loss = micro_batch_scale * loss_func(model(X), y)
            loss.backward()
            batch_loss += loss.item()
            assert optimizer.batch_size == sum(micro_batch_sizes[: i + 1])

        losses.append(batch_loss)

        print(f"Step {step:03g}, Loss: {losses[-1]:.5f}")
        optimizer.step()

    assert losses[0] > losses[-1]

    # after .step(), a new forward pass through the model will start a new accumulation
    some_B = 7
    model(zeros(some_B, D_in))
    assert optimizer.batch_size == some_B

    # forward passes with the model in evaluation mode do not increase the counter
    some_other_B = 8
    model.eval()
    model(zeros(some_other_B, D_in))
    assert optimizer.batch_size == some_B


class ParamsOutsideLayers(Module):
    """Neural network which has parameters outside its layers."""

    def __init__(self, D_in: int, D_hidden: int, D_out: int):
        """Set up the internal layers.

        Args:
            D_in: Input dimension.
            D_hidden: Hidden dimension.
            D_out: Output dimension.
        """
        super().__init__()
        self.outside_weight = Parameter(rand(D_in, D_in))
        self.sequential = Sequential(
            OrderedDict(
                {
                    "linear1": Linear(D_in, D_hidden, bias=False),
                    "relu": ReLU(),
                    "linear2": Linear(D_hidden, D_out, bias=False),
                    "sigmoid": Sigmoid(),
                }
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.sequential(self.outside_weight @ x)


def test_one_param_group_per_preconditioner_params_outside_layers():
    """Test that parameters outside layers are properly detected.

    In our definition, a layer is a module that does not have submodules.
    """
    D_in, D_hidden, D_out = 5, 4, 3
    model = ParamsOutsideLayers(D_in, D_hidden, D_out)
    SIRFShampoo(model, verbose_init=True)


def test_lost_parameters_during_preconditioner_detection():
    """Test that lost parameters are detected when re-writing parameter groups."""
    D_in, D_hidden, D_out = 5, 4, 3
    model = nested_network(D_in, D_hidden, D_out)

    with raises(ValueError):
        SIRFShampoo(model, combine_params=())  # empty detection rule

    with raises(ValueError):
        SIRFShampoo(
            model, combine_params=(LinearWeightBias(),)
        )  # loses the linear layer without bias
