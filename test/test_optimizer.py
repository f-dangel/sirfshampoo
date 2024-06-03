"""Test `sirfshampoo.optimizer` module."""

from collections import OrderedDict
from typing import Callable, Optional, Tuple, Union

from pytest import mark, raises
from torch import bfloat16, dtype, float16, float32, manual_seed, rand
from torch.nn import Linear, MSELoss, ReLU, Sequential, Sigmoid
from torch.optim.lr_scheduler import StepLR

from sirfshampoo.optimizer import SIRFShampoo


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
        "structures": "dense",
        "preconditioner_dtypes": None,
    }

    # one parameter group
    optimizer = SIRFShampoo(model, verbose_init=True)
    assert len(optimizer.param_groups) == 3
    assert optimizer.param_groups == [
        {
            "params": [model.linear1.weight, model.linear1.bias],
            **defaults,
            "preconditioner_dtypes": (float32, float32),
            "structures": ("dense", "dense"),
        },
        {
            "params": [model.inner.linear.weight],
            **defaults,
            "preconditioner_dtypes": (float32, float32),
            "structures": ("dense", "dense"),
        },
        {
            "params": [model.linear2.weight, model.linear2.bias],
            **defaults,
            "preconditioner_dtypes": (float32, float32),
            "structures": ("dense", "dense"),
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
    assert len(optimizer.param_groups) == 3
    assert optimizer.param_groups == [
        {
            "params": [model.linear1.weight, model.linear1.bias],
            **defaults,
            "preconditioner_dtypes": (float32, float32),
            "structures": ("dense", "dense"),
            "alpha1": 0.5,
        },
        {
            "params": [model.inner.linear.weight],
            **defaults,
            "preconditioner_dtypes": (float32, float32),
            "structures": ("dense", "dense"),
        },
        {
            "params": [model.linear2.bias],
            **defaults,
            "alpha1": 0.5,
            "preconditioner_dtypes": (float32,),
            "structures": ("dense",),
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
            "preconditioner_dtypes": (float32, float32),
            "structures": ("dense", "dense"),
        },
        {
            "params": [model.linear1.bias],
            **defaults,
            "structures": ("dense",),
            "preconditioner_dtypes": (float32,),
        },
        {
            "params": [model.inner.linear.weight],
            **defaults,
            "preconditioner_dtypes": (float32, float32),
            "structures": ("dense", "dense"),
        },
        {
            "params": [model.linear2.bias],
            **defaults,
            "alpha1": 0.5,
            "structures": ("dense",),
            "preconditioner_dtypes": (float32,),
        },
    ]

    # different data types for pre-conditioner
    param_groups = [
        {
            "params": [model.linear1.weight],
            **defaults,
            "preconditioner_dtypes": (float16, bfloat16),
        },
    ]
    optimizer = SIRFShampoo(model, params=param_groups, verbose_init=True)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups == [
        {
            "params": [model.linear1.weight],
            **defaults,
            "preconditioner_dtypes": (float16, bfloat16),
            "structures": ("dense", "dense"),
        },
    ]


def test_batch_size():
    """Test batch size detection of the optimizer."""
    manual_seed(0)
    D_in, D_hidden, D_out = 5, 4, 3
    batch_sizes = [6, 7, 8]

    # detection through forward hook
    model = nested_network(D_in, D_hidden, D_out)
    optimizer = SIRFShampoo(model)

    # batch size is not an integer before a forward pass
    assert optimizer.batch_size is None
    with raises(RuntimeError):
        optimizer._get_current_batch_size()

    for batch_size in batch_sizes:
        X = rand(batch_size, D_in)
        model(X)
        assert optimizer._get_current_batch_size() == batch_size

    # specifying an integer
    model = nested_network(D_in, D_hidden, D_out)
    const_batch_size = 9
    optimizer = SIRFShampoo(model, batch_size=const_batch_size)
    assert optimizer._get_current_batch_size() == const_batch_size

    # forward passes do not change the batch size
    for batch_size in batch_sizes:
        X = rand(batch_size, D_in)
        model(X)
        assert optimizer._get_current_batch_size() == const_batch_size


T_CASES = [1, lambda step: step in [0, 3]]
T_CASE_IDS = ["every", "custom"]

SCHEDULE_LRS = [False, True]
SCHEDULE_LR_IDS = ["constant-lr", "scheduled-lr"]

PRECONDITIONER_DTYPES = [
    None,
    (float32, bfloat16),
    (bfloat16, float32),
    (float16, bfloat16),
]
PRECONDITIONER_DTYPE_IDS = [
    "default-dtypes",
    "float32-bfloat16",
    "bfloat16-float32",
    "float16-bfloat16",
]


@mark.parametrize(
    "preconditioner_dtypes", PRECONDITIONER_DTYPES, ids=PRECONDITIONER_DTYPE_IDS
)
@mark.parametrize("schedule_lr", SCHEDULE_LRS, ids=SCHEDULE_LR_IDS)
@mark.parametrize("T", T_CASES, ids=T_CASE_IDS)
def test_step_integration(
    T: Union[int, Callable[[int], bool]],
    schedule_lr: bool,
    preconditioner_dtypes: Optional[Union[dtype, Tuple[dtype, ...]]],
):
    """Check the optimizer is able to take a couple of steps without erroring.

    Args:
        T: Pre-conditioner update schedule.
        schedule_lr: Whether to use a learning rate scheduler.
        preconditioner_dtypes: Pre-conditioner data types.
    """
    manual_seed(0)
    batch_size = 6
    D_in, D_hidden, D_out = 5, 4, 3
    model = nested_network(D_in, D_hidden, D_out)
    loss_func = MSELoss()
    X, y = rand(batch_size, D_in), rand(batch_size, D_out)

    optimizer = SIRFShampoo(
        model, lr=0.1, kappa=0.001, T=T, preconditioner_dtypes=preconditioner_dtypes
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
    """Check that the preconditioner dtypes are as expected."""
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
            (mom_dt,) = {t.dtype for _, t in mom.named_tensors()}
            assert mom_dt == dt


def verify_preconditioner_structures(optimizer: SIRFShampoo):
    """Check that the preconditioner structures are as expected."""
    for group, preconditioner, preconditioner_momenta in zip(
        optimizer.param_groups,
        optimizer.preconditioner,
        optimizer.preconditioner_momenta,
    ):
        structures = group["structures"]
        assert len(preconditioner) == len(preconditioner_momenta) == len(structures)
        for prec, mom, s in zip(preconditioner, preconditioner_momenta, structures):
            assert isinstance(prec, optimizer.SUPPORTED_STRUCTURES[s])
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
    too_many = ("dense", "dense", "dense")
    with raises(ValueError):
        SIRFShampoo(model, structures=too_many)
