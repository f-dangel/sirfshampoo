"""Utility functions for tests."""

from typing import Any

from singd.structures.base import StructuredMatrix
from torch import Tensor, allclose, cuda, device, isclose

from sirfshampoo.optimizer import SIRFShampoo

DEVICES = [device("cpu")] + [device(f"cuda:{i}") for i in range(cuda.device_count())]
DEVICE_IDS = [f"dev={str(dev)}" for dev in DEVICES]


def report_nonclose(
    tensor1: Tensor,
    tensor2: Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    name: str = "array",
):
    """Compare two tensors, raise exception if nonclose values and print them.

    Args:
        tensor1: First tensor.
        tensor2: Second tensor.
        rtol: Relative tolerance (see `torch.allclose`). Default: `1e-5`.
        atol: Absolute tolerance (see `torch.allclose`). Default: `1e-8`.
        equal_nan: Whether comparing two NaNs should be considered as `True`
            (see `torch.allclose`). Default: `False`.
        name: Optional name what the compared tensors mean. Default: `'array'`.

    Raises:
        ValueError: If the two tensors don't match in shape or have nonclose values.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"{name} shapes don't match.")

    if allclose(tensor1, tensor2, rtol=rtol, atol=atol, equal_nan=equal_nan):
        print(f"{name} values match.")
    else:
        mismatch = 0
        for a1, a2 in zip(tensor1.flatten(), tensor2.flatten()):
            if not isclose(a1, a2, atol=atol, rtol=rtol, equal_nan=equal_nan):
                mismatch += 1
                print(f"{a1} ≠ {a2}")
        print(f"Min entries: {tensor1.min()}, {tensor2.min()}")
        print(f"Max entries: {tensor1.max()}, {tensor2.max()}")
        raise ValueError(f"{name} values don't match ({mismatch} / {tensor1.numel()}).")


def compare(obj1: Any, obj2: Any, name: str = "object"):
    """Compare two objects.

    Args:
        obj1: First object.
        obj2: Second object.
        name: Optional name what the compared objects mean. Default: `'object'`.

    Raises:
        NotImplementedError: If the comparison is not implemented.
    """
    assert isinstance(obj1, type(obj2))
    if isinstance(obj1, (float, int, tuple)):
        assert obj1 == obj2
    elif isinstance(obj1, list) and all(isinstance(e, Tensor) for e in obj1):
        assert len(obj1) == len(obj2)
        for e1, e2 in zip(obj1, obj2):
            report_nonclose(e1, e2, name=name)
    elif isinstance(obj1, list) and all(isinstance(e, StructuredMatrix) for e in obj1):
        assert len(obj1) == len(obj2)
        for e1, e2 in zip(obj1, obj2):
            report_nonclose(e1.to_dense(), e2.to_dense(), name=name)
    else:
        raise NotImplementedError(f"Unsupported comparison for type: {type(obj1)}.")


def compare_optimizers(optim1: SIRFShampoo, optim2: SIRFShampoo):
    """Compare two SIRFShampoo optimizers.

    Args:
        optim1: First optimizer.
        optim2: Second optimizer.
    """
    assert optim1.global_step == optim2.global_step
    assert optim1.batch_size == optim2.batch_size
    assert optim1.batch_size_valid == optim2.batch_size_valid

    # preconditioner matrices
    preconditioner1, preconditioner2 = optim1.preconditioner, optim2.preconditioner
    assert len(preconditioner1) == len(preconditioner2)
    for prec1_list, prec2_list in zip(preconditioner1, preconditioner2):
        compare(prec1_list, prec2_list, name="preconditioner")

    # preconditioner momenta
    momenta1, momenta2 = optim1.preconditioner_momenta, optim2.preconditioner_momenta
    assert len(momenta1) == len(momenta2)
    for mom1_list, mom2_list in zip(momenta1, momenta2):
        assert len(mom1_list) == len(mom2_list)
        compare(mom1_list, mom2_list, name="preconditioner momentum")

    # state dictioneries
    assert len(optim1.state) == len(optim2.state)
    for (p1, s1), (p2, s2) in zip(optim1.state.items(), optim2.state.items()):
        assert p1.shape == p2.shape
        report_nonclose(p1, p2, name="parameter")
        assert list(s1.keys()) == list(s2.keys()) == ["momentum_buffer"]
        report_nonclose(p1, p2, name="parameter")
        report_nonclose(s1["momentum_buffer"], s2["momentum_buffer"], name="momentum")

    # groups
    param_group_entries = {
        "lr",
        "beta2",
        "alpha1",
        "alpha2",
        "lam",
        "kappa",
        "T",
        "structures",
        "preconditioner_dtypes",
        "params",
    }
    assert len(optim1.param_groups) == len(optim2.param_groups)
    for group1, group2 in zip(optim1.param_groups, optim2.param_groups):
        assert set(group1.keys()) == set(group2.keys()) == param_group_entries
        for key in param_group_entries:
            entry1, entry2 = group1[key], group2[key]
            compare(entry1, entry2)
