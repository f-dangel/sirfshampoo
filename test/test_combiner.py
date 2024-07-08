"""Test combining and splitting tensors."""

from test.utils import DEVICE_IDS, DEVICES

from pytest import mark
from torch import allclose, device, manual_seed, rand
from torch.nn import Linear

from sirfshampoo.combiner import PerParameter


@mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
def test_PerParameter_identify(dev: device):
    """Test parameter identification of `PerParameter` class.

    Args:
        dev: Device to run the test on.
    """
    model = Linear(2, 3).to(dev)
    assert PerParameter().identify(model) == [[model.weight], [model.bias]]


@mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
def test_PerParameter_group_and_ungroup(dev: device):
    """Test tensor (un-)grouping of `PerParameter` class.

    Args:
        dev: Device to run the test on.
    """
    manual_seed(0)

    # regular shape
    tensor = rand(5, 10, 7, device=dev)
    shapes = [tensor.shape]

    grouped = PerParameter().group([tensor])
    assert allclose(grouped, tensor)
    (ungrouped,) = PerParameter().ungroup(grouped, shapes)
    assert allclose(ungrouped, tensor)

    # shape with ones
    tensor = rand(5, 7, device=dev)
    tensor_unsqueezed = tensor.unsqueeze(1).unsqueeze(2)
    shapes = [tensor_unsqueezed.shape]

    grouped = PerParameter().group([tensor_unsqueezed])
    assert allclose(grouped, tensor)
    (ungrouped,) = PerParameter().ungroup(grouped, shapes)
    assert allclose(ungrouped, tensor_unsqueezed)

    # scalar
    scalar = rand(1, device=dev).squeeze()
    scalar_unsqueezed = scalar.unsqueeze(0)
    shapes = [scalar_unsqueezed.shape]

    grouped = PerParameter().group([scalar_unsqueezed])
    assert allclose(grouped, scalar)
    (ungrouped,) = PerParameter().ungroup(grouped, shapes)
    assert allclose(ungrouped, scalar_unsqueezed)
