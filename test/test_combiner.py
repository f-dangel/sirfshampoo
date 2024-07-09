"""Test combining and splitting tensors."""

from test.utils import DEVICE_IDS, DEVICES

from pytest import mark
from torch import allclose, device, manual_seed, rand, zeros
from torch.nn import Linear

from sirfshampoo.combiner import LinearWeightBias, PerParameter


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


@mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
def test_LinearWeightBias_identify(dev: device):
    """Test parameter identification of `LinearWeightBias` class.

    Args:
        dev: Device to run the test on.
    """
    model = Linear(2, 3).to(dev)
    assert LinearWeightBias().identify(model) == [[model.weight, model.bias]]


@mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
def test_LinearWeightBias_group_and_ungroup(dev: device):
    """Test tensor (un-)grouping of `LinearWeightBias` class.

    Args:
        dev: Device to run the test on.
    """
    manual_seed(0)

    # regular shape
    D_out, D_in = 5, 3
    W, b = rand(D_out, D_in, device=dev), rand(D_out, device=dev)
    tensor = zeros(D_out, D_in + 1, device=dev)
    tensor[:, :D_in] = W
    tensor[:, D_in] = b
    shapes = [W.shape, b.shape]

    grouped = LinearWeightBias().group([W, b])
    assert allclose(grouped, tensor)
    (W_ungrouped, b_ungrouped) = LinearWeightBias().ungroup(grouped, shapes)
    assert allclose(W_ungrouped, W) and allclose(b_ungrouped, b)

    # one input and output dimension
    D_out, D_in = 1, 1
    W, b = rand(D_out, D_in, device=dev), rand(D_out, device=dev)
    tensor = zeros(D_out, D_in + 1, device=dev)
    tensor[:, :D_in] = W
    tensor[:, D_in] = b
    tensor = tensor.squeeze(0)
    shapes = [W.shape, b.shape]

    grouped = LinearWeightBias().group([W, b])
    assert allclose(grouped, tensor)
    (W_ungrouped, b_ungrouped) = LinearWeightBias().ungroup(grouped, shapes)
    assert allclose(W_ungrouped, W) and allclose(b_ungrouped, b)
