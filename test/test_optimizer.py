"""Test `sirfshampoo.optimizer` module."""

from collections import OrderedDict

from pytest import raises
from torch import manual_seed, rand
from torch.nn import Linear, MSELoss, ReLU, Sequential, Sigmoid

from sirfshampoo.optimizer import SIRFShampoo


def test__create_mappings():
    """Test creation of optimizer-internal mappings."""
    manual_seed(0)
    D_in, D_hidden, D_out = 5, 4, 3

    # use a nested network
    inner = Sequential(
        OrderedDict(
            {
                "linear": Linear(D_hidden, D_hidden, bias=False),
                "sigmoid": Sigmoid(),
            }
        )
    )
    model = Sequential(
        OrderedDict(
            {
                "linear1": Linear(D_in, D_hidden),
                "relu1": ReLU(),
                "inner": inner,
                "linear2": Linear(D_hidden, D_out),
            }
        )
    )

    # one parameter group
    optimizer = SIRFShampoo(model)
    assert optimizer._params_in_layer == {
        "linear1": ["weight", "bias"],
        "inner.linear": ["weight"],
        "linear2": ["weight", "bias"],
    }
    assert optimizer._layer_to_param_group == {
        "linear1": 0,
        "inner.linear": 0,
        "linear2": 0,
    }

    # two parameter groups (sub-set)
    param_groups = [
        {
            "params": [
                model.get_parameter("linear1.weight"),
                model.get_parameter("linear2.bias"),
            ]
        },
        {"params": [model.get_parameter("inner.linear.weight")]},
    ]
    optimizer = SIRFShampoo(model, params=param_groups)
    assert optimizer._params_in_layer == {
        "linear1": ["weight"],
        "inner.linear": ["weight"],
        "linear2": ["bias"],
    }
    assert optimizer._layer_to_param_group == {
        "linear1": 0,
        "inner.linear": 1,
        "linear2": 0,
    }

    # two parameter groups (sub-set), layer parameters in different groups
    param_groups = [
        {"params": [model.get_parameter("linear2.bias")]},
        {"params": [model.get_parameter("linear2.weight")]},
    ]
    with raises(ValueError):
        SIRFShampoo(model, params=param_groups)


def test_step_integration():
    """Check the optimizer is able to take a couple of steps without erroring."""
    manual_seed(0)
    batch_size = 6
    D_in, D_hidden, D_out = 5, 4, 3
    # use a nested network
    inner = Sequential(
        OrderedDict(
            {
                "linear": Linear(D_hidden, D_hidden, bias=False),
                "sigmoid": Sigmoid(),
            }
        )
    )
    model = Sequential(
        OrderedDict(
            {
                "linear1": Linear(D_in, D_hidden),
                "relu1": ReLU(),
                "inner": inner,
                "linear2": Linear(D_hidden, D_out),
            }
        )
    )
    loss_func = MSELoss()
    X, y = rand(batch_size, D_in), rand(batch_size, D_out)

    optimizer = SIRFShampoo(model, beta1=0.1, kappa=0.001)
    num_steps = 5
    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = loss_func(model(X), y)
        loss.backward()
        losses.append(loss.item())
        print(f"Step {step:03g}, Loss: {losses[-1]:.5f}")
        optimizer.step()

    assert losses[0] > losses[-1]
