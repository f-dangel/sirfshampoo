"""# Per-parameter Options.

Here we demonstrate `SIRFShampoo`'s more fine-grained configuration options.

We will use [parameter
groups](https://pytorch.org/docs/stable/optim.html#per-parameter-options) which
allow training parameters of a neural network differently and demonstrate this
by taking a CNN and training the parameters in the linear layers differently than
those of the convolutional layers.

First, the imports.
"""

from torch import cuda, device, manual_seed
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from sirfshampoo import SIRFShampoo

manual_seed(0)  # make deterministic
MAX_STEPS = 200  # quit training after this many steps (or one epoch)
DEV = device("cuda" if cuda.is_available() else "cpu")

# %%
# ## Problem Setup
#
# Next, we load the data set, define the neural network, and the loss
# function:

BATCH_SIZE = 32
train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Sequential(
    Conv2d(1, 3, kernel_size=5, stride=2),
    ReLU(),
    Flatten(),
    Linear(432, 50),
    ReLU(),
    Linear(50, 10),
).to(DEV)
loss_func = CrossEntropyLoss().to(DEV)

# %%
# ## Optimizer Setup
#
# As mentioned above, we will train parameters of convolutions different than
# those in linear layers. We will do so by specifying two groups, and passing
# them to the optimizer via `param_groups`.
#
# Specifically, we will use a dense pre-conditioner for convolutions, and a
# diagonal pre-conditioner for linear layers. We will also update the pre-conditioners
# at different steps.
#
# First, we identify each group's parameters:

conv_params = [
    p
    for m in model.modules()
    if isinstance(m, Conv2d)
    for p in m.parameters()
    if p.requires_grad
]
linear_params = [
    p
    for m in model.modules()
    if isinstance(m, Linear)
    for p in m.parameters()
    if p.requires_grad
]

# %%
#
# Second, let's set up the schedules for updating the pre-conditioners, as well as their
# structures:


def T_conv(step: int) -> bool:
    """Pre-conditioner update schedule for parameters in convolutional layers.

    Args:
        step: Global step of the optimizer.

    Returns:
        Whether to update the pre-conditioner.
    """
    steps = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    if step in steps:
        print(f"Updating pre-conditioner of a convolution parameter at step {step}.")
    return step in steps


T_linear = 5  # every 5 steps


structures_conv = "dense"
structures_linear = "diagonal"

# %%
#
# We are now ready to set up the two groups:
conv_group = {
    "params": conv_params,
    "structures": structures_conv,
    "T": T_conv,
}
linear_group = {
    "params": linear_params,
    "structures": structures_linear,
    "T": T_linear,
}

# %%
#
# The `param_groups` are just a list containing the groups. We can pass it to
# the optimizer's `params` argument. Let's turn on the `verbose_init` flag to inspect
# the pre-conditioner structures:

param_groups = [conv_group, linear_group]
optimizer = SIRFShampoo(
    model,
    params=param_groups,
    lr=0.01,  # shared across all groups
    verbose_init=True,
)

# %%
#
# That's everything. What follows is just a canonical training loop.
#
# ## Training
#
# Let's train for a couple of steps and print the loss. SIRFShampoo works like most
# other PyTorch optimizers:

PRINT_LOSS_EVERY = 25  # logging interval

for step, (inputs, target) in enumerate(train_loader):
    optimizer.zero_grad()  # clear gradients from previous iterations

    # regular forward-backward pass
    loss = loss_func(model(inputs.to(DEV)), target.to(DEV))
    loss.backward()
    if step % PRINT_LOSS_EVERY == 0:
        print(f"Step {step}, Loss {loss.item():.3f}")

    optimizer.step()  # update neural network parameters

    if step >= MAX_STEPS:  # don't train a full epoch to keep the example light-weight
        break

# %%
#
# ## Conclusion
#
# Congratulations! You now know how to train each layer of a neural network
# differently with `SIRFShampoo`.
#
# For example, this may be useful when the network has layers with large
# pre-conditioner dimensions. One way to reduce cost would be to use a more
# light-weight pre-conditioner type (e.g. `'diagonal'`) for such layers.
# But of course you can also use this to tweak learning rates, momenta, etc.
# per layer.
#
# To find out more about `SIRFShampoo`'s configuration options, check out the
# optimizer's [docstring](https://sirfshampoo.readthedocs.io/en/latest/api/).
