"""# Pre-conditioner Groups.

In this tutorial, we show how to customize `SIRFShampoo` to treat multiple parameters
with a single pre-conditioner, i.e. how to form a pre-conditioner group.
First, we will illustrate the default behaviour, then talk about other built-in options.
Last, we will show how to define custom rules.

Some use cases where this is useful are:

- Combining the weight matrix and bias vector of an `nn.Linear` layer into one matrix
  by appending the bias as last column
- Combining the `d`-dimensional weight and bias vectors of a normalization layer into
  a `dx2` matrix
- Combining multiple (say `L`) weights of shape `d_out x d_in` into a 3d tensor of shape
  `L x d_out x d_in` (think LLMs)
- Reshaping a parameter tensor with a large dimension into a tensor with higher rank but
  smaller dimensions per axis (think embedding layers or large last linear layers)

First, the imports.
"""

from collections import OrderedDict
from typing import List

from pytest import raises
from torch import Size, Tensor, cat, cuda, device, manual_seed, rand
from torch.nn import Linear, Module, MSELoss, Parameter, ReLU, Sequential

from sirfshampoo import SIRFShampoo
from sirfshampoo.combiner import LinearWeightBias, PerParameter, PreconditionerGroup

manual_seed(0)  # make deterministic
DEV = device("cuda" if cuda.is_available() else "cpu")

# %%
# ## Setup
#
# We will not train neural networks in this tutorial, but only look at `SIRFShampoo`'s
# configuration after setting up the optimizer. We will use a simple MLP with three
# fully-connected layers activated by ReLU:

model = Sequential(
    OrderedDict(
        {
            "linear1": Linear(512, 256, bias=False),
            "relu1": ReLU(),
            "linear2": Linear(256, 128),
            "relu2": ReLU(),
            "linear3": Linear(128, 64),
        }
    )
).to(DEV)

# %%
# ## Default Behaviour (One Pre-conditioner per Parameter)
#
# By default, `SIRFShampoo` will treat each parameter with a separate pre-conditioner.
# We can observe this while setting up the optimizer by turning on `verbose_init=True`:

optimizer = SIRFShampoo(model, verbose_init=True)

# %%
# We can read off that each parameter forms its own group that is handled with an
# independent pre-conditioner. For instance, `'linear2.bias'` has its own `128 x 128`
# pre-conditioner.

# %%
# ## Built-in: Treating Weights and Biases of a Linear Layer Jointly
#
# If you look carefully in the example above, you can also see under the
# `'Other'` entry of each group that there is a key `'combine_params'`. The
# associated value is an instance of [`PerParameter`](
# https://sirfshampoo.readthedocs.io/en/latest/api/#sirfshampoo.PerParameter),
# which is one defined rule to assign parameters to pre-conditioners (in this
# case, one pre-conditioner per parameter).
#
# We also provide a rule for treating weights and biases of an `nn.Linear`
# layer jointly. This is done by appending the bias as additional column to the
# weight matrix. To use this rule, we need to create an instance of
# [`LinearWeightBias`](
# https://sirfshampoo.readthedocs.io/en/latest/api/#sirfshampoo.LinearWeightBias)
# and pass it to the optimizer's `combine_params` argument:

optimizer = SIRFShampoo(
    model,
    combine_params=(LinearWeightBias(), PerParameter()),
    verbose_init=True,
)


# %%
#
# Note that now there are only three parameter groups, and two of them contain the
# weight and bias of a linear layer (also, the pre-conditioner dimensions are slightly
# different). The third group is simply weight of the first layer which does not have a
# bias term. You can also see under `'Other'` that the first two groups use a
# `LinearWeightBias` instance under `'combine_params'`, while the last group uses a
# `PerParameter` instance.
#
# We had to pass the tuple `combine_params=(LinearWeightBias(), PerParameter())` to
# the optimizer, which will iterate over the supplied rules and identify
# pre-conditioner groups, prioritizing the rules that were supplied first.
#
# Had we only passed `combine_params=(LinearWeightBias(),)`, then the optimizer would
# have crashed because it would not have been able to assign the first layer's weight
# to a pre-conditioner:

with raises(ValueError):
    optimizer = SIRFShampoo(
        model,
        # no fall-back option to `PerParameter` leads to crash because the net has
        # a linear layer without bias
        combine_params=(LinearWeightBias(),),
    )


# %%
#
# ## Writing Custom Pre-conditioner Groups
#
# So far we discussed the default option to treat each parameter with its own
# pre-conditioner via [`PerParameter`](
# https://sirfshampoo.readthedocs.io/en/latest/api/#sirfshampoo.PerParameter),
# and to combine weight and bias of a linear layer with [`LinearWeightBias`](
# https://sirfshampoo.readthedocs.io/en/latest/api/#sirfshampoo.LinearWeightBias).
#
# Here, we discuss how to implement a custom rule to group parameters together.
# `sirfshampoo` offers a [`PreconditionerGroup`](
# https://sirfshampoo.readthedocs.io/en/latest/api/#sirfshampoo.PreconditionerGroup)
# interface which can be implemented to create new rules.
#
# Let's implement our custom (albeit a little artificial) rule, which will treat:
#
# - each weight parameter independently
# - all biases jointly with one pre-conditioner by stacking them together
#
# Let's call this rule `SeparateWeightsJointBiases`. Here is its implementation:


class SeparateWeightsJointBiases(PreconditionerGroup):
    """Pre-conditioner group to treat weights independently and biases jointly."""

    def identify(self, model: Module) -> List[List[Parameter]]:
        """Detect parameters that should be treated jointly.

        Args:
            model: The neural network.

        Returns:
            A list of lists. Each sub-list contains either a single weight or all
            biases.
        """
        independent = []

        biases = []
        for name, param in model.named_parameters():
            if "weight" in name:
                independent.append([param])
            elif "bias" in name:
                biases.append(param)
        independent.append(biases)

        return independent

    def group(self, tensors: List[Tensor]) -> Tensor:
        """Combine tensors that are pre-conditioned together into one tensor.

        Args:
            tensors: List of tensors to combine. The list either has a single entry
                that is a weight-shaped tensor, or multiple entries that are bias-
                shaped tensors.

        Returns:
            The combined tensor.
        """
        # does nothing if `tensors` has one entry, otherwise
        # concatenates bias-shaped entries
        combined = cat(tensors)
        # NOTE It is good practise to remove axes of size 1 from the combined tensor,
        # because this will otherwise create 1x1 pre-conditioners which are unnecessary.
        combined = combined.squeeze()
        # However, the combined tensor must have at least one axis
        combined = combined.unsqueeze(0) if combined.ndim == 0 else combined

        return combined

    def ungroup(
        self, grouped_tensor: Tensor, tensor_shapes: List[Size]
    ) -> List[Tensor]:
        """Split the combined tensor into the original components.

        This is the inverse operation of `group`.

        Args:
            grouped_tensor: Combined tensor.
            tensor_shapes: Shapes of the tensors to split into.

        Returns:
            List of tensors that have the specified shapes.
        """
        if len(tensor_shapes) == 1:  # weight case or just one overall bias
            return [grouped_tensor.reshape(tensor_shapes[0])]

        # bias case
        bias_dims = [s.numel() for s in tensor_shapes]
        tensors = grouped_tensor.split(bias_dims)
        return [t.reshape(s) for t, s in zip(tensors, tensor_shapes)]


# %%
#
# Let's create an optimizer with this custom rule (note that specifying per-parameter
# rule as fallback would not be necessary here because our custom rule matches all
# parameters that are trained; but doing so is in general good practise).

optimizer = SIRFShampoo(
    model,
    combine_params=(SeparateWeightsJointBiases(), PerParameter()),
    verbose_init=True,
)

# %%
#
# As expected, the optimizer has four groups. Three contain one weight matrix each.
# The last group contains all bias parameters. We can also see a difference in the
# `'combine_params` values in the `'Other'` section.
#
# To make sure everything works, let's train on synthetic data for a couple of steps.

# synthetic data
BATCH_SIZE = 32
X, y = rand(BATCH_SIZE, 512, device=DEV), rand(BATCH_SIZE, 64, device=DEV)
loss_func = MSELoss().to(DEV)

STEPS = 200
PRINT_LOSS_EVERY = 25  # logging interval
initial_loss = loss_func(model(X), y).item()

for step in range(STEPS):
    optimizer.zero_grad()  # clear gradients from previous iterations

    # regular forward-backward pass
    loss = loss_func(model(X), y)
    loss.backward()
    if step % PRINT_LOSS_EVERY == 0:
        print(f"Step: {step}, Loss: {loss.item():.3f}")

    optimizer.step()  # update neural network parameters

# make sure the loss decreased
final_loss = loss_func(model(X), y).item()
assert final_loss < initial_loss

# %%
#
# ## Conclusion
#
# Congratulations! You now know how to jointly pre-condition multiple parameters
# using `sirfshampoo`'s built-in rules, and how to write your custom rules via
# the `PreconditionerGroup` interface.
#
# To learn more about the `PreconditionerGroup` interface, check out its
# [`documentation`](
# https://sirfshampoo.readthedocs.io/en/latest/api/#sirfshampoo.PreconditionerGroup).
