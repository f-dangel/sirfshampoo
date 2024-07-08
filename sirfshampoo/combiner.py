"""Abstraction for combining multiple tensors into one."""

from abc import ABC, abstractmethod
from typing import List

from torch import Size, Tensor
from torch.nn import Module, Parameter


class TensorGroup(ABC):
    """Interface for treating multiple tensors with one pre-conditioner."""

    @abstractmethod
    def identify(self, model: Module) -> List[List[Parameter]]:
        """Detect parameters that should be treated jointly.

        Args:
            Module: The neural network.

        Returns:
            A list whose entries are list of parameters that are treated jointly.

        Raises:
            NotImplementedError: Must be implemented by a child class.
        """
        raise NotImplementedError

    @abstractmethod
    def group(self, tensors: List[Tensor]) -> Tensor:
        """Combine multiple tensors into one.

        This is the inverse operation of `ungroup`.

        Args:
            tensors: List of tensors to combine.

        Returns:
            Combined tensor.

        Raises:
            NotImplementedError: Must be implemented by a child class.
        """
        raise NotImplementedError

    @abstractmethod
    def ungroup(
        self, grouped_tensor: Tensor, tensor_shapes: List[Size]
    ) -> List[Tensor]:
        """Split a combined tensor into multiple tensors.

        This is the inverse operation of `group`.

        Args:
            grouped_tensor: Combined tensor.
            tensor_shapes: Shapes of the tensors to split into.

        Returns:
            List of tensors.

        Raises:
            NotImplementedError: Must be implemented by a child class.
        """
        raise NotImplementedError


class PerParameter(TensorGroup):
    """Treat each parameter with a separate pre-conditioner."""

    def identify(self, model: Module) -> List[List[Parameter]]:
        """Detect parameters that should be treated jointly.

        Args:
            Module: The neural network.

        Returns:
            A list of lists. Each sub-list contains one parameter.
        """
        return [[p] for p in model.parameters()]

    def group(self, tensors: List[Tensor]) -> Tensor:
        """Combine tensors that are pre-conditioned jointly.

        Args:
            tensors: List of tensors to combine.

        Returns:
            Combined tensor. Axes of size 1 are squeezed to avoid an unnecessary 1x1
            Kronecker factor in the pre-conditioner.
        """
        (tensor,) = tensors
        squeezed = tensor.squeeze()
        return squeezed.unsqueeze(0) if squeezed.ndim == 0 else squeezed

    def ungroup(
        self, grouped_tensor: Tensor, tensor_shapes: List[Size]
    ) -> List[Tensor]:
        """Split a combined tensor into multiple tensors.

        This is the inverse operation of `group`.

        Args:
            grouped_tensor: Combined tensor.
            tensor_shapes: Shapes of the tensors to split into.

        Returns:
            List of tensors of length 1. Entry tensor has the same shape as the
            parameter.
        """
        (shape,) = tensor_shapes
        return [grouped_tensor.reshape(shape)]


# class LinearWeightAndBias(TensorGroup):
#     """Treat weight and bias of a linear layer jointly.

#     Stacks the bias as last column to the weight matrix.
#     """

#     def identify(self, model: Module) -> List[List[Parameter]]:
#         return [
#             [module.weight, module.bias]
#             for module in model.modules()
#             if isinstance(module, Linear) and module.bias is not None
#         ]

#     def group(self, tensors: List[Tensor]) -> Tensor:
#         t_weight, t_bias = tensors
#         combined = stack([t_weight, t_bias.unsqueeze(1)], dim=1).squeeze()
#         return combined.unsqueeze(0) if combined.ndim == 0 else combined

#     def ungroup(
#         self, grouped_tensor: Tensor, tensor_shapes: List[Size]
#     ) -> List[Tensor]:
#         (shape_weight, shape_bias) = tensor_shapes

#         # if the `Linear` layer has `out_features=1`, then one dimension of the
#         # weight matrix will be squeezed during the call to `group`.
#         split_dim = 0 if grouped_tensor.ndim == 1 else 1

#         d_in = shape_weight[1]
#         t_weight, t_bias = split(grouped_tensor, [d_in, 1], dim=split_dim)

#         return [t_weight.reshape(shape_weight), t_bias.reshape(shape_bias)]
