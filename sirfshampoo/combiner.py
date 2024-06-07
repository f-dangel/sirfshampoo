"""Abstraction for combining multiple tensors into one."""

from typing import List

from torch import Size, Tensor, cat, split, stack


class TensorCombiner:
    """Class for combining multiple tensors into one."""

    @staticmethod
    def group(tensors: List[Tensor]) -> Tensor:
        """Combine multiple tensors into one.

        This is the inverse operation of `ungroup`.

        Args:
            tensors: List of tensors to combine.

        Returns:
            Combined tensor.

        Raises:
            NotImplementedError: If the supplied tensor shapes cannot be combined.
        """
        # one tensor only
        if len(tensors) == 1:
            # squeeze dimensions of size 1, treat scalars as vectors
            squeezed = tensors[0].squeeze()
            return squeezed.unsqueeze(0) if squeezed.ndim == 0 else squeezed

        # combining a weight and bias
        if len(tensors) == 2:
            # TODO Could allow general order here
            W, b = tensors

            # linear layer
            if W.ndim == 2 and b.ndim == 1 and W.shape[0] == b.shape[0]:
                return cat([W, b.unsqueeze(1)], dim=1)

            # 2d convolutional layer
            if W.ndim == 4 and b.ndim == 1 and W.shape[0] == b.shape[0]:
                return cat([W.flatten(start_dim=1), b.unsqueeze(1)], dim=1)

        # all tensors have the same shape (e.g. weight and bias of a normalization
        # layer): stack them along a new trailing dimension
        if len({tensor.shape for tensor in tensors}) == 1:
            return stack(tensors, dim=tensors[0].ndim)

        raise NotImplementedError(
            f"Cannot combine tensors of shape {[t.shape for t in tensors]}."
        )

    @staticmethod
    def ungroup(grouped_tensor: Tensor, tensor_shapes: List[Size]) -> List[Tensor]:
        """Split a combined tensor into multiple tensors.

        This is the inverse operation of `group`.

        Args:
            grouped_tensor: Combined tensor.
            tensor_shapes: Shapes of the tensors to split into.

        Returns:
            List of tensors.

        Raises:
            NotImplementedError: If the supplied tensor shapes cannot be split.
        """
        # one tensor only
        if len(tensor_shapes) == 1:
            return [grouped_tensor.reshape(tensor_shapes[0])]

        # combining a weight and bias
        if len(tensor_shapes) == 2:
            # TODO Could allow general order here
            W_shape, b_shape = tensor_shapes

            # linear layer
            if len(W_shape) == 2 and len(b_shape) == 1 and W_shape[0] == b_shape[0]:
                W, b = split(grouped_tensor, [W_shape[1], 1], dim=1)
                return [W, b.squeeze(1)]

            # 2d convolutional layer
            if len(W_shape) == 4 and len(b_shape) == 1 and W_shape[0] == b_shape[0]:
                W, b = split(grouped_tensor, [W_shape[1:].numel(), 1], dim=1)
                return [W.reshape(W_shape), b.squeeze(1)]

        # all tensors have the same shape (e.g. weight and bias of a normalization
        # layer): split them along the trailing dimension
        if (
            len(set(tensor_shapes)) == 1
            and grouped_tensor.shape[:-1] == tensor_shapes[0]
        ):
            return [
                t.squeeze(-1)
                for t in split(grouped_tensor, 1, dim=grouped_tensor.ndim - 1)
            ]

        raise NotImplementedError(
            f"Cannot ungroup tensor of shape {grouped_tensor.shape} into tensors of"
            + f" shape {tensor_shapes}."
        )
