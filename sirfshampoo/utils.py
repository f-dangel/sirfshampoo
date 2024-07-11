"""Utility functions."""

from typing import Any, Dict, List

from singd.structures.base import StructuredMatrix
from torch import Size, Tensor
from torch.nn import Module

from sirfshampoo.combiner import PerParameter


def tensormatdot(
    tensor: Tensor, mat: StructuredMatrix, dim: int, transpose: bool = False
) -> Tensor:
    """Multiply a structured matrix onto one axis of a tensor.

    This function is similar to `torch.tensordot`, but assumes the second argument to
    be a `StructuredMatrix`.

    If `tensor` is a matrix, then this function computes `mat @ tensor` for `dim=0`
    and `tensor @ mat.T` for `dim=1`. If `transpose` is set to `True`, `mat` will be
    transposed before the multiplication.

    Args:
        tensor: A tensor of shape `[I_1, I_2, ..., I_dim, ..., I_N]`.
        mat: A structured matrix representing a matrix of shape `[I_dim, I_dim]`.
        dim: The dimension along which the matrix is multiplied.
        transpose: If `True`, the matrix is transposed before multiplication.
            Default: `False`.

    Returns:
        A tensor of shape `[I_1, I_2, ..., I_dim, ..., I_N]` representing the result of
        the matrix multiplication of `tensor` with `matrix` along dimension `dim`.

    Raises:
        ValueError: If `dim` exceeds the dimension of `tensor` or is negative.
    """
    if not 0 <= dim < tensor.ndim:
        raise ValueError(
            f"Dimension {dim} out of bounds for tensor of shape {tensor.shape}."
        )
    # interpret vectors as matrices and move axis that is contracted to first position
    tensor_as_mat = (
        tensor.unsqueeze(-1)
        if tensor.ndim == 1
        else tensor.movedim(dim, 0).flatten(start_dim=1)
    )

    result_as_mat = mat.rmatmat(tensor_as_mat) if transpose else mat @ tensor_as_mat
    shape = (
        Size([result_as_mat.shape[0]]) + tensor.shape[:dim] + tensor.shape[dim + 1 :]
    )
    return result_as_mat.reshape(shape).movedim(0, dim)


def set_up_param_groups_for_algorithmic_efficiency(
    model: Module, beta2: float, diag_threshold: int = 2048
) -> List[Dict[str, Any]]:
    """Set up parameter groups for models of the algorithmic-efficiency benchmark.

    Each parameter is treated with an independent pre-conditioner.
    Norm, projection, and bias parameters use a smaller learning rate for their
    pre-conditioner.

    Args:
        model: The neural network for which optimizer parameter groups will be created.
        beta2: The learning rate for the pre-conditioners.
        diag_threshold: Axis dimension threshold to use a diagonal rather than dense
            preconditioner. Default: `2048`.

    Returns:
        List of parameter groups for the optimizer. Each parameter group specifies the
        `'params'` and `'structures'` keys.
    """
    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # use a smaller learning rate for specific parameters
        use_beta2 = (
            0.25 * beta2
            if any(word in name for word in ["norm", "proj", "bias"])
            else beta2
        )

        # use a diagonal preconditioner for large dimensions
        preconditioner_dims = PerParameter().group([param]).shape
        N = len(preconditioner_dims)
        structures = tuple(
            "diagonal" if dim > diag_threshold else "dense"
            for dim in preconditioner_dims
        )

        param_groups.append(
            {
                "params": [param],
                "beta2": use_beta2,
                "structures": {N: structures},
            }
        )

    return param_groups
