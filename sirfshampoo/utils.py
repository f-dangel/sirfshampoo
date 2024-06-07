"""Utility functions."""

from singd.structures.base import StructuredMatrix
from torch import Size, Tensor


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
