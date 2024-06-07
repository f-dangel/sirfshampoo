"""Tests utility functions."""

from test.utils import DEVICE_IDS, DEVICES
from typing import Type

from pytest import mark, raises
from singd.structures.base import StructuredMatrix
from singd.structures.dense import DenseMatrix
from singd.structures.diagonal import DiagonalMatrix
from singd.structures.triltoeplitz import TrilToeplitzMatrix
from singd.structures.triutoeplitz import TriuToeplitzMatrix
from torch import allclose, device, einsum, manual_seed, rand

from sirfshampoo.utils import tensormatdot

STRUCTURES = [DenseMatrix, TrilToeplitzMatrix, DiagonalMatrix, TriuToeplitzMatrix]
STRUCTURE_IDS = [structure.__name__ for structure in STRUCTURES]

TRANSPOSE = [True, False]
TRANSPOSE_IDS = [f"transpose={t}" for t in TRANSPOSE]


@mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("transpose", TRANSPOSE, ids=TRANSPOSE_IDS)
@mark.parametrize("structure", STRUCTURES, ids=STRUCTURE_IDS)
def test_tensormatdot(structure: Type[StructuredMatrix], transpose: bool, dev: device):
    """Test multiplying a structured matrix into a tensor.

    Args:
        structure: The type of structured matrix to use.
        transpose: Whether to transpose the matrix before multiplication.
        dev: The device to run the test on.
    """
    manual_seed(0)

    # multiplication into a tensor
    T = rand(2, 3, 4, 5, device=dev)
    M_sym = rand(4, 4, device=dev)
    M_sym = M_sym @ M_sym.T
    M = structure.from_dense(M_sym)

    truth = einsum("mk,ijkl->ijml", M.to_dense().T if transpose else M.to_dense(), T)
    dim = 2
    result = tensormatdot(T, M, dim, transpose=transpose)
    assert allclose(result, truth)

    # multiplication into a vector
    T = rand(4, device=dev)
    M_sym = rand(4, 4, device=dev)
    M_sym = M_sym @ M_sym.T
    M = structure.from_dense(M_sym)

    truth = einsum("mk,k->m", M.to_dense().T if transpose else M.to_dense(), T)
    dim = 0
    result = tensormatdot(T, M, dim, transpose=transpose)
    assert allclose(result, truth)

    # invalid value of dim
    invalid_dims = [1, -1]  # too large, too small
    for dim in invalid_dims:
        with raises(ValueError):
            tensormatdot(T, M, dim, transpose=transpose)
