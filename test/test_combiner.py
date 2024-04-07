"""Test combining and splitting tensors."""

from typing import List, Tuple

from pytest import mark, raises
from torch import Size, allclose, manual_seed, rand

from sirfshampoo.combiner import TensorCombiner

GROUPING_SHAPES = [
    [(2, 3, 4)],  # one tensor only
    [(2, 3, 4), (2, 3, 4), (2, 3, 4)],  # multiple tensors of same shape
    [(4, 3), (4,)],  # weight and bias of a linear layer
    [(4, 3, 2, 2), (4,)],  # weight and bias of a 2d convolutional layer
]


UNSUPPORTED_GROUPING_SHAPES = [
    [(2, 3, 4), (5,)],  # no obvious strategy
    # NOTE We could support arbitrary order of bias and weight
    [(4,), (4, 3)],  # bias and weight of a linear layer
    [(4,), (4, 3, 2, 2)],  # bias and weight of a 2d convolutional layer
]


def _check_group_then_ungroup(shapes: List[Tuple[int, ...]]):
    """Generate random tensors and try to group and ungroup them.

    Verify that `ungroup(group) = identity`.

    Args:
        shapes: Shapes of the tensors to generate.
    """
    tensors = [rand(shape) for shape in shapes]
    grouped = TensorCombiner.group(tensors)
    # TODO We could relax this condition in cases where one wants to
    # pad smaller tensors so they can be stacked with other tensors,
    # which would increase the number of elements in the grouped tensor.
    assert grouped.numel() == sum(t.numel() for t in tensors)
    # `ungroup` requires the shapes to be of type `Size`
    tensor_shapes = [Size(shape) for shape in shapes]
    ungrouped = TensorCombiner.ungroup(grouped, tensor_shapes)
    assert len(tensors) == len(ungrouped)
    assert all(allclose(t1, t2) for t1, t2 in zip(tensors, ungrouped))


@mark.parametrize("shapes", GROUPING_SHAPES, ids=str)
def test_TensorCombiner_group_then_ungroup(shapes: List[Tuple[int, ...]]):
    """Test `group` and `ungroup` methods of `TensorCombiner`.

    Args:
        shapes: Shapes of the tensors to generate and combine.
    """
    manual_seed(0)
    _check_group_then_ungroup(shapes)


@mark.parametrize("shapes", UNSUPPORTED_GROUPING_SHAPES, ids=str)
def test_TensorCombiner_unsupported_group_then_ungroup(shapes: List[Tuple[int, ...]]):
    """Test unsupported cases of `group` and `ungroup`.

    Args:
        shapes: Shapes of the tensors to generate and combine.
    """
    manual_seed(0)
    with raises(NotImplementedError):
        _check_group_then_ungroup(shapes)


# first entry is shape of grouped tensor, second entry is list of shapes of ungrouped
# tensors
UNGROUPING_CASES = [
    # one tensor only
    (
        (2, 3, 4),
        [(2, 3, 4)],
    ),
    # multiple tensors of same shape
    (
        (2, 3, 4, 3),
        [(2, 3, 4), (2, 3, 4), (2, 3, 4)],
    ),
    # weight and bias of a linear layer
    (
        (4, 4),
        [(4, 3), (4,)],
    ),
    # weight and bias of a 2d convolutional layer
    (
        (4, 13),
        [(4, 3, 2, 2), (4,)],
    ),
]

UNSUPPORTED_UNGROUPING_CASES = [
    # no obvious strategy
    [
        (120,),
        [(2, 3, 4), (5,)],
    ],
    # NOTE We could support arbitrary order of bias and weight
    # bias and weight of a linear layer
    [
        (4, 4),
        [(4,), (4, 3)],
    ],
    # bias and weight of a 2d convolutional layer
    [
        (4, 13),
        [(4,), (4, 3, 2, 2)],
    ],
]


def _check_ungroup_then_group(
    grouped_shape: Tuple[int, ...], ungrouped_shapes=List[Tuple[int, ...]]
):
    """Generate a random tensor and try to ungroup group it.

    Verify that `group(ungroup) = identity`.

    Args:
        grouped_shape: Shape of the grouped tensor.
        ungrouped_shapes: Shapes of the tensors to ungroup.
    """
    # `ungroup` requires the shapes to be of type `Size`
    ungrouped_shapes = [Size(shape) for shape in ungrouped_shapes]
    tensor = rand(grouped_shape)
    ungrouped = TensorCombiner.ungroup(tensor, ungrouped_shapes)
    assert len(ungrouped) == len(ungrouped_shapes)
    assert all(t.shape == s for t, s in zip(ungrouped, ungrouped_shapes))
    # TODO We could relax this condition in cases where one wants to
    # pad smaller tensors so they can be stacked with other tensors,
    # which would increase the number of elements in the grouped tensor.
    assert tensor.numel() == sum(t.numel() for t in ungrouped)
    grouped = TensorCombiner.group(ungrouped)
    assert allclose(tensor, grouped)


@mark.parametrize("case", UNGROUPING_CASES, ids=str)
def test_TensorCombiner_ungroup_then_group(
    case: Tuple[Tuple[int, ...], List[Tuple[int, ...]]]
):
    """Test `ungroup` and `group` methods of `TensorCombiner`.

    Args:
        case: Shape of the grouped tensor and shapes of the ungrouped tensors.
    """
    manual_seed(0)
    _check_ungroup_then_group(*case)


@mark.parametrize("case", UNSUPPORTED_UNGROUPING_CASES, ids=str)
def test_TensorCombiner_unsupported_ungroup_then_group(
    case: Tuple[Tuple[int, ...], List[Tuple[int, ...]]]
):
    """Test unsupported cases of `ungroup` and `group`.

    Args:
        case: Shape of the grouped tensor and shapes of the ungrouped tensors.
    """
    manual_seed(0)
    with raises(NotImplementedError):
        _check_ungroup_then_group(*case)
