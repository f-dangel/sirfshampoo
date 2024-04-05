"""Tests for sirfshampoo/__init__.py."""

import time

import pytest

import sirfshampoo

NAMES = ["world", "github"]
IDS = NAMES


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello(name: str):
    """Test hello function.

    Args:
        name: Name to greet.
    """
    sirfshampoo.hello(name)


@pytest.mark.expensive
@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello_expensive(name: str):
    """Expensive test of hello. Will only be run on master/main and development.

    Args:
        name: Name to greet.
    """
    time.sleep(1)
    sirfshampoo.hello(name)
