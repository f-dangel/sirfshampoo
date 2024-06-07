"""Utility functions for tests."""

from torch import cuda, device

DEVICES = [device("cpu")] + [device(f"cuda:{i}") for i in range(cuda.device_count())]
DEVICE_IDS = [f"dev={str(dev)}" for dev in DEVICES]
