"""Test saving and loading the optimizer for checkpointing."""

from test.utils import compare_optimizers
from typing import Tuple

from torch import load, manual_seed, save
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, Module, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from sirfshampoo.optimizer import SIRFShampoo


def setup() -> Tuple[Sequential, Module, SIRFShampoo]:
    """Set up the model, loss function, and optimizer.

    Returns:
        A tuple containing the model, loss function, and optimizer.
    """
    model = Sequential(
        Conv2d(1, 3, kernel_size=5, stride=2),
        ReLU(),
        Flatten(),
        Linear(432, 200),
        ReLU(),
        Linear(200, 50),
        ReLU(),
        Linear(50, 10),
    )
    loss_func = CrossEntropyLoss()
    optimizer = SIRFShampoo(model)
    return model, loss_func, optimizer


def test_checkpointing():
    """Check whether optimizer is saved/restored correctly while training."""
    manual_seed(0)  # make deterministic
    MAX_STEPS = 100  # quit training after this many steps

    BATCH_SIZE = 32
    train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    model, loss_func, optimizer = setup()
    checkpoints = [10, 33, 50]

    # Loop over each batch from the training set
    for batch_idx, (inputs, target) in enumerate(train_loader):
        print(f"Step {optimizer.global_step}")

        # Save model and optimizer, then restore and compare with original ones
        if batch_idx in checkpoints:
            # keep a reference to compare with restored optimizer
            original = optimizer

            print("Saving checkpoint")
            save(optimizer.state_dict(), f"optimizer_checkpoint_{batch_idx}.pt")
            save(model.state_dict(), f"model_checkpoint_{batch_idx}.pt")
            print("Deleting model and optimizer")
            del model, optimizer

            print("Loading checkpoint")
            model, _, optimizer = setup()
            optimizer.load_state_dict(load(f"optimizer_checkpoint_{batch_idx}.pt"))
            model.load_state_dict(load(f"model_checkpoint_{batch_idx}.pt"))

            # compare restored and copied optimizer
            compare_optimizers(optimizer, original)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Backward pass
        loss = loss_func(model(inputs), target)
        loss.backward()

        # Update parameters
        optimizer.step()

        if batch_idx >= MAX_STEPS:
            break
