# test_vision_aided_loss

import pytest
import torch
import vision_aided_loss

@pytest.mark.parametrize("cv_type", ["swin", "clip", "dino", "vgg","swin+clip"])
def test_discriminator_forward(cv_type):
    # Decide on a device (selection logic: GPU -> MPS -> CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Create a random tensor (1 batch, 3 channels, 224x224 image)
    input_tensor = torch.randn(1, 3, 224, 224, device=device)

    # Initialize the Discriminator with the given cv_type
    net_disc = vision_aided_loss.Discriminator(cv_type=cv_type, loss_type = "multilevel_sigmoid_s", device=str(device))
    net_disc.to(device)
    # Run a forward pass. We are only verifying that this executes without error.
    with torch.no_grad():
        # The discriminator might return a "loss" or some intermediate output
        output = net_disc(input_tensor,for_real=False)

    # Basic checks: the output should be a torch.Tensor
    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor."

    # Check that the output has the same dtype and device as the input
    assert output.dtype == input_tensor.dtype, "Output tensor dtype differs from input tensor dtype."
    assert output.device == input_tensor.device, "Output tensor device differs from input tensor device."

@pytest.mark.parametrize("policy,expected_policy", [
    (None, 'color,translation,cutout'),  # Default behavior
    ('', ''),                           # Empty string
    ('color', 'color')                  # Specific policy
])
def test_discriminator_policy(policy, expected_policy):
    # Decide on a device (selection logic: GPU -> MPS -> CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Create the Discriminator with the given policy
    kwargs = {"policy": policy} if policy is not None else {}
    net_disc = vision_aided_loss.Discriminator(
        cv_type="swin", 
        loss_type="multilevel_sigmoid_s", 
        device=str(device), 
        **kwargs
    )

    # Check that the policy in the CVBackbone matches the expected policy
    assert net_disc.cv_ensemble.policy == expected_policy, (
        f"Policy mismatch: Expected '{expected_policy}', got '{net_disc.cv_ensemble.policy}'"
    )

    net_disc.to(device)
    # Create a random tensor (1 batch, 3 channels, 224x224 image)
    input_tensor = torch.randn(1, 3, 224, 224, device=device)
    # Run a forward pass. We are only verifying that this executes without error.
    with torch.no_grad():
        # The discriminator might return a "loss" or some intermediate output
        output = net_disc(input_tensor,for_real=False)

    # Basic checks: the output should be a torch.Tensor
    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor."

    # Check that the output has the same dtype and device as the input
    assert output.dtype == input_tensor.dtype, "Output tensor dtype differs from input tensor dtype."
    assert output.device == input_tensor.device, "Output tensor device differs from input tensor device."