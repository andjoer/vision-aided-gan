import pytest
from prepare_disc import prepare_cv_ensamble_mps
import torch
import vision_aided_loss

@pytest.mark.parametrize("cv_type", ["swin", "clip", "dino","swin+clip"])
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
    net_disc = vision_aided_loss.Discriminator(
        cv_type=cv_type, loss_type="multilevel_sigmoid_s", device=str(device)
    )
    net_disc.to(device)
    net_disc = prepare_cv_ensamble_mps(cv_type, net_disc)

    # Verify that the specified layers have been moved to the CPU
    cv_list = cv_type.split('+')
    for idx, cv_model in enumerate(net_disc.cv_ensemble.models):
        if cv_list[idx] == 'clip':
            assert (
                cv_model.model.conv1.weight.device.type == "cpu"
            ), "CLIP model's conv1 is not on the CPU."
        elif cv_list[idx] == 'dino':
            assert (
                cv_model.model.patch_embed.proj.weight.device.type == "cpu"
            ), "DINO model's patch_embed.proj is not on the CPU."
        elif cv_list[idx] == 'swin':
            assert (
                next(cv_model.model.patch_embed.proj.parameters()).device.type == "cpu"
            ), "Swin model's patch_embed is not on the CPU."

    # Run a forward pass. We are only verifying that this executes without error.
    with torch.no_grad():
        # The discriminator might return a "loss" or some intermediate output
        output = net_disc(input_tensor, for_real=False)

    # Basic checks: the output should be a torch.Tensor
    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor."

    # Check that the output has the same dtype and device as the input
    assert output.dtype == input_tensor.dtype, "Output tensor dtype differs from input tensor dtype."
    assert output.device == input_tensor.device, "Output tensor device differs from input tensor device."

