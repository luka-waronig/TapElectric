from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Swin2SRForImageSuperResolution

MODEL_ID = "caidas/swin2SR-classical-sr-x2-64"
MODEL_REVISION = "9a57888931c46364d7df003ca9b7a27347ea9539"
UPSCALE_FACTOR = 2


class Swin2SRRestorer:
    def __init__(self, model: Swin2SRForImageSuperResolution, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device

    @classmethod
    def from_pretrained(cls, device: str = "auto") -> "Swin2SRRestorer":
        model = Swin2SRForImageSuperResolution.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            use_safetensors=True,
        )
        return cls(model, resolve_device(device))

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "auto") -> "Swin2SRRestorer":
        model = Swin2SRForImageSuperResolution.from_pretrained(path, use_safetensors=True)
        return cls(model, resolve_device(device))

    def restore(self, image: Image.Image) -> Image.Image:
        self.model.eval()
        pixel_values = image_to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            reconstruction = self.model(pixel_values=pixel_values).reconstruction
        expected_height = image.height * UPSCALE_FACTOR
        expected_width = image.width * UPSCALE_FACTOR
        reconstruction = reconstruction[0, :, :expected_height, :expected_width]
        return tensor_to_image(reconstruction)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    pixels = tensor.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(pixels)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
