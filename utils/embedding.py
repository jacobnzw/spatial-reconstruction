from collections import OrderedDict
from pathlib import Path

import torch
from torchvision import transforms

from models import SuperVLADModel

from .view import ViewData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUPERVLAD_CHECKPOINT = "models/checkpoints/SuperVLAD_WithoutCrossImageEncoder.pth"


class ViewEmbedder:
    """Creates image embeddings for viewpoint similarity search."""

    def __init__(self, model_checkpoint: str = SUPERVLAD_CHECKPOINT):
        self.model = self._load_embedder(model_checkpoint)

    def _load_embedder(self, checkpoint: str) -> SuperVLADModel:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"SuperVLAD checkpoint not found at {checkpoint}. Please download it from the official repository."
            )

        supervlad_ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        # Strip the "module." prefix from the keys in the state dict
        supervlad_state_dict = OrderedDict(
            {k.replace("module.", ""): v for (k, v) in supervlad_ckpt["model_state_dict"].items()}
        )
        model = SuperVLADModel().eval().to(device)
        model.load_state_dict(supervlad_state_dict)
        return model

    def __call__(self, view: ViewData) -> ViewData:
        """Compute and attach the embedding for a given view."""

        IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((322, 322)),  # or whatever size you trained with
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        x = transform(view.pixels).unsqueeze(0).to(device)  # [1, 3, 322, 322]

        with torch.inference_mode():
            view.embedding = self.model(x).cpu().numpy()  # [1, 3072]

        return view
