import logging
import math
from dataclasses import dataclass

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

from models.vision_transformer import vit_base


# TODO: rename to SuperVLADLayer
class SuperVLAD(nn.Module):
    """SuperVLAD layer implementation"""

    def __init__(self, clusters_num=64, ghost_clusters_num=1, dim=128, normalize_input=True, work_with_tokens=False):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        clusters_num += ghost_clusters_num
        self.clusters_num = clusters_num
        self.ghost_clusters_num = ghost_clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        if work_with_tokens:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        # self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        # self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3)
            residual = residual * soft_assign[:, D : D + 1, :].unsqueeze(2)
            vlad[:, D : D + 1, :] = residual.sum(dim=-1)
        vlad = vlad[:, : -self.ghost_clusters_num, :]
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def initialize_supervlad_layer(self, args, cluster_ds, model):
        backbone = model.backbone
        descriptors_num = 500000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(np.random.choice(len(cluster_ds), images_num, replace=False))
        random_dl = DataLoader(
            dataset=cluster_ds, num_workers=args.num_workers, batch_size=args.infer_batch_size, sampler=random_sampler
        )
        with torch.no_grad():
            backbone = backbone.eval()
            logging.debug("Extracting features to initialize SuperVLAD layer")
            descriptors = np.zeros(shape=(descriptors_num, args.features_dim), dtype=np.float32)
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to(args.device)
                outputs = backbone(inputs)

                ######### for the DINOv2 backbone ###########
                B, P, D = outputs["x_prenorm"].shape
                W = H = int(math.sqrt(P - 1))
                outputs = outputs["x_norm_patchtokens"].view(B, W, H, D).permute(0, 3, 1, 2)

                ######### for the CCT backbone ###########
                # outputs = outputs.view(-1,24,24,384).permute(0, 3, 1, 2)

                ######### for the ViT backbone ###########
                # B,P,D = outputs.last_hidden_state.shape
                # W = H = int(math.sqrt(P-1))
                # outputs = outputs.last_hidden_state[:, 1:, :].view(B,W,H,D).permute(0, 3, 1, 2)

                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(norm_outputs.shape[0], args.features_dim, -1).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    startix = batchix + ix * descs_num_per_image
                    descriptors[startix : startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(args.features_dim, self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
        logging.debug(f"All clusters shape: {kmeans.centroids.shape}")
        self.init_params(kmeans.centroids, descriptors)
        self = self.to(args.device)


@dataclass
class SuperVLADArgs:
    work_with_tokens: bool = False
    # Output dimension of fully connected layer. If None, don't use a fully connected layer
    fc_output_dim: int | None = None
    # TODO: freeze first freeze_te blocks of the backbone?
    freeze_te: int = 10  # between [0, 10]
    # Number of clusters for SuperVLAD layer
    supervlad_clusters: int = 4
    # Number of ghost clusters for SuperVLAD layer
    ghost_clusters: int = 1
    features_dim: int = 768  # TODO: remove? gets set in get_backbone()
    crossimage_encoder: bool = False


# TODO: rename to SuperVLAD()
class SuperVLADModel(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer."""

    def __init__(self, args: SuperVLADArgs = SuperVLADArgs(), backbone_path=None):
        super().__init__()
        self.backbone = get_backbone(args, backbone_path)
        self.arch_name = "dino"  # args.backbone
        self.crossimage_encoder = args.crossimage_encoder
        self.aggregation = SuperVLAD(
            clusters_num=args.supervlad_clusters,
            ghost_clusters_num=args.ghost_clusters,
            dim=args.features_dim,
            work_with_tokens=args.work_with_tokens,
        )

        if self.crossimage_encoder:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=768, nhead=16, dim_feedforward=2048, activation="gelu", dropout=0.1, batch_first=False
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)  # Cross-image encoder

        if args.fc_output_dim != None:
            # Concatenate fully connected layer to the aggregation layer
            self.aggregation = nn.Sequential(
                self.aggregation, nn.Linear(args.features_dim, args.fc_output_dim), L2Norm()
            )
            args.features_dim = args.fc_output_dim

    def forward(self, x, queryflag=0):
        x = self.backbone(x)

        if self.arch_name.startswith("vit"):
            B, P, D = x.last_hidden_state.shape
            W = H = int(math.sqrt(P - 1))
            x1 = x.last_hidden_state[:, 1:, :].view(B, W, H, D).permute(0, 3, 1, 2)
            x = self.aggregation(x1)
        elif self.arch_name.startswith("cct"):
            B, P, D = x.shape
            x = x.view(-1, 24, 24, 384)
            x = x.permute(0, 3, 1, 2)
            x = self.aggregation(x)
        elif self.arch_name.startswith("dino"):
            B, P, D = x["x_prenorm"].shape
            W = H = int(math.sqrt(P - 1))
            x0 = x["x_norm_clstoken"]
            x1 = x["x_norm_patchtokens"].view(B, W, H, D).permute(0, 3, 1, 2)
            x = self.aggregation(x1)
        else:
            x = self.aggregation(x)

        if self.crossimage_encoder:
            x = self.encoder(x.view(B, -1, D)).view(B, -1)

        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


def get_backbone(args, backbone_path: str | None = None):
    backbone = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0)
    if backbone_path is not None:
        model_dict = backbone.state_dict()
        state_dict = torch.load(backbone_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)

    if args.freeze_te:
        for p in backbone.parameters():
            p.requires_grad = False
        for name, child in backbone.blocks.named_children():
            if int(name) >= args.freeze_te:
                for params in child.parameters():
                    params.requires_grad = True

    args.features_dim = 768  # 1024
    return backbone


def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]


def get_image_embeding(img: np.ndarray, model: SuperVLADModel, device: torch.device):
    IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((322, 322)),  # or whatever size you trained with
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    x = transform(img).unsqueeze(0).to(device)  # [1, 3, 322, 322]

    with torch.no_grad():
        emb = model(x)  # [1, 3072]
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
