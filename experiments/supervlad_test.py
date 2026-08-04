import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    import numpy as np
    from models import SuperVLADModel
    from torchvision import transforms
    from collections import OrderedDict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")


@app.cell
def _():
    supervlad_ckpt = torch.load(
        "models/checkpoints/SuperVLAD_WithoutCrossImageEncoder.pth",
        map_location=device,
        weights_only=False,
    )
    print(supervlad_ckpt.keys())

    # Strip out module. prefix from key names
    # NOTE: Contains dino backbone weights already!
    supervlad_state_dict = OrderedDict(
        {k.replace("module.", ""): v for (k, v) in supervlad_ckpt["model_state_dict"].items()}
    )
    return supervlad_ckpt, supervlad_state_dict


@app.cell
def _(supervlad_state_dict):
    supervlad = SuperVLADModel().to(device)
    supervlad.load_state_dict(supervlad_state_dict)
    return (supervlad,)


@app.cell
def _(supervlad_state_dict):
    supervlad_keys = list(supervlad_state_dict.keys())
    supervlad_state_dict[supervlad_keys[0]].requires_grad
    return


@app.cell
def _(supervlad_state_dict):
    def compare_state_dicts(model_dict_0, model_dict_1):
        param_keys = model_dict_0.keys()
        if param_keys != model_dict_1.keys():
            raise ValueError("Dict keys don't match.")

        for k in param_keys:
            if not torch.equal(model_dict_0[k], model_dict_1[k]):
                print(f"Params not torch.equal at key: {k}")
                if not torch.allclose(model_dict_0[k], model_dict_1[k]):
                    print(
                        f"Param not torch.allclose at key: {k} {model_dict_0[k].requires_grad} {model_dict_1[k].requires_grad}"
                    )
        print("Compare finished!")

    # Compare:
    # - SuperVLAD initialized from DINOv2 backbone checkpoint vs.
    # - SuperVLAD straight from SuperVLAD checkpoint
    dino_backbone = "models/checkpoints/dinov2_vitb14_pretrain.pth"
    supervlad_dino = SuperVLADModel(backbone_path=dino_backbone).to(device)
    compare_state_dicts(supervlad_state_dict, supervlad_dino.state_dict())
    return (dino_backbone,)


@app.cell
def _(supervlad, supervlad_ckpt):
    all(
        [
            (model_state == ckpt_state.lstrip("module."), model_state, ckpt_state)
            for model_state, ckpt_state in list(
                zip(
                    (supervlad.state_dict().keys()), list(supervlad_ckpt["model_state_dict"].keys())
                )
            )
        ]
    )
    return


@app.cell
def _():
    "module.backbone.pos_embed".lstrip("module.")
    return


@app.cell
def _(supervlad_ckpt):
    supervlad_ckpt.keys()
    return


@app.cell
def _(dino_backbone):
    dino_ckpt = torch.load(dino_backbone)
    # dino_state_dict = dino_ckpt["model_state_dict"]
    dino_ckpt.keys()
    return


@app.function
def get_image_embeding(img: np.ndarray, model: SuperVLADModel, device: torch.device) -> torch.Tensor:
    """
    Args:
        img: Image of shape (H, W, C).
    """
    IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((322, 322)),  # or whatever size you trained with
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    x = transform(img).unsqueeze(0).to(device)  # [1, 3, 322, 322]
    print(f"Input size: {(x.size().numel() * x.dtype.itemsize) / (2**20)} MiB")

    return model(x)


@app.cell
def _(supervlad):
    test_img = np.random.randint(0, 255, (768, 1024, 3)).astype(np.float32)
    test_embed = get_image_embeding(test_img, supervlad, device)
    print(f"{test_embed.shape=} {test_embed.norm().item()=}")
    return


@app.cell
def _(supervlad):
    import cv2 as cv
    import faiss
    from dataclasses import dataclass
    from pathlib import Path

    @dataclass
    class ImageData:
        idx: int
        path: str
        pixels: np.ndarray | None = None
        embedding: torch.Tensor | None = None

    # Common Pitfalls w/ FAISS:
    # - Data Types: Always use float32; float64 often causes errors. 
    # - Shape: Inputs must be C-contiguous 2D arrays (N, d), not 1D rows (d,). 
    # - IVF Training: IndexIVF indexes must be trained on a representative sample before adding data. 

    def ingest_image_embeddings(img_dir: str, ext: str = "jpg"):
        img_dir = Path(img_dir)
        img_data_list = [ImageData(idx, path) for idx, path in enumerate(sorted(img_dir.glob(f"*.{ext}")))]
    
        d = 3072  # n_vlad_clusters=4 * feature_dim=768
        index = faiss.IndexFlatL2(d)

        for img_data in img_data_list:
            img = cv.imread(img_data.path, cv.IMREAD_COLOR_RGB)
            img_embed = get_image_embeding(img, supervlad, device).cpu().detach().numpy()
        
            index.add(img_embed)
            # print(f"{img_data.idx} {img_embed.shape} {img_embed.dtype}")
            # index.add_with_ids(img_embed, np.array([img_data.idx]))
        
            img_data.pixels = img
            img_data.embedding = img_embed

        return img_data_list, index

    img_dir, ext = "data/raw/statue_orbit/", "jpg"
    img_data_list, db_index = ingest_image_embeddings(img_dir, ext)
    return Path, db_index, img_data_list


@app.cell
def _(db_index, img_data_list):
    def query_all(img_data_list, k=5):
        # QUERY VECTOR DB for similarity
        search_results = []
        for i, img in enumerate(img_data_list):
            distance, indices = db_index.search(img.embedding, k)
            search_results.append((img.idx, img.path, distance, indices))

        return search_results

    k = 5
    search_results = query_all(img_data_list, k)
    print(f"Query Index --> Top-{k} Matching Indices")
    for results in search_results:
        print(f"{results[0]:2d} --> {results[-1][0][1:]} \t|\t {results[-2][0][1:]}")
    return k, search_results


@app.cell
def _(img_data_list):
    heading = mo.md(r"## Show Top-K Most Similar Images to Query")
    img_index_number = mo.ui.number(label="Query Image Index: ", start=0, stop=len(img_data_list), step=1)

    mo.vstack((heading, img_index_number))
    return (img_index_number,)


@app.cell
def _(Path, img_data_list, img_index_number, k, search_results):
    search_idx = img_index_number.value
    idx, img_path, distances, indices = search_results[search_idx]
    print(f"{idx=} {Path(img_path).name}\n{distances[0]=}\n{indices[0]=}")

    resize = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(512),
        ]
    )

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, k, figsize=(12, 6))
    for i, img_idx in enumerate(indices.squeeze()):
        img = resize(img_data_list[img_idx].pixels)
        ax[i].imshow(img)
        ax[i].set_title(f"d={float(distances[0][i]):.2f} ({int(img_idx)})", fontsize=10)
        ax[i].axis("off")

    plt.show()
    return


@app.cell
def _(img_data_list):
    img_data_list
    return


if __name__ == "__main__":
    app.run()
