import logging
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from dpt import DPT
from tqdm import tqdm


def silog_loss(pred, y, lambda_=0.85):
    mask = (y > 0) & (pred > 0)
    d = torch.log(pred[mask]) - torch.log(y[mask])
    return torch.sqrt((d**2).mean() - lambda_ * (d.mean() ** 2))


class Trainer:

    def __init__(self, args, model):
        self.args = args
        self.model = model

        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        self.logger = logging

        self.logger.info(
            f"self.model.parameters={sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

    def train(self, dataloaders):
        hist = defaultdict(list)
        self.model.to(self.args.device)
        pbar = tqdm(total=self.args.epochs, desc="Training")
        for epoch in range(self.args.epochs):
            self.model.train()
            for batch in dataloaders["train"]:
                self.optimizer.zero_grad()
                images = batch["image"].to(self.args.device)
                outputs = self.model(images)
                loss = self.compute_loss(outputs, batch)
                loss.backward()
                self.optimizer.step()
                hist["train_loss"].append(loss.item())
            self.model.eval()
            for batch in dataloaders["val"]:
                images = batch["image"].to(self.args.device)
                outputs = self.model(images)
                loss = self.compute_loss(outputs, batch)
                hist["val_loss"].append(loss.item())
            pbar.set_postfix(
                train_loss=np.mean(hist["train_loss"]),
                val_loss=np.mean(hist["val_loss"]),
            )
            pbar.update(1)
        return hist

    def compute_loss(self, outputs, batch):
        return silog_loss(outputs["depth"], batch["depth"])


def main(args):
    depth_path = "./sample_data/2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png"
    rgb_path = depth_path.replace("groundtruth_depth", "image")
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 256.0  # to m
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    rgb = np.array(rgb)

    rgb_transform = (
        lambda x: torch.from_numpy(np.array(x)).float().permute(2, 0, 1) / 255.0
    )
    x = rgb_transform(rgb).unsqueeze(0)
    y = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)

    target_hw = (384, 384)
    x = torchvision.transforms.CenterCrop(target_hw)(x)
    y = torchvision.transforms.CenterCrop(target_hw)(y)

    hw = x.shape[-2:]
    dpt = DPT(hw=hw)
    trainer = Trainer(args=args, model=dpt)

    train_loader = [
        {
            "image": x.to(args.device),
            "depth": y.to(args.device),
        }
    ]
    val_loader = train_loader

    dataloaders = {"train": train_loader, "val": val_loader}
    hist = trainer.train(dataloaders=dataloaders)

    # show preds
    pred = dpt(x.to(args.device))["depth"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    im = axs[0].imshow(pred[0, 0].detach().cpu().numpy(), cmap="plasma")
    axs[0].set_title("Predicted")
    im2 = axs[1].imshow(y[0, 0].detach().cpu().numpy(), cmap="plasma")
    axs[1].set_title("GT")
    plt.colorbar(im, orientation="vertical", ax=axs[0], fraction=0.02, pad=0.04)
    plt.colorbar(im2, orientation="vertical", ax=axs[1], fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.show()
    # show losses
    plt.figure(figsize=(10, 5))
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    main(args=args)
