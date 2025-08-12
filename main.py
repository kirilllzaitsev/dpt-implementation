import logging
from collections import defaultdict

import numpy as np
import torch
from dpt import DPT
from tqdm import tqdm


class Trainer:

    def __init__(self, args, model):
        self.args = args
        self.model = model

        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        self.logger = logging

    def train(self, dataloaders):
        hist = defaultdict(list)
        self.model.to(self.args.device)
        for epoch in tqdm(range(self.args.epochs)):
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
            self.logger.info(
                f"Epoch {epoch}: Train Loss: {np.mean(hist['train_loss']):.4f}, Val Loss: {np.mean(hist['val_loss']):.4f}"
            )
        return hist

    def compute_loss(self, outputs, batch):
        return torch.nn.functional.l1_loss(outputs["depth"], batch["depth"])


def main(args):
    x = torch.randn(1, 3, 256, 256)
    hw = x.shape[-2:]
    dpt = DPT(hw=hw)
    trainer = Trainer(args=args, model=dpt)

    train_loader = [
        {
            "image": x.to(args.device),
            "depth": torch.randn(1, 1, 256, 256).to(args.device),
        }
    ]
    val_loader = train_loader

    dataloaders = {"train": train_loader, "val": val_loader}
    trainer.train(dataloaders=dataloaders)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    main(args=args)
