from torch.utils.data import random_split, DataLoader
from inference_functions import compute_eval_stats
from transforms import trainval_transforms
from configs import SEED, BATCH_SIZE
from train_functions import train
from dataset import ImageDataset
from loss_function import Loss
from pathlib import Path
from model import Model
import argparse
import torch
import time

def main():
    # add command line argument to resume training at a certain checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from."
    )
    args = parser.parse_args()

    annot_file_trainval = Path("../data/preprocessed/trainval/annotations.csv")
    img_dir_trainval = Path("../data/preprocessed/trainval/Images")

    # 1. Datasets and DataLoaders
    trainval_dataset = ImageDataset(annot_file_trainval, img_dir_trainval,
                                    transform=trainval_transforms)

    generator_ = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(trainval_dataset, [0.8, 0.2]
                                              ,generator=generator_)

    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4,
                        pin_memory=True)

    # 2. instantiate model, loss function, device, optimizer, and scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model().to(device, non_blocking=True)
    loss_fn = Loss().to(device, non_blocking=True)

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    start_epoch = 0
    num_epochs = 75

    if args.resume is not None:
        checkpoint = torch.load(args.resume)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"]

        print(f"Resuming from epoch {start_epoch}")

    # 3. training loop
    checkpoint_dir = Path("../outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        print(f"Starting Epoch {epoch+1}")

        # track time it takes for one epoch to complete
        total_start_time = time.perf_counter()

        train_start_time = time.perf_counter()
        # a. train model
        train_loss = train(model, loss_fn, optimizer, train_dl, device)

        train_end_time = time.perf_counter()

        print(F"Training time: {(train_end_time - train_start_time):.4f}")

        val_start_time = time.perf_counter()
        # b. evaluate model performance on validation dataset every 5 epochs, but
        # track validation loss every epoch
        if (epoch+1) % 5 == 0:
            val_mAP, val_loss = compute_eval_stats(model, val_dl, device, loss_fn)

        else:
            val_loss = compute_eval_stats(model, val_dl, device, loss_fn,
                                          loss_only=True)
            val_mAP = None

        val_end_time = time.perf_counter()

        print(f"Validation time: {(val_end_time - val_start_time):.4f}")
        # c. save checkpoints
        checkpoint = {
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss
        }

        torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")

        # d. display statistics
        print(f"(Epoch {epoch+1}) Training: Loss - {train_loss:.4f} "
              f"| Validation: Loss - {val_loss:.4f} mAP - {val_mAP}")

        # e. display epoch duration
        total_end_time = time.perf_counter()
        print(f"Epoch took {(total_end_time - total_start_time):.4f} seconds to complete.")

if __name__ == "__main__":
    main()