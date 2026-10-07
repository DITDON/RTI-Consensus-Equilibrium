import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from denoiser import RTIDenoiser


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GRID_SIZE = 30
NUM_SAMPLES = 1000
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def create_smooth_target():
    target = np.zeros(
        (GRID_SIZE, GRID_SIZE),
        dtype=np.float32
    )

    num_targets = np.random.randint(1, 6)

    for _ in range(num_targets):

        cx = np.random.randint(4, GRID_SIZE - 4)
        cy = np.random.randint(4, GRID_SIZE - 4)

        sigma = np.random.uniform(1.0, 3.0)

        amplitude = np.random.uniform(
            0.7,
            1.0
        )

        y, x = np.meshgrid(
            np.arange(GRID_SIZE),
            np.arange(GRID_SIZE),
            indexing="ij"
        )

        blob = amplitude * np.exp(
            -(
                (x - cx) ** 2
                +
                (y - cy) ** 2
            )
            /
            (2 * sigma ** 2)
        )

        target += blob

    target = target / (
        np.max(target) + 1e-8
    )

    return target.astype(np.float32)


def create_training_data():

    clean_images = []
    noisy_images = []

    print("=" * 60)
    print("GENERATING CNN TRAINING DATA")
    print("=" * 60)

    for i in range(NUM_SAMPLES):

        clean = create_smooth_target()

        noise_level = np.random.uniform(
            0.05,
            0.25
        )

        noise = np.random.normal(
            0,
            noise_level,
            clean.shape
        ).astype(np.float32)

        noisy = clean + noise

        noisy = np.clip(
            noisy,
            0,
            1
        )

        clean_images.append(clean)
        noisy_images.append(noisy)

    clean_images = np.array(
        clean_images,
        dtype=np.float32
    )

    noisy_images = np.array(
        noisy_images,
        dtype=np.float32
    )

    clean_images = torch.tensor(
        clean_images
    ).unsqueeze(1)

    noisy_images = torch.tensor(
        noisy_images
    ).unsqueeze(1)

    print(
        "Clean data shape :",
        clean_images.shape
    )

    print(
        "Noisy data shape :",
        noisy_images.shape
    )

    return noisy_images, clean_images


def train():

    print()
    print("=" * 60)
    print("RTI CNN DENOISER TRAINING")
    print("=" * 60)

    print("Device :", DEVICE)
    print("Samples:", NUM_SAMPLES)
    print("Epochs :", EPOCHS)
    print("Batch  :", BATCH_SIZE)

    noisy, clean = create_training_data()

    dataset = torch.utils.data.TensorDataset(
        noisy,
        clean
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = RTIDenoiser().to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print()
    print("CNN model created.")
    print()
    print("Starting training...")
    print("-" * 60)

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for noisy_batch, clean_batch in dataloader:

            noisy_batch = noisy_batch.to(
                DEVICE
            )

            clean_batch = clean_batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            output = model(
                noisy_batch
            )

            loss = criterion(
                output,
                clean_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * noisy_batch.size(0)
            )

        average_loss = (
            total_loss
            / len(dataset)
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS} "
            f"| Loss = "
            f"{average_loss:.6f}"
        )

    os.makedirs(
        "results",
        exist_ok=True
    )

    model_path = (
        "results/"
        "cnn_denoiser.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Model saved:",
        model_path
    )

    print(
        "Device:",
        DEVICE
    )


if __name__ == "__main__":
    train()