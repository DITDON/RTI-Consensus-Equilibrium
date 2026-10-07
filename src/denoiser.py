import torch
import torch.nn as nn


class RTIDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.network(x)


def test_model():
    print("=" * 50)
    print("RTI CNN DENOISER TEST")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    model = RTIDenoiser().to(device)

    test_input = torch.randn(1, 1, 30, 30).to(device)

    output = model(test_input)

    print("Input shape :", test_input.shape)
    print("Output shape:", output.shape)

    print()
    print("CNN model created successfully.")
    print("=" * 50)


if __name__ == "__main__":
    test_model()