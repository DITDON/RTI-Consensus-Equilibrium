import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# PATH SETUP
# ============================================================

sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from denoiser import RTIDenoiser

from ce_solver import consensus_equilibrium

from rti_model import (
    generate_uniform_sensors,
    generate_links,
    generate_grid,
    build_weight_matrix
)

from targets import create_smooth_targets

from metrics import calculate_all_metrics


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

AREA_WIDTH = 7.0
AREA_HEIGHT = 7.0

NUM_SENSORS = 24

GRID_SIZE = 30

NUM_TARGETS = 3

NOISE_LEVEL = 0.02

RANDOM_SEED = 42

# Tikhonov regularization
LAMBDA_REG = 0.1

# Consensus Equilibrium
CE_ITERATIONS = 20
CE_DATA_STEP = 0.1
CE_RELAXATION = 0.5

# CNN model
MODEL_PATH = os.path.join(
    "results",
    "cnn_denoiser.pth"
)

# Results directory
RESULTS_DIR = "results"


# ============================================================
# FIXED TARGET LOCATIONS
# ============================================================

TARGET_CENTERS = [
    (1.75, 1.75),
    (5.25, 1.75),
    (3.50, 5.25)
]

SMOOTH_SIGMA = 0.45


# ============================================================
# UTILITY: NORMALIZE IMAGE
# ============================================================

def normalize_image(image):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum - minimum < 1e-12:

        return np.zeros_like(
            image
        )

    normalized = (
        image - minimum
    ) / (
        maximum - minimum
    )

    return normalized


# ============================================================
# UTILITY: PRINT METRICS
# ============================================================

def print_metrics(
    method_name,
    metrics
):

    print()
    print(
        method_name
    )

    print(
        "-" * 50
    )

    for key, value in metrics.items():

        if key.upper() == "PSNR":

            print(
                f"{key.upper():5s} : "
                f"{value:.4f} dB"
            )

        else:

            print(
                f"{key.upper():5s} : "
                f"{value:.4f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "        CONSENSUS EQUILIBRIUM + CNN DENOISER"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # RANDOM SEED
    # ========================================================

    np.random.seed(
        RANDOM_SEED
    )

    torch.manual_seed(
        RANDOM_SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )

    print()
    print(
        f"Random seed : {RANDOM_SEED}"
    )


    # ========================================================
    # [1] DEVICE
    # ========================================================

    print()
    print(
        "[1] Device"
    )

    print(
        "-" * 50
    )

    print(
        "Device:",
        DEVICE
    )


    # ========================================================
    # [2] SENSOR NETWORK
    # ========================================================

    print()
    print(
        "[2] Generating sensor network..."
    )

    print(
        "-" * 50
    )

    sensors = generate_uniform_sensors(
        NUM_SENSORS,
        width=AREA_WIDTH,
        height=AREA_HEIGHT
    )

    links = generate_links(
        NUM_SENSORS
    )

    print(
        "Sensors:",
        NUM_SENSORS
    )

    print(
        "Links  :",
        len(links)
    )


    # ========================================================
    # [3] RECONSTRUCTION GRID
    # ========================================================

    print()
    print(
        "[3] Creating reconstruction grid..."
    )

    print(
        "-" * 50
    )

    X, Y, grid_points = generate_grid(
        grid_size=GRID_SIZE,
        width=AREA_WIDTH,
        height=AREA_HEIGHT
    )

    print(
        "Grid   :",
        f"{GRID_SIZE} x {GRID_SIZE}"
    )

    print(
        "Voxels :",
        GRID_SIZE * GRID_SIZE
    )


    # ========================================================
    # [4] RTI WEIGHT MATRIX
    # ========================================================

    print()
    print(
        "[4] Building RTI weight matrix..."
    )

    print(
        "-" * 50
    )

    W = build_weight_matrix(
        sensors,
        links,
        grid_points
    )

    W = np.asarray(
        W,
        dtype=np.float32
    )

    print(
        "W shape:",
        W.shape
    )


    # ========================================================
    # [5] GROUND TRUTH
    # ========================================================

    print()
    print(
        "[5] Creating ground truth..."
    )

    print(
        "-" * 50
    )

    print(
        "Target type : Smooth Gaussian"
    )

    print(
        "Target count:",
        NUM_TARGETS
    )

    print(
        "Target centers:",
        TARGET_CENTERS
    )

    ground_truth = create_smooth_targets(
        grid_points,
        target_centers=TARGET_CENTERS,
        sigma=SMOOTH_SIGMA,
        amplitude=1.0
    )

    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float32
    )

    # Normalize ground truth
    ground_truth = normalize_image(
        ground_truth
    )

    # Convert vector -> image
    ground_truth_image = (
        ground_truth.reshape(
            GRID_SIZE,
            GRID_SIZE
        )
    )

    # Vector used by forward model
    ground_truth_vector = (
        ground_truth.reshape(-1)
    )

    print(
        "Ground truth shape:",
        ground_truth_image.shape
    )

    print(
        "Ground truth max  :",
        f"{np.max(ground_truth):.4f}"
    )


    # ========================================================
    # [6] FORWARD RTI MODEL + RSS MEASUREMENTS
    # ========================================================

    print()
    print(
        "[6] Generating RSS measurements..."
    )

    print(
        "-" * 50
    )

    # Clean measurements
    clean_measurement = (
        W @ ground_truth_vector
    )

    # Noise standard deviation
    noise_std = (
        NOISE_LEVEL
        * np.std(clean_measurement)
    )

    # Gaussian noise
    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=clean_measurement.shape
    ).astype(
        np.float32
    )

    # Noisy RSS measurements
    y_measurements = (
        clean_measurement
        + noise
    ).astype(
        np.float32
    )

    print(
        "Measurement shape:",
        y_measurements.shape
    )

    print(
        "Noise level      :",
        NOISE_LEVEL
    )

    print(
        "Noise std        :",
        f"{noise_std:.6f}"
    )


    # ========================================================
    # [7] INITIAL TIKHONOV RECONSTRUCTION
    # ========================================================

    print()
    print(
        "[7] Creating initial Tikhonov reconstruction..."
    )

    print(
        "-" * 50
    )

    # W^T W
    WT_W = (
        W.T @ W
    )

    # Identity matrix
    identity = np.eye(
        W.shape[1],
        dtype=np.float32
    )

    # W^T y
    WT_y = (
        W.T @ y_measurements
    )

    # Tikhonov:
    #
    # x = (W^T W + lambda I)^(-1) W^T y

    initial_reconstruction = np.linalg.solve(
        WT_W
        + LAMBDA_REG * identity,
        WT_y
    )

    initial_reconstruction = (
        initial_reconstruction
        .astype(
            np.float32
        )
    )

    # Remove negative values
    initial_reconstruction = np.clip(
        initial_reconstruction,
        0.0,
        None
    )

    # Convert to image
    initial_image = (
        initial_reconstruction
        .reshape(
            GRID_SIZE,
            GRID_SIZE
        )
    )

    # Normalize for image comparison
    initial_image = normalize_image(
        initial_image
    )

    print(
        "Lambda:",
        LAMBDA_REG
    )

    print(
        "Initial reconstruction created."
    )


    # ========================================================
    # [8] INITIAL RECONSTRUCTION METRICS
    # ========================================================

    print()
    print(
        "[8] Calculating initial reconstruction metrics..."
    )

    print(
        "-" * 50
    )

    initial_metrics = calculate_all_metrics(
        ground_truth_image,
        initial_image
    )

    print_metrics(
        "INITIAL TIKHONOV",
        initial_metrics
    )


    # ========================================================
    # [9] LOAD TRAINED CNN
    # ========================================================

    print()
    print(
        "[9] Loading trained CNN..."
    )

    print(
        "-" * 50
    )

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nCNN model not found:\n"
            f"{MODEL_PATH}\n\n"
            f"Run this first:\n"
            f"python src/train_denoiser.py"
        )

    model = RTIDenoiser().to(
        DEVICE
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print(
        "CNN loaded successfully."
    )

    print(
        "Model:",
        MODEL_PATH
    )


    # ========================================================
    # [10] CONSENSUS EQUILIBRIUM
    # ========================================================

    print()
    print(
        "[10] Running Consensus Equilibrium..."
    )

    print(
        "-" * 50
    )

    print(
        "Iterations :",
        CE_ITERATIONS
    )

    print(
        "Data step  :",
        CE_DATA_STEP
    )

    print(
        "Relaxation :",
        CE_RELAXATION
    )

    print()

    ce_reconstruction, history = (
        consensus_equilibrium(
            x0=initial_reconstruction,
            W=W,
            y=y_measurements,
            denoiser=model,
            device=DEVICE,
            iterations=CE_ITERATIONS,
            data_step=CE_DATA_STEP,
            relaxation=CE_RELAXATION
        )
    )


    # ========================================================
    # PROCESS CE OUTPUT
    # ========================================================

    ce_reconstruction = np.asarray(
        ce_reconstruction,
        dtype=np.float32
    )

    # Remove negative values
    ce_reconstruction = np.clip(
        ce_reconstruction,
        0.0,
        None
    )

    # Convert to image
    ce_image = (
        ce_reconstruction
        .reshape(
            GRID_SIZE,
            GRID_SIZE
        )
    )

    # Normalize
    ce_image = normalize_image(
        ce_image
    )


    # ========================================================
    # [11] CE METRICS
    # ========================================================

    print()
    print(
        "[11] Calculating CE + CNN metrics..."
    )

    print(
        "-" * 50
    )

    ce_metrics = calculate_all_metrics(
        ground_truth_image,
        ce_image
    )

    print_metrics(
        "CE + CNN",
        ce_metrics
    )


    # ========================================================
    # [12] COMPARE RESULTS
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "                FINAL COMPARISON"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"{'Metric':<10}"
        f"{'Tikhonov':>15}"
        f"{'CE + CNN':>15}"
    )

    print(
        "-" * 40
    )

    for key in initial_metrics:

        print(
            f"{key:<10}"
            f"{initial_metrics[key]:>15.6f}"
            f"{ce_metrics[key]:>15.6f}"
        )


    # ========================================================
    # [13] CREATE RESULTS DIRECTORY
    # ========================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )


    # ========================================================
    # [14] SAVE GROUND TRUTH
    # ========================================================

    ground_truth_path = os.path.join(
        RESULTS_DIR,
        "ce_ground_truth.png"
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        ground_truth_image,
        origin="lower",
        extent=[
            0,
            AREA_WIDTH,
            0,
            AREA_HEIGHT
        ],
        vmin=0,
        vmax=1,
        cmap="viridis"
    )

    plt.colorbar(
        label="Attenuation"
    )

    plt.xlabel(
        "X position (m)"
    )

    plt.ylabel(
        "Y position (m)"
    )

    plt.title(
        "Ground Truth - Smooth Targets"
    )

    plt.tight_layout()

    plt.savefig(
        ground_truth_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # [15] COMPARISON FIGURE
    # ========================================================

    print()
    print(
        "[12] Creating reconstruction comparison..."
    )

    comparison_path = os.path.join(
        RESULTS_DIR,
        "ce_cnn_comparison.png"
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    image0 = axes[0].imshow(
        ground_truth_image,
        origin="lower",
        extent=[
            0,
            AREA_WIDTH,
            0,
            AREA_HEIGHT
        ],
        vmin=0,
        vmax=1,
        cmap="viridis"
    )

    axes[0].set_title(
        "Ground Truth"
    )

    axes[0].set_xlabel(
        "X position (m)"
    )

    axes[0].set_ylabel(
        "Y position (m)"
    )

    # --------------------------------------------------------
    # Tikhonov
    # --------------------------------------------------------

    image1 = axes[1].imshow(
        initial_image,
        origin="lower",
        extent=[
            0,
            AREA_WIDTH,
            0,
            AREA_HEIGHT
        ],
        vmin=0,
        vmax=1,
        cmap="viridis"
    )

    axes[1].set_title(
        "Initial Tikhonov"
    )

    axes[1].set_xlabel(
        "X position (m)"
    )

    axes[1].set_ylabel(
        "Y position (m)"
    )

    # --------------------------------------------------------
    # CE + CNN
    # --------------------------------------------------------

    image2 = axes[2].imshow(
        ce_image,
        origin="lower",
        extent=[
            0,
            AREA_WIDTH,
            0,
            AREA_HEIGHT
        ],
        vmin=0,
        vmax=1,
        cmap="viridis"
    )

    axes[2].set_title(
        "CE + CNN"
    )

    axes[2].set_xlabel(
        "X position (m)"
    )

    axes[2].set_ylabel(
        "Y position (m)"
    )

    # --------------------------------------------------------
    # One shared colorbar
    # --------------------------------------------------------

    fig.colorbar(
        image2,
        ax=axes,
        shrink=0.85,
        label="Normalized attenuation"
    )

    fig.suptitle(
        "RTI Reconstruction Comparison\n"
        f"{NUM_SENSORS} Sensors | "
        f"{NUM_TARGETS} Smooth Targets",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        comparison_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # [16] CE CONVERGENCE PLOT
    # ========================================================

    print()
    print(
        "[13] Creating CE convergence plot..."
    )

    convergence_path = os.path.join(
        RESULTS_DIR,
        "ce_convergence.png"
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        range(
            1,
            len(history) + 1
        ),
        history,
        marker="o"
    )

    plt.xlabel(
        "CE Iteration"
    )

    plt.ylabel(
        "Change"
    )

    plt.title(
        "Consensus Equilibrium Convergence"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        convergence_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # [17] SAVE METRICS TO CSV
    # ========================================================

    metrics_path = os.path.join(
        RESULTS_DIR,
        "ce_metrics.csv"
    )

    with open(
        metrics_path,
        "w",
        newline=""
    ) as file:

        file.write(
            "Method,RMSE,PSNR,SSIM,FSIM\n"
        )

        file.write(
            "Tikhonov,"
            f"{initial_metrics.get('RMSE', np.nan)},"
            f"{initial_metrics.get('PSNR', np.nan)},"
            f"{initial_metrics.get('SSIM', np.nan)},"
            f"{initial_metrics.get('FSIM', np.nan)}\n"
        )

        file.write(
            "CE_CNN,"
            f"{ce_metrics.get('RMSE', np.nan)},"
            f"{ce_metrics.get('PSNR', np.nan)},"
            f"{ce_metrics.get('SSIM', np.nan)},"
            f"{ce_metrics.get('FSIM', np.nan)}\n"
        )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "             CE EXPERIMENT COMPLETED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Saved files:"
    )

    print(
        "1.",
        comparison_path
    )

    print(
        "2.",
        convergence_path
    )

    print(
        "3.",
        ground_truth_path
    )

    print(
        "4.",
        metrics_path
    )

    print()

    print(
        "Final metrics:"
    )

    print_metrics(
        "TIKHONOV",
        initial_metrics
    )

    print_metrics(
        "CE + CNN",
        ce_metrics
    )

    print()
    print(
        "=" * 70
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()