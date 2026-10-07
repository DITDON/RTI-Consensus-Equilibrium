import os
import sys
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(0, CURRENT_DIR)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from rti_model import (
    generate_uniform_sensors,
    generate_links,
    generate_grid,
    build_weight_matrix
)

from targets import (
    create_solid_targets,
    create_smooth_targets
)

from reconstruction import (
    tikhonov_reconstruction
)

from denoiser import (
    RTIDenoiser
)

from ce_solver import (
    consensus_equilibrium
)

from metrics import (
    calculate_all_metrics
)


# ============================================================
# CONFIGURATION
# ============================================================

AREA_WIDTH = 7.0
AREA_HEIGHT = 7.0

GRID_SIZE = 30

SENSOR_COUNTS = [
    16,
    24,
    32
]

TARGET_TYPES = [
    "solid",
    "smooth"
]

TARGET_COUNTS = [
    1,
    2,
    3,
    5
]

# Solid square side length in meters
SOLID_TARGET_SIZE = 1.0

# Gaussian sigma in meters
SMOOTH_SIGMA = 0.4

# Maximum target amplitude
TARGET_AMPLITUDE = 1.0

# Measurement noise
NOISE_STD = 0.02

# Tikhonov regularization
TIKHONOV_ALPHA = 0.1

# CE parameters
CE_ITERATIONS = 20
CE_DATA_STEP = 0.1
CE_RELAXATION = 0.5

# Reproducibility
RANDOM_SEED = 42

# Trained CNN
MODEL_PATH = os.path.join(
    "results",
    "cnn_denoiser.pth"
)

# Output files
RESULTS_DIR = "results"

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "ce_benchmark_results.csv"
)

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "ce_benchmark_summary.csv"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TARGET CENTERS
# ============================================================
#
# These positions are inside the 7m x 7m RTI area.
#
# For 1 target -> first 1
# For 2 targets -> first 2
# For 3 targets -> first 3
# For 5 targets -> all 5
#
# This keeps target positions deterministic across
# all sensor configurations.
# ============================================================

TARGET_CENTERS = [
    (1.75, 1.75),
    (5.25, 1.75),
    (1.75, 5.25),
    (5.25, 5.25),
    (3.50, 3.50)
]


# ============================================================
# NORMALIZE IMAGE
# ============================================================

def normalize_image(image):
    """
    Normalize an RTI image to [0, 1].
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.clip(
        image,
        0.0,
        None
    )

    maximum = np.max(image)

    if maximum > 1e-8:
        image = image / maximum

    return np.clip(
        image,
        0.0,
        1.0
    )


# ============================================================
# CREATE GROUND TRUTH
# ============================================================

def create_ground_truth(
    grid_points,
    target_type,
    num_targets
):
    """
    Create the requested RTI ground-truth image.

    target_type:
        "solid"
        "smooth"
    """

    centers = TARGET_CENTERS[
        :num_targets
    ]

    if target_type == "solid":

        ground_truth = create_solid_targets(
            grid_points,
            centers,
            target_size=SOLID_TARGET_SIZE,
            amplitude=TARGET_AMPLITUDE
        )

    elif target_type == "smooth":

        ground_truth = create_smooth_targets(
            grid_points,
            centers,
            sigma=SMOOTH_SIGMA,
            amplitude=TARGET_AMPLITUDE
        )

    else:

        raise ValueError(
            f"Unknown target type: {target_type}"
        )

    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float32
    )

    ground_truth = ground_truth.reshape(
        GRID_SIZE,
        GRID_SIZE
    )

    ground_truth = normalize_image(
        ground_truth
    )

    return ground_truth


# ============================================================
# GENERATE MEASUREMENTS
# ============================================================

def generate_measurements(
    W,
    ground_truth,
    rng
):
    """
    Generate synthetic RSS/RTI measurements.

    y = W x + noise
    """

    ground_truth_vector = (
        ground_truth.reshape(-1)
    )

    clean_measurements = (
        W @ ground_truth_vector
    )

    noise = rng.normal(
        loc=0.0,
        scale=NOISE_STD,
        size=clean_measurements.shape
    )

    measurements = (
        clean_measurements
        + noise
    )

    return measurements.astype(
        np.float32
    )


# ============================================================
# TIKHONOV RECONSTRUCTION
# ============================================================

def run_tikhonov(
    W,
    measurements
):
    """
    Run classical Tikhonov reconstruction.
    """

    reconstruction = tikhonov_reconstruction(
        W,
        measurements,
        alpha=TIKHONOV_ALPHA
    )

    reconstruction = np.asarray(
        reconstruction,
        dtype=np.float32
    )

    reconstruction = reconstruction.reshape(
        GRID_SIZE,
        GRID_SIZE
    )

    reconstruction = normalize_image(
        reconstruction
    )

    return reconstruction


# ============================================================
# LOAD CNN
# ============================================================

def load_cnn():
    """
    Load the trained RTI CNN denoiser.
    """

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "\nCNN model not found.\n"
            f"Expected file:\n{MODEL_PATH}\n\n"
            "Run this first:\n"
            "python src/train_denoiser.py"
        )

    model = RTIDenoiser().to(
        DEVICE
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# RUN CE + CNN
# ============================================================

def run_ce(
    initial_reconstruction,
    W,
    measurements,
    model
):
    """
    Run Consensus Equilibrium using:

        Agent 1 -> RTI data fidelity
        Agent 2 -> CNN denoiser
    """

    x0 = (
        initial_reconstruction
        .reshape(-1)
        .astype(np.float32)
    )

    ce_reconstruction, history = (
        consensus_equilibrium(
            x0=x0,
            W=W,
            y=measurements,
            denoiser=model,
            device=DEVICE,
            iterations=CE_ITERATIONS,
            data_step=CE_DATA_STEP,
            relaxation=CE_RELAXATION
        )
    )

    ce_reconstruction = np.asarray(
        ce_reconstruction,
        dtype=np.float32
    )

    ce_reconstruction = np.clip(
        ce_reconstruction,
        0.0,
        None
    )

    ce_reconstruction = ce_reconstruction.reshape(
        GRID_SIZE,
        GRID_SIZE
    )

    ce_reconstruction = normalize_image(
        ce_reconstruction
    )

    return ce_reconstruction, history


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    ground_truth,
    reconstruction
):
    """
    Calculate:

        RMSE
        PSNR
        SSIM
        FSIM
    """

    metrics = calculate_all_metrics(
        ground_truth,
        reconstruction
    )

    return {
        "RMSE": float(metrics["RMSE"]),
        "PSNR": float(metrics["PSNR"]),
        "SSIM": float(metrics["SSIM"]),
        "FSIM": float(metrics["FSIM"])
    }


# ============================================================
# SAVE COMPARISON IMAGE
# ============================================================

def save_comparison(
    ground_truth,
    tikhonov,
    ce_reconstruction,
    sensor_count,
    target_type,
    num_targets
):
    """
    Save:

        Ground Truth
        Tikhonov
        CE + CNN
    """

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4)
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    axes[0].imshow(
        ground_truth,
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

    # --------------------------------------------------------
    # Tikhonov
    # --------------------------------------------------------

    axes[1].imshow(
        tikhonov,
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
        "Tikhonov"
    )

    # --------------------------------------------------------
    # CE + CNN
    # --------------------------------------------------------

    axes[2].imshow(
        ce_reconstruction,
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

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    for ax in axes:

        ax.set_xlabel(
            "x (m)"
        )

        ax.set_ylabel(
            "y (m)"
        )

    fig.suptitle(
        f"{target_type.title()} | "
        f"{num_targets} Target(s) | "
        f"{sensor_count} Sensors"
    )

    plt.tight_layout()

    filename = (
        f"ce_benchmark_"
        f"{target_type}_"
        f"{num_targets}targets_"
        f"{sensor_count}sensors.png"
    )

    output_path = os.path.join(
        RESULTS_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# SAVE CONVERGENCE PLOT
# ============================================================

def save_convergence_plot(
    history,
    sensor_count,
    target_type,
    num_targets
):
    """
    Save CE convergence curve.
    """

    if len(history) == 0:
        return None

    plt.figure(
        figsize=(7, 5)
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
        f"CE Convergence | "
        f"{target_type.title()} | "
        f"{num_targets} Target(s) | "
        f"{sensor_count} Sensors"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    filename = (
        f"ce_convergence_"
        f"{target_type}_"
        f"{num_targets}targets_"
        f"{sensor_count}sensors.png"
    )

    output_path = os.path.join(
        RESULTS_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_single_experiment(
    sensor_count,
    target_type,
    num_targets,
    model,
    rng
):
    """
    Run one complete experiment.

    Pipeline:

        Sensors
          ↓
        Links
          ↓
        Weight Matrix
          ↓
        Ground Truth
          ↓
        RSS Measurements
          ↓
        Tikhonov
          ↓
        CNN + CE
          ↓
        Metrics
    """

    print()
    print("=" * 70)

    print(
        f"Sensors      : {sensor_count}"
    )

    print(
        f"Target Type  : {target_type}"
    )

    print(
        f"Target Count : {num_targets}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # SENSOR NETWORK
    # --------------------------------------------------------

    sensors = generate_uniform_sensors(
        sensor_count,
        width=AREA_WIDTH,
        height=AREA_HEIGHT
    )

    links = generate_links(
        sensor_count
    )

    print(
        f"Links        : {len(links)}"
    )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    X, Y, grid_points = generate_grid(
        grid_size=GRID_SIZE,
        width=AREA_WIDTH,
        height=AREA_HEIGHT
    )

    print(
        f"Grid         : {GRID_SIZE} x {GRID_SIZE}"
    )

    # --------------------------------------------------------
    # WEIGHT MATRIX
    # --------------------------------------------------------

    print(
        "Building W..."
    )

    W = build_weight_matrix(
        sensors,
        links,
        grid_points
    )

    print(
        f"W shape      : {W.shape}"
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    ground_truth = create_ground_truth(
        grid_points,
        target_type,
        num_targets
    )

    # --------------------------------------------------------
    # MEASUREMENTS
    # --------------------------------------------------------

    measurements = generate_measurements(
        W,
        ground_truth,
        rng
    )

    print(
        f"Measurements : {measurements.shape}"
    )

    # --------------------------------------------------------
    # TIKHONOV
    # --------------------------------------------------------

    print(
        "Running Tikhonov..."
    )

    tikhonov = run_tikhonov(
        W,
        measurements
    )

    tikhonov_metrics = calculate_metrics(
        ground_truth,
        tikhonov
    )

    print()
    print(
        "TIKHONOV"
    )

    print(
        f"RMSE : {tikhonov_metrics['RMSE']:.4f}"
    )

    print(
        f"PSNR : {tikhonov_metrics['PSNR']:.4f} dB"
    )

    print(
        f"SSIM : {tikhonov_metrics['SSIM']:.4f}"
    )

    print(
        f"FSIM : {tikhonov_metrics['FSIM']:.4f}"
    )

    # --------------------------------------------------------
    # CE + CNN
    # --------------------------------------------------------

    print()
    print(
        "Running CE + CNN..."
    )

    ce_reconstruction, history = run_ce(
        tikhonov,
        W,
        measurements,
        model
    )

    ce_metrics = calculate_metrics(
        ground_truth,
        ce_reconstruction
    )

    print()
    print(
        "CE + CNN"
    )

    print(
        f"RMSE : {ce_metrics['RMSE']:.4f}"
    )

    print(
        f"PSNR : {ce_metrics['PSNR']:.4f} dB"
    )

    print(
        f"SSIM : {ce_metrics['SSIM']:.4f}"
    )

    print(
        f"FSIM : {ce_metrics['FSIM']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    comparison_path = save_comparison(
        ground_truth,
        tikhonov,
        ce_reconstruction,
        sensor_count,
        target_type,
        num_targets
    )

    # --------------------------------------------------------
    # SAVE CONVERGENCE
    # --------------------------------------------------------

    convergence_path = save_convergence_plot(
        history,
        sensor_count,
        target_type,
        num_targets
    )

    print()
    print(
        "Saved comparison:"
    )

    print(
        comparison_path
    )

    if convergence_path is not None:

        print(
            "Saved convergence:"
        )

        print(
            convergence_path
        )

    # --------------------------------------------------------
    # RESULT ROWS
    # --------------------------------------------------------

    tikhonov_row = {
        "Sensors": sensor_count,
        "Links": len(links),
        "Target_Type": target_type,
        "Target_Count": num_targets,
        "Method": "Tikhonov",
        "RMSE": tikhonov_metrics["RMSE"],
        "PSNR": tikhonov_metrics["PSNR"],
        "SSIM": tikhonov_metrics["SSIM"],
        "FSIM": tikhonov_metrics["FSIM"]
    }

    ce_row = {
        "Sensors": sensor_count,
        "Links": len(links),
        "Target_Type": target_type,
        "Target_Count": num_targets,
        "Method": "CE+CNN",
        "RMSE": ce_metrics["RMSE"],
        "PSNR": ce_metrics["PSNR"],
        "SSIM": ce_metrics["SSIM"],
        "FSIM": ce_metrics["FSIM"]
    }

    return [
        tikhonov_row,
        ce_row
    ]


# ============================================================
# SAVE RAW CSV
# ============================================================

def save_results_csv(
    results
):
    """
    Save every experiment result.
    """

    fieldnames = [
        "Sensors",
        "Links",
        "Target_Type",
        "Target_Count",
        "Method",
        "RMSE",
        "PSNR",
        "SSIM",
        "FSIM"
    ]

    with open(
        CSV_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in results:

            writer.writerow(
                row
            )


# ============================================================
# CREATE SUMMARY
# ============================================================

def create_summary(
    results
):
    """
    Create an average summary grouped by:

        Sensors
        Target Type
        Target Count
        Method
    """

    groups = {}

    for row in results:

        key = (
            row["Sensors"],
            row["Target_Type"],
            row["Target_Count"],
            row["Method"]
        )

        if key not in groups:

            groups[key] = {
                "RMSE": [],
                "PSNR": [],
                "SSIM": [],
                "FSIM": []
            }

        groups[key]["RMSE"].append(
            row["RMSE"]
        )

        groups[key]["PSNR"].append(
            row["PSNR"]
        )

        groups[key]["SSIM"].append(
            row["SSIM"]
        )

        groups[key]["FSIM"].append(
            row["FSIM"]
        )

    fieldnames = [
        "Sensors",
        "Target_Type",
        "Target_Count",
        "Method",
        "Mean_RMSE",
        "Mean_PSNR",
        "Mean_SSIM",
        "Mean_FSIM"
    ]

    with open(
        SUMMARY_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for key in sorted(
            groups.keys()
        ):

            sensors = key[0]
            target_type = key[1]
            target_count = key[2]
            method = key[3]

            values = groups[key]

            writer.writerow({
                "Sensors": sensors,
                "Target_Type": target_type,
                "Target_Count": target_count,
                "Method": method,
                "Mean_RMSE": np.mean(
                    values["RMSE"]
                ),
                "Mean_PSNR": np.mean(
                    values["PSNR"]
                ),
                "Mean_SSIM": np.mean(
                    values["SSIM"]
                ),
                "Mean_FSIM": np.mean(
                    values["FSIM"]
                )
            })


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_final_summary(
    results
):

    print()
    print("=" * 90)
    print("CE + CNN BENCHMARK SUMMARY")
    print("=" * 90)

    print(
        f"{'Sensors':<9}"
        f"{'Type':<10}"
        f"{'Targets':<9}"
        f"{'Method':<12}"
        f"{'RMSE':<10}"
        f"{'PSNR':<10}"
        f"{'SSIM':<10}"
        f"{'FSIM':<10}"
    )

    print("-" * 90)

    for row in results:

        print(
            f"{row['Sensors']:<9}"
            f"{row['Target_Type']:<10}"
            f"{row['Target_Count']:<9}"
            f"{row['Method']:<12}"
            f"{row['RMSE']:<10.4f}"
            f"{row['PSNR']:<10.4f}"
            f"{row['SSIM']:<10.4f}"
            f"{row['FSIM']:<10.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("          RTI CONSENSUS EQUILIBRIUM BENCHMARK")
    print("=" * 80)

    print()
    print(
        "Device       :",
        DEVICE
    )

    print(
        "Grid         :",
        f"{GRID_SIZE} x {GRID_SIZE}"
    )

    print(
        "Sensors      :",
        SENSOR_COUNTS
    )

    print(
        "Target types :",
        TARGET_TYPES
    )

    print(
        "Target counts:",
        TARGET_COUNTS
    )

    print(
        "Noise std    :",
        NOISE_STD
    )

    print(
        "CE iterations:",
        CE_ITERATIONS
    )

    print()

    # --------------------------------------------------------
    # CREATE RESULTS DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # RANDOM SEED
    # --------------------------------------------------------

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

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # LOAD CNN
    # --------------------------------------------------------

    print(
        "[1] Loading trained CNN..."
    )

    model = load_cnn()

    print(
        "CNN loaded successfully."
    )

    # --------------------------------------------------------
    # CALCULATE NUMBER OF EXPERIMENTS
    # --------------------------------------------------------

    num_experiments = (
        len(SENSOR_COUNTS)
        * len(TARGET_TYPES)
        * len(TARGET_COUNTS)
    )

    print()
    print(
        "[2] Experiment plan"
    )

    print(
        f"Experimental cases : {num_experiments}"
    )

    print(
        f"Reconstructions    : {num_experiments * 2}"
    )

    print(
        "Methods             : Tikhonov + CE+CNN"
    )

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    results = []

    experiment_number = 0

    for sensor_count in SENSOR_COUNTS:

        for target_type in TARGET_TYPES:

            for num_targets in TARGET_COUNTS:

                experiment_number += 1

                print()
                print()
                print(
                    "#" * 90
                )

                print(
                    f"EXPERIMENT "
                    f"{experiment_number}/{num_experiments}"
                )

                print(
                    "#" * 90
                )

                experiment_results = (
                    run_single_experiment(
                        sensor_count,
                        target_type,
                        num_targets,
                        model,
                        rng
                    )
                )

                results.extend(
                    experiment_results
                )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "[3] Saving benchmark results..."
    )

    save_results_csv(
        results
    )

    create_summary(
        results
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_final_summary(
        results
    )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("             BENCHMARK COMPLETED")
    print("=" * 80)

    print()
    print(
        "Results saved to:"
    )

    print(
        CSV_PATH
    )

    print()
    print(
        "Summary saved to:"
    )

    print(
        SUMMARY_PATH
    )

    print()
    print(
        f"Experimental cases : {num_experiments}"
    )

    print(
        f"Reconstructions    : {len(results)}"
    )

    print()
    print(
        "All CE + CNN benchmark experiments completed."
    )

    print(
        "=" * 80
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()