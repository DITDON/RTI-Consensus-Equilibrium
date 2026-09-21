import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from rti_model import (
    generate_uniform_sensors,
    generate_links,
    generate_grid,
    build_weight_matrix
)

from reconstruction import (
    tikhonov_reconstruction,
    fused_l1_reconstruction,
    normalize_image
)

from metrics import calculate_all_metrics


# =========================================================
# EXPERIMENT CONFIGURATION
# =========================================================

AREA_WIDTH = 7.0
AREA_HEIGHT = 7.0

GRID_SIZE = 30

SENSOR_COUNTS = [16, 24, 32]

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

NOISE_LEVEL = 0.05

RANDOM_SEED = 42

# Tikhonov
TIKHONOV_ALPHA = 0.1

# Fused-L1
LAMBDA_SPARSE = 0.01
LAMBDA_FUSED = 0.01
ADMM_RHO = 1.0
ADMM_MAX_ITERATIONS = 100
ADMM_TOLERANCE = 1e-4


# =========================================================
# TARGET CONFIGURATION
# =========================================================

# Fixed target centers.
# Using fixed locations makes experiments reproducible.

TARGET_CENTERS = [
    (1.75, 1.75),
    (5.25, 1.75),
    (1.75, 5.25),
    (5.25, 5.25),
    (3.50, 3.50)
]

SOLID_TARGET_SIZE = 1.0

SMOOTH_TARGET_SIGMA = 0.45


# =========================================================
# CREATE SOLID TARGETS
# =========================================================

def create_solid_targets(
    X,
    Y,
    number_of_targets
):
    """
    Create multiple solid square targets.
    """

    target = np.zeros_like(
        X,
        dtype=np.float64
    )

    for i in range(number_of_targets):

        center_x, center_y = TARGET_CENTERS[i]

        half_size = (
            SOLID_TARGET_SIZE / 2.0
        )

        mask = (
            (X >= center_x - half_size)
            & (X <= center_x + half_size)
            & (Y >= center_y - half_size)
            & (Y <= center_y + half_size)
        )

        target[mask] = 1.0

    return target


# =========================================================
# CREATE SMOOTH TARGETS
# =========================================================

def create_smooth_targets(
    X,
    Y,
    number_of_targets
):
    """
    Create multiple smooth Gaussian targets.
    """

    target = np.zeros_like(
        X,
        dtype=np.float64
    )

    sigma = SMOOTH_TARGET_SIGMA

    for i in range(number_of_targets):

        center_x, center_y = TARGET_CENTERS[i]

        gaussian = np.exp(
            -(
                (X - center_x) ** 2
                + (Y - center_y) ** 2
            )
            / (
                2.0 * sigma ** 2
            )
        )

        target = np.maximum(
            target,
            gaussian
        )

    return np.clip(
        target,
        0.0,
        1.0
    )


# =========================================================
# CREATE TARGET
# =========================================================

def create_target(
    X,
    Y,
    target_type,
    number_of_targets
):

    if target_type == "solid":

        return create_solid_targets(
            X,
            Y,
            number_of_targets
        )

    elif target_type == "smooth":

        return create_smooth_targets(
            X,
            Y,
            number_of_targets
        )

    else:

        raise ValueError(
            f"Unknown target type: "
            f"{target_type}"
        )


# =========================================================
# GENERATE MEASUREMENTS
# =========================================================

def generate_measurements(
    W,
    ground_truth,
    noise_level,
    rng
):

    ground_truth_vector = (
        ground_truth.reshape(-1)
    )

    # Forward RTI model
    clean_measurement = (
        W @ ground_truth_vector
    )

    # Gaussian noise
    noise_std = (
        noise_level
        * np.std(clean_measurement)
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=clean_measurement.shape
    )

    measurement = (
        clean_measurement
        + noise
    )

    return measurement


# =========================================================
# RUN ONE EXPERIMENT
# =========================================================

def run_single_experiment(
    num_sensors,
    target_type,
    number_of_targets,
    X,
    Y,
    grid_points,
    rng
):

    # -----------------------------------------------------
    # Sensor network
    # -----------------------------------------------------

    sensors = generate_uniform_sensors(
        num_sensors,
        width=AREA_WIDTH,
        height=AREA_HEIGHT
    )

    links = generate_links(
        num_sensors
    )

    # -----------------------------------------------------
    # Weight matrix
    # -----------------------------------------------------

    W = build_weight_matrix(
        sensors,
        links,
        grid_points
    )

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    ground_truth = create_target(
        X,
        Y,
        target_type,
        number_of_targets
    )

    # -----------------------------------------------------
    # Measurements
    # -----------------------------------------------------

    measurement = generate_measurements(
        W,
        ground_truth,
        NOISE_LEVEL,
        rng
    )

    # -----------------------------------------------------
    # Tikhonov
    # -----------------------------------------------------

    tikhonov_vector = (
        tikhonov_reconstruction(
            W,
            measurement,
            alpha=TIKHONOV_ALPHA
        )
    )

    tikhonov_image = (
        tikhonov_vector.reshape(
            GRID_SIZE,
            GRID_SIZE
        )
    )

    tikhonov_image = normalize_image(
        tikhonov_image
    )

    # -----------------------------------------------------
    # Tikhonov metrics
    # -----------------------------------------------------

    tikhonov_metrics = (
        calculate_all_metrics(
            ground_truth,
            tikhonov_image
        )
    )

    # -----------------------------------------------------
    # Fused-L1
    # -----------------------------------------------------

    fused_vector = (
        fused_l1_reconstruction(
            W,
            measurement,
            GRID_SIZE,
            lambda_sparse=LAMBDA_SPARSE,
            lambda_fused=LAMBDA_FUSED,
            rho=ADMM_RHO,
            max_iterations=ADMM_MAX_ITERATIONS,
            tolerance=ADMM_TOLERANCE
        )
    )

    fused_image = (
        fused_vector.reshape(
            GRID_SIZE,
            GRID_SIZE
        )
    )

    fused_image = normalize_image(
        fused_image
    )

    # -----------------------------------------------------
    # Fused-L1 metrics
    # -----------------------------------------------------

    fused_metrics = (
        calculate_all_metrics(
            ground_truth,
            fused_image
        )
    )

    return (
        ground_truth,
        tikhonov_image,
        fused_image,
        tikhonov_metrics,
        fused_metrics,
        len(links)
    )


# =========================================================
# SAVE RESULTS TO CSV
# =========================================================

def save_results(
    results,
    filename
):

    fieldnames = [
        "Sensors",
        "Links",
        "Target_Type",
        "Num_Targets",
        "Method",
        "RMSE",
        "PSNR",
        "SSIM",
        "FSIM"
    ]

    with open(
        filename,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                result
            )


# =========================================================
# SAVE REPRESENTATIVE FIGURE
# =========================================================

def save_representative_figure(
    ground_truth,
    tikhonov_image,
    fused_image,
    sensors,
    num_sensors,
    target_type,
    number_of_targets
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    axes[0].imshow(
        ground_truth,
        cmap="viridis",
        origin="lower",
        extent=[
            0,
            AREA_WIDTH,
            0,
            AREA_HEIGHT
        ],
        vmin=0,
        vmax=1
    )

    axes[0].set_title(
        "Ground Truth"
    )

    axes[0].set_xlabel(
        "x (m)"
    )

    axes[0].set_ylabel(
        "y (m)"
    )

    # -----------------------------------------------------
    # Tikhonov
    # -----------------------------------------------------

    axes[1].imshow(
        tikhonov_image,
        cmap="viridis",
        origin="lower",
        extent=[
            0,
            AREA_WIDTH,
            0,
            AREA_HEIGHT
        ],
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "Tikhonov"
    )

    axes[1].set_xlabel(
        "x (m)"
    )

    axes[1].set_ylabel(
        "y (m)"
    )

    # -----------------------------------------------------
    # Fused-L1
    # -----------------------------------------------------

    axes[2].imshow(
        fused_image,
        cmap="viridis",
        origin="lower",
        extent=[
            0,
            AREA_WIDTH,
            0,
            AREA_HEIGHT
        ],
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "Fused-L1"
    )

    axes[2].set_xlabel(
        "x (m)"
    )

    axes[2].set_ylabel(
        "y (m)"
    )

    plt.suptitle(
        f"{target_type.capitalize()} | "
        f"{number_of_targets} Target(s) | "
        f"{num_sensors} Sensors",
        fontsize=14
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    filename = (
        f"results/"
        f"benchmark_{target_type}_"
        f"{number_of_targets}targets_"
        f"{num_sensors}sensors.png"
    )

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return filename


# =========================================================
# MAIN EXPERIMENT
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 75)
    print("             CLASSICAL RTI EXPERIMENT BENCHMARK")
    print("=" * 75)

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print(
        f"\nRandom seed : {RANDOM_SEED}"
    )

    # -----------------------------------------------------
    # Create results directory
    # -----------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    # -----------------------------------------------------
    # Create grid ONCE
    # -----------------------------------------------------

    print(
        "\nCreating reconstruction grid..."
    )

    X, Y, grid_points = generate_grid(
        width=AREA_WIDTH,
        height=AREA_HEIGHT,
        grid_size=GRID_SIZE
    )

    print(
        f"Grid : {GRID_SIZE} x {GRID_SIZE}"
    )

    print(
        f"Voxels : {GRID_SIZE * GRID_SIZE}"
    )

    # -----------------------------------------------------
    # Results storage
    # -----------------------------------------------------

    results = []

    total_experiments = (
        len(SENSOR_COUNTS)
        * len(TARGET_TYPES)
        * len(TARGET_COUNTS)
    )

    completed = 0

    print()
    print(
        f"Total experimental cases : "
        f"{total_experiments}"
    )

    print(
        f"Reconstruction methods   : 2"
    )

    print(
        f"Total reconstructions    : "
        f"{total_experiments * 2}"
    )

    print()
    print("=" * 75)

    # =====================================================
    # EXPERIMENT LOOP
    # =====================================================

    for num_sensors in SENSOR_COUNTS:

        print()
        print(
            f"\n######## {num_sensors} SENSORS ########"
        )

        for target_type in TARGET_TYPES:

            for number_of_targets in TARGET_COUNTS:

                completed += 1

                print()
                print(
                    "-" * 75
                )

                print(
                    f"Experiment "
                    f"{completed}/{total_experiments}"
                )

                print(
                    f"Sensors       : "
                    f"{num_sensors}"
                )

                print(
                    f"Target type   : "
                    f"{target_type}"
                )

                print(
                    f"Target count  : "
                    f"{number_of_targets}"
                )

                # -------------------------------------------------
                # Run experiment
                # -------------------------------------------------

                (
                    ground_truth,
                    tikhonov_image,
                    fused_image,
                    tikhonov_metrics,
                    fused_metrics,
                    number_of_links
                ) = run_single_experiment(
                    num_sensors,
                    target_type,
                    number_of_targets,
                    X,
                    Y,
                    grid_points,
                    rng
                )

                # -------------------------------------------------
                # Store Tikhonov
                # -------------------------------------------------

                results.append({
                    "Sensors": num_sensors,
                    "Links": number_of_links,
                    "Target_Type": target_type,
                    "Num_Targets": number_of_targets,
                    "Method": "Tikhonov",
                    "RMSE": tikhonov_metrics["RMSE"],
                    "PSNR": tikhonov_metrics["PSNR"],
                    "SSIM": tikhonov_metrics["SSIM"],
                    "FSIM": tikhonov_metrics["FSIM"]
                })

                # -------------------------------------------------
                # Store Fused-L1
                # -------------------------------------------------

                results.append({
                    "Sensors": num_sensors,
                    "Links": number_of_links,
                    "Target_Type": target_type,
                    "Num_Targets": number_of_targets,
                    "Method": "Fused-L1",
                    "RMSE": fused_metrics["RMSE"],
                    "PSNR": fused_metrics["PSNR"],
                    "SSIM": fused_metrics["SSIM"],
                    "FSIM": fused_metrics["FSIM"]
                })

                # -------------------------------------------------
                # Print current metrics
                # -------------------------------------------------

                print()
                print("Tikhonov:")

                print(
                    f"  RMSE : "
                    f"{tikhonov_metrics['RMSE']:.4f}"
                )

                print(
                    f"  PSNR : "
                    f"{tikhonov_metrics['PSNR']:.4f} dB"
                )

                print(
                    f"  SSIM : "
                    f"{tikhonov_metrics['SSIM']:.4f}"
                )

                print(
                    f"  FSIM : "
                    f"{tikhonov_metrics['FSIM']:.4f}"
                )

                print()
                print("Fused-L1:")

                print(
                    f"  RMSE : "
                    f"{fused_metrics['RMSE']:.4f}"
                )

                print(
                    f"  PSNR : "
                    f"{fused_metrics['PSNR']:.4f} dB"
                )

                print(
                    f"  SSIM : "
                    f"{fused_metrics['SSIM']:.4f}"
                )

                print(
                    f"  FSIM : "
                    f"{fused_metrics['FSIM']:.4f}"
                )

                # -------------------------------------------------
                # Save representative figure
                # -------------------------------------------------

                figure_file = (
                    save_representative_figure(
                        ground_truth,
                        tikhonov_image,
                        fused_image,
                        None,
                        num_sensors,
                        target_type,
                        number_of_targets
                    )
                )

                print(
                    f"\nFigure saved: "
                    f"{figure_file}"
                )

    # =====================================================
    # SAVE CSV
    # =====================================================

    csv_filename = (
        "results/experiment_results.csv"
    )

    save_results(
        results,
        csv_filename
    )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print()
    print("=" * 75)
    print("                 BENCHMARK COMPLETE")
    print("=" * 75)

    print()
    print(
        f"Experimental cases : "
        f"{total_experiments}"
    )

    print(
        f"Reconstructions    : "
        f"{len(results)}"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        csv_filename
    )

    print()
    print(
        "Each row contains:"
    )

    print(
        "Sensors | Links | Target Type | "
        "Target Count | Method | "
        "RMSE | PSNR | SSIM | FSIM"
    )

    print()
    print(
        "=" * 75
    )

    print(
        "\nAll classical experiments completed successfully."
    )