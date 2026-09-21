import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from rti_model import (
    generate_uniform_sensors,
    generate_links,
    generate_grid,
    build_weight_matrix
)

from metrics import calculate_all_metrics


# =========================================================
# CONFIGURATION
# =========================================================

NUM_SENSORS = 24
GRID_SIZE = 30

AREA_WIDTH = 7.0
AREA_HEIGHT = 7.0

NOISE_LEVEL = 0.05

TIKHONOV_ALPHA = 0.1

LAMBDA_SPARSE = 0.01
LAMBDA_FUSED = 0.01

ADMM_RHO = 1.0
ADMM_MAX_ITERATIONS = 100
ADMM_TOLERANCE = 1e-4

RANDOM_SEED = 42

# Ground-truth square
TARGET_CENTER_X = 3.5
TARGET_CENTER_Y = 3.5
TARGET_SIZE = 1.5
TARGET_AMPLITUDE = 1.0


# =========================================================
# TIKHONOV
# =========================================================

def tikhonov_reconstruction(W, y, alpha=0.1):

    N = W.shape[1]

    A = (
        W.T @ W
        + alpha * np.eye(N)
    )

    b = W.T @ y

    x = np.linalg.solve(A, b)

    return x


# =========================================================
# SOFT THRESHOLD
# =========================================================

def soft_threshold(x, threshold):

    return (
        np.sign(x)
        * np.maximum(
            np.abs(x) - threshold,
            0.0
        )
    )


# =========================================================
# DIFFERENCE MATRIX
# =========================================================

def create_difference_matrix(grid_size):

    N = grid_size * grid_size

    rows = []
    cols = []
    data = []

    row_number = 0

    # Horizontal differences
    for r in range(grid_size):

        for c in range(grid_size - 1):

            i = r * grid_size + c
            j = i + 1

            rows.extend([
                row_number,
                row_number
            ])

            cols.extend([
                i,
                j
            ])

            data.extend([
                -1.0,
                1.0
            ])

            row_number += 1

    # Vertical differences
    for r in range(grid_size - 1):

        for c in range(grid_size):

            i = r * grid_size + c
            j = i + grid_size

            rows.extend([
                row_number,
                row_number
            ])

            cols.extend([
                i,
                j
            ])

            data.extend([
                -1.0,
                1.0
            ])

            row_number += 1

    D = csc_matrix(
        (
            data,
            (rows, cols)
        ),
        shape=(
            row_number,
            N
        )
    )

    return D


# =========================================================
# FUSED-L1
# =========================================================

def fused_l1_reconstruction(
    W,
    y,
    grid_size,
    lambda_sparse=0.01,
    lambda_fused=0.01,
    rho=1.0,
    max_iterations=100,
    tolerance=1e-4
):

    N = W.shape[1]

    D = create_difference_matrix(
        grid_size
    )

    number_of_differences = D.shape[0]

    # ADMM variables
    x = np.zeros(N)
    z = np.zeros(N)

    v = np.zeros(
        number_of_differences
    )

    u = np.zeros(N)

    w = np.zeros(
        number_of_differences
    )

    # System matrix
    WTW = W.T @ W

    DTD = (
        D.T @ D
    ).toarray()

    A = (
        WTW
        + rho * np.eye(N)
        + rho * DTD
    )

    factorized_A = splu(
        csc_matrix(A)
    )

    WTy = W.T @ y

    # ADMM iterations
    for iteration in range(
        max_iterations
    ):

        # x update
        rhs = (
            WTy
            + rho * (z - u)
            + rho * D.T @ (v - w)
        )

        x_new = factorized_A.solve(
            rhs
        )

        # Sparse L1 update
        z_new = soft_threshold(
            x_new + u,
            lambda_sparse / rho
        )

        # Fused update
        Dx = D @ x_new

        v_new = soft_threshold(
            Dx + w,
            lambda_fused / rho
        )

        # Dual updates
        u_new = (
            u
            + x_new
            - z_new
        )

        w_new = (
            w
            + Dx
            - v_new
        )

        # Convergence
        residual_1 = np.linalg.norm(
            x_new - z_new
        )

        residual_2 = np.linalg.norm(
            Dx - v_new
        )

        primal_residual = max(
            residual_1,
            residual_2
        )

        x = x_new
        z = z_new
        v = v_new
        u = u_new
        w = w_new

        if primal_residual < tolerance:

            print(
                f"    ADMM converged at "
                f"iteration {iteration + 1}"
            )

            break

    return x


# =========================================================
# NORMALIZE IMAGE
# =========================================================

def normalize_image(image):

    image = np.asarray(
        image,
        dtype=np.float64
    )

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum - minimum < 1e-12:

        return np.zeros_like(
            image
        )

    return (
        image - minimum
    ) / (
        maximum - minimum
    )


# =========================================================
# CREATE SOLID SQUARE TARGET
# =========================================================

def create_single_solid_target(
    X,
    Y,
    center_x,
    center_y,
    size,
    amplitude=1.0
):

    """
    Create a single solid square target.

    This is intentionally implemented here so the
    reconstruction experiment does not depend on the
    current target-wrapper function.
    """

    target = np.zeros_like(
        X,
        dtype=np.float64
    )

    half_size = size / 2.0

    mask = (
        (X >= center_x - half_size)
        & (X <= center_x + half_size)
        & (Y >= center_y - half_size)
        & (Y <= center_y + half_size)
    )

    target[mask] = amplitude

    return target


# =========================================================
# PRINT METRICS
# =========================================================

def print_metrics(
    method_name,
    metrics
):

    print()
    print(method_name)
    print("-" * 40)

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"PSNR : {metrics['PSNR']:.4f} dB"
    )

    print(
        f"SSIM : {metrics['SSIM']:.4f}"
    )

    print(
        f"FSIM : {metrics['FSIM']:.4f}"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("              RTI CLASSICAL RECONSTRUCTION")
    print("=" * 70)

    # -----------------------------------------------------
    # Random seed
    # -----------------------------------------------------

    np.random.seed(
        RANDOM_SEED
    )

    print(
        f"\nRandom seed : {RANDOM_SEED}"
    )

    # -----------------------------------------------------
    # 1. SENSOR NETWORK
    # -----------------------------------------------------

    print(
        "\n[1] Generating sensor network..."
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
        f"Sensors : {NUM_SENSORS}"
    )

    print(
        f"Links   : {len(links)}"
    )

    # -----------------------------------------------------
    # 2. GRID
    # -----------------------------------------------------

    print(
        "\n[2] Generating reconstruction grid..."
    )

    X, Y, grid_points = generate_grid(
        width=AREA_WIDTH,
        height=AREA_HEIGHT,
        grid_size=GRID_SIZE
    )

    print(
        f"Grid    : {GRID_SIZE} x {GRID_SIZE}"
    )

    print(
        f"Voxels  : {GRID_SIZE * GRID_SIZE}"
    )

    # -----------------------------------------------------
    # 3. WEIGHT MATRIX
    # -----------------------------------------------------

    print(
        "\n[3] Building RTI weight matrix..."
    )

    W = build_weight_matrix(
        sensors,
        links,
        grid_points
    )

    print(
        f"W shape : {W.shape}"
    )

    # -----------------------------------------------------
    # 4. GROUND TRUTH
    # -----------------------------------------------------

    print(
        "\n[4] Creating ground-truth target..."
    )

    ground_truth = create_single_solid_target(
        X,
        Y,
        center_x=TARGET_CENTER_X,
        center_y=TARGET_CENTER_Y,
        size=TARGET_SIZE,
        amplitude=TARGET_AMPLITUDE
    )

    ground_truth = np.clip(
        ground_truth,
        0.0,
        1.0
    )

    print(
        f"Target center : "
        f"({TARGET_CENTER_X}, {TARGET_CENTER_Y})"
    )

    print(
        f"Target size   : "
        f"{TARGET_SIZE} m"
    )

    print(
        f"Target max    : "
        f"{np.max(ground_truth):.1f}"
    )

    # -----------------------------------------------------
    # 5. FORWARD MEASUREMENT
    # -----------------------------------------------------

    print(
        "\n[5] Generating measurements..."
    )

    ground_truth_vector = (
        ground_truth.reshape(-1)
    )

    clean_measurement = (
        W @ ground_truth_vector
    )

    noise_std = (
        NOISE_LEVEL
        * np.std(clean_measurement)
    )

    noise = np.random.normal(
        0.0,
        noise_std,
        size=clean_measurement.shape
    )

    measurement = (
        clean_measurement
        + noise
    )

    print(
        f"Noise level : "
        f"{NOISE_LEVEL}"
    )

    print(
        f"Noise std   : "
        f"{noise_std:.6f}"
    )

    # -----------------------------------------------------
    # 6. TIKHONOV
    # -----------------------------------------------------

    print(
        "\n[6] Running Tikhonov reconstruction..."
    )

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

    print(
        "    Tikhonov completed."
    )

    # -----------------------------------------------------
    # 7. FUSED-L1
    # -----------------------------------------------------

    print(
        "\n[7] Running Fused-L1 reconstruction..."
    )

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

    print(
        "    Fused-L1 completed."
    )

    # -----------------------------------------------------
    # 8. METRICS
    # -----------------------------------------------------

    print(
        "\n[8] Calculating metrics..."
    )

    tikhonov_metrics = (
        calculate_all_metrics(
            ground_truth,
            tikhonov_image
        )
    )

    fused_metrics = (
        calculate_all_metrics(
            ground_truth,
            fused_image
        )
    )

    # -----------------------------------------------------
    # 9. RESULTS
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("                         RESULTS")
    print("=" * 70)

    print_metrics(
        "TIKHONOV",
        tikhonov_metrics
    )

    print_metrics(
        "FUSED-L1",
        fused_metrics
    )

    print()
    print("=" * 70)

    # -----------------------------------------------------
    # 10. VISUALIZATION
    # -----------------------------------------------------

    print(
        "\n[9] Creating visual comparison..."
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # Ground truth
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

    # Tikhonov
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

    # Fused-L1
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
        f"RTI Reconstruction - {NUM_SENSORS} Sensors",
        fontsize=14,
        y=0.98
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.93]
    )

    # Save
    plt.savefig(
        "results/classical_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\nSaved:"
    )

    print(
        "results/classical_comparison.png"
    )

    plt.show()

    print(
        "\nExperiment completed successfully."
    )