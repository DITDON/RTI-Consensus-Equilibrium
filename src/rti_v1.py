import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from skimage.metrics import structural_similarity


# ============================================================
# 1. RTI PARAMETERS
# ============================================================

AREA_SIZE = 7.0          # 7m × 7m
GRID_SIZE = 30           # 30 × 30 = 900 voxels
NUM_SENSORS = 24

ELLIPSE_WIDTH = 0.5      # lambda
NOISE_STD = 0.05

ALPHA = 0.1              # Tikhonov regularization parameter


# ============================================================
# 2. CREATE SENSOR LOCATIONS
# ============================================================

def create_sensors():
    """
    Create 24 sensors around the boundary of a 7m × 7m area.
    """

    sensors = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0],
        [6.0, 0.0],
        [7.0, 0.0],

        [7.0, 1.0],
        [7.0, 2.0],
        [7.0, 3.0],
        [7.0, 4.0],
        [7.0, 5.0],
        [7.0, 6.0],

        [7.0, 7.0],
        [6.0, 7.0],
        [5.0, 7.0],
        [4.0, 7.0],
        [3.0, 7.0],
        [2.0, 7.0],
        [1.0, 7.0],
        [0.0, 7.0],

        [0.0, 6.0],
        [0.0, 3.0]
    ])

    return sensors


# ============================================================
# 3. CREATE IMAGE GRID
# ============================================================

def create_grid():
    """
    Create 30 × 30 pixel centers over the 7m × 7m area.
    """

    x = np.linspace(
        AREA_SIZE / (2 * GRID_SIZE),
        AREA_SIZE - AREA_SIZE / (2 * GRID_SIZE),
        GRID_SIZE
    )

    y = np.linspace(
        AREA_SIZE / (2 * GRID_SIZE),
        AREA_SIZE - AREA_SIZE / (2 * GRID_SIZE),
        GRID_SIZE
    )

    xx, yy = np.meshgrid(x, y)

    pixels = np.column_stack(
        (xx.ravel(), yy.ravel())
    )

    return pixels, xx, yy


# ============================================================
# 4. CREATE SENSOR LINKS
# ============================================================

def create_links(sensors):
    """
    Generate every unique sensor-to-sensor link.
    """

    links = list(combinations(range(NUM_SENSORS), 2))

    print("Number of sensors:", len(sensors))
    print("Number of links:", len(links))

    return links


# ============================================================
# 5. BUILD RTI WEIGHT MATRIX W
# ============================================================

def build_weight_matrix(sensors, pixels, links):
    """
    Construct the normalized elliptical RTI weight matrix.

    W has:
        rows    = number of links
        columns = number of pixels
    """

    num_links = len(links)
    num_pixels = len(pixels)

    W = np.zeros((num_links, num_pixels))

    for link_index, (sensor_a, sensor_b) in enumerate(links):

        transmitter = sensors[sensor_a]
        receiver = sensors[sensor_b]

        # Distance between sensors
        d = np.linalg.norm(transmitter - receiver)

        # Distance from every pixel to transmitter
        d1 = np.linalg.norm(
            pixels - transmitter,
            axis=1
        )

        # Distance from every pixel to receiver
        d2 = np.linalg.norm(
            pixels - receiver,
            axis=1
        )

        # Ellipse condition
        inside = (d1 + d2) < (d + ELLIPSE_WIDTH)

        # Normalized elliptical weight
        W[link_index, inside] = 1 / np.sqrt(d)

    return W


# ============================================================
# 6. CREATE GROUND TRUTH OBJECT
# ============================================================

def create_ground_truth(pixels):
    """
    Create a simple square object inside the RTI area.
    """

    x_min = 2.8
    x_max = 4.2

    y_min = 2.8
    y_max = 4.2

    object_mask = (
        (pixels[:, 0] >= x_min) &
        (pixels[:, 0] <= x_max) &
        (pixels[:, 1] >= y_min) &
        (pixels[:, 1] <= y_max)
    )

    x_true = np.zeros(len(pixels))

    x_true[object_mask] = 1.0

    return x_true


# ============================================================
# 7. SIMULATE RSS ATTENUATION
# ============================================================

def simulate_measurements(W, x_true):
    """
    Generate noisy RTI measurements:

        y = W x + n
    """

    clean_measurement = W @ x_true

    noise = np.random.normal(
        0,
        NOISE_STD,
        size=clean_measurement.shape
    )

    noisy_measurement = clean_measurement + noise

    return clean_measurement, noisy_measurement


# ============================================================
# 8. TIKHONOV RECONSTRUCTION
# ============================================================

def tikhonov_reconstruction(W, y):
    """
    Tikhonov reconstruction:

    x = (WᵀW + αI)^(-1) Wᵀy
    """

    num_pixels = W.shape[1]

    identity = np.eye(num_pixels)

    A = W.T @ W + ALPHA * identity

    b = W.T @ y

    x_reconstructed = np.linalg.solve(A, b)

    return x_reconstructed

# ============================================================
# 10. EVALUATION METRICS
# ============================================================

def calculate_metrics(x_true, x_reconstructed):

    # -------------------------
    # RMSE
    # -------------------------

    rmse = np.sqrt(
        np.mean(
            (x_true - x_reconstructed) ** 2
        )
    )

    # -------------------------
    # PSNR
    # -------------------------

    data_range = (
        np.max(x_true) -
        np.min(x_true)
    )

    mse = np.mean(
        (x_true - x_reconstructed) ** 2
    )

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10(
            (data_range ** 2) / mse
        )

    # -------------------------
    # SSIM
    # -------------------------

    true_image = x_true.reshape(
        GRID_SIZE,
        GRID_SIZE
    )

    reconstructed_image = x_reconstructed.reshape(
        GRID_SIZE,
        GRID_SIZE
    )

    ssim = structural_similarity(
        true_image,
        reconstructed_image,
        data_range=data_range
    )

    return rmse, psnr, ssim

# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print("\n==============================")
    print(" RTI SIMULATOR V1")
    print("==============================\n")

    # Sensors
    sensors = create_sensors()

    # Image grid
    pixels, xx, yy = create_grid()

    # Links
    links = create_links(sensors)

    # Weight matrix
    print("\nBuilding RTI weight matrix...")

    W = build_weight_matrix(
        sensors,
        pixels,
        links
    )

    print("W shape:", W.shape)

    # Ground truth
    x_true = create_ground_truth(pixels)

    # Measurements
    clean_y, noisy_y = simulate_measurements(
        W,
        x_true
    )

    print("\nMeasurement vector:")
    print("Clean y shape:", clean_y.shape)
    print("Noisy y shape:", noisy_y.shape)

    # Reconstruction
    print("\nPerforming Tikhonov reconstruction...")

    x_reconstructed = tikhonov_reconstruction(
        W,
        noisy_y
    )
    
    # ========================================================
    # CALCULATE PERFORMANCE
    # ========================================================

    rmse, psnr, ssim = calculate_metrics(
        x_true,
        x_reconstructed
    )

    print("\n==============================")
    print(" RECONSTRUCTION PERFORMANCE")
    print("==============================")

    print(f"RMSE : {rmse:.4f}")
    print(f"PSNR : {psnr:.2f} dB")
    print(f"SSIM : {ssim:.4f}")

    # Convert vectors back to images
    ground_truth_image = x_true.reshape(
        GRID_SIZE,
        GRID_SIZE
    )

    reconstructed_image = x_reconstructed.reshape(
        GRID_SIZE,
        GRID_SIZE
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    plt.figure(figsize=(7, 7))

    plt.scatter(
        sensors[:, 0],
        sensors[:, 1],
        s=80,
        label="Sensors"
    )

    for i, sensor in enumerate(sensors):
        plt.text(
            sensor[0] + 0.08,
            sensor[1] + 0.08,
            str(i + 1),
            fontsize=8
        )

    plt.xlim(-0.5, AREA_SIZE + 0.5)
    plt.ylim(-0.5, AREA_SIZE + 0.5)

    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("RTI Sensor Network")

    plt.grid(True)
    plt.legend()

    plt.show()


    # ========================================================
    # GROUND TRUTH
    # ========================================================

    plt.figure(figsize=(6, 5))

    plt.imshow(
        ground_truth_image,
        origin="lower",
        extent=[0, AREA_SIZE, 0, AREA_SIZE]
    )

    plt.colorbar(label="Attenuation")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("Ground Truth Spatial Loss Field")

    plt.show()


    # ========================================================
    # RECONSTRUCTION
    # ========================================================

    plt.figure(figsize=(6, 5))

    plt.imshow(
        reconstructed_image,
        origin="lower",
        extent=[0, AREA_SIZE, 0, AREA_SIZE]
    )

    plt.colorbar(label="Reconstructed attenuation")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("Tikhonov RTI Reconstruction")

    plt.show()


if __name__ == "__main__":
    main()