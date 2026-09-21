import numpy as np

from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity
)

# Proper FSIM implementation
from image_similarity_measures.quality_metrics import fsim


# =========================================================
# RMSE
# =========================================================

def calculate_rmse(ground_truth, reconstruction):
    """
    Calculate Root Mean Square Error (RMSE).

    Lower RMSE = better reconstruction.
    """

    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)

    mse = mean_squared_error(
        ground_truth,
        reconstruction
    )

    rmse = np.sqrt(mse)

    return rmse


# =========================================================
# PSNR
# =========================================================

def calculate_psnr(ground_truth, reconstruction):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Higher PSNR = better reconstruction.

    Since our RTI images are normalized between 0 and 1,
    data_range is fixed at 1.0.
    """

    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)

    psnr = peak_signal_noise_ratio(
        ground_truth,
        reconstruction,
        data_range=1.0
    )

    return psnr


# =========================================================
# SSIM
# =========================================================

def calculate_ssim(ground_truth, reconstruction):
    """
    Calculate Structural Similarity Index (SSIM).

    Higher SSIM = better reconstruction.

    RTI images are grayscale and normalized between 0 and 1.
    """

    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)

    ssim = structural_similarity(
        ground_truth,
        reconstruction,
        data_range=1.0
    )

    return ssim


# =========================================================
# FSIM
# =========================================================

def calculate_fsim(ground_truth, reconstruction):
    """
    Calculate Feature Similarity Index (FSIM).

    FSIM evaluates similarity using low-level image
    features, mainly:

        1. Phase Congruency
        2. Gradient Magnitude

    Higher FSIM = better reconstruction.

    The image-similarity-measures implementation expects
    image values in an image-like dynamic range, so our
    normalized RTI images [0, 1] are converted to [0, 255].
    """

    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float64
    )

    reconstruction = np.asarray(
        reconstruction,
        dtype=np.float64
    )

    # Make sure both images have identical shape
    if ground_truth.shape != reconstruction.shape:
        raise ValueError(
            "Ground truth and reconstruction must have "
            "the same shape."
        )

    # -----------------------------------------------------
    # Clip RTI images to valid normalized range
    # -----------------------------------------------------

    ground_truth = np.clip(
        ground_truth,
        0.0,
        1.0
    )

    reconstruction = np.clip(
        reconstruction,
        0.0,
        1.0
    )

    # -----------------------------------------------------
    # Convert [0, 1] → [0, 255]
    # -----------------------------------------------------

    ground_truth_255 = (
        ground_truth * 255.0
    ).astype(np.uint8)

    reconstruction_255 = (
        reconstruction * 255.0
    ).astype(np.uint8)

    # -----------------------------------------------------
    # FSIM implementation expects channel-last images.
    #
    # For our grayscale RTI images:
    #
    # (30, 30)
    #
    # becomes
    #
    # (30, 30, 1)
    # -----------------------------------------------------

    if ground_truth_255.ndim == 2:

        ground_truth_255 = np.expand_dims(
            ground_truth_255,
            axis=-1
        )

        reconstruction_255 = np.expand_dims(
            reconstruction_255,
            axis=-1
        )

    # -----------------------------------------------------
    # Calculate FSIM
    # -----------------------------------------------------

    fsim_value = fsim(
        ground_truth_255,
        reconstruction_255
    )

    return float(fsim_value)


# =========================================================
# ALL METRICS
# =========================================================

def calculate_all_metrics(
    ground_truth,
    reconstruction
):
    """
    Calculate all four RTI reconstruction metrics.

    Metrics:

        RMSE
        PSNR
        SSIM
        FSIM

    Returns:
        Dictionary containing all metrics.
    """

    rmse = calculate_rmse(
        ground_truth,
        reconstruction
    )

    psnr = calculate_psnr(
        ground_truth,
        reconstruction
    )

    ssim = calculate_ssim(
        ground_truth,
        reconstruction
    )

    fsim_value = calculate_fsim(
        ground_truth,
        reconstruction
    )

    return {
        "RMSE": rmse,
        "PSNR": psnr,
        "SSIM": ssim,
        "FSIM": fsim_value
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("\n========================================")
    print("       RTI METRICS TEST")
    print("========================================")

    # -----------------------------------------------------
    # Create simple ground-truth RTI image
    # -----------------------------------------------------

    ground_truth = np.zeros(
        (30, 30),
        dtype=np.float64
    )

    ground_truth[10:20, 10:20] = 1.0

    # -----------------------------------------------------
    # Create slightly degraded reconstruction
    # -----------------------------------------------------

    reconstruction = ground_truth.copy()

    reconstruction[11:19, 11:19] = 0.8

    # -----------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------

    metrics = calculate_all_metrics(
        ground_truth,
        reconstruction
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nGround Truth Shape:")
    print(ground_truth.shape)

    print("\nReconstruction Shape:")
    print(reconstruction.shape)

    print("\n----------------------------------------")
    print("IMAGE QUALITY METRICS")
    print("----------------------------------------")

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

    print("----------------------------------------")
    print("Metric calculation completed.")
    print("========================================\n")