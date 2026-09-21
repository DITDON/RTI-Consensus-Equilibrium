import numpy as np


# ============================================================
# TARGET GENERATION
# ============================================================


def create_empty_target(grid_points):
    """
    Create an empty spatial loss field.
    """

    return np.zeros(len(grid_points))


# ============================================================
# SOLID TARGETS
# ============================================================

def add_solid_circle(
    target,
    grid_points,
    center,
    radius,
    amplitude=1.0
):
    """
    Add a solid circular target.

    Pixels inside the circle receive a constant
    attenuation value.
    """

    cx, cy = center

    distance_squared = (
        (grid_points[:, 0] - cx) ** 2
        +
        (grid_points[:, 1] - cy) ** 2
    )

    mask = distance_squared <= radius ** 2

    target[mask] += amplitude

    return target


def add_solid_rectangle(
    target,
    grid_points,
    center,
    width,
    height,
    amplitude=1.0
):
    """
    Add a solid rectangular target.
    """

    cx, cy = center

    x_min = cx - width / 2
    x_max = cx + width / 2

    y_min = cy - height / 2
    y_max = cy + height / 2

    mask = (
        (grid_points[:, 0] >= x_min)
        &
        (grid_points[:, 0] <= x_max)
        &
        (grid_points[:, 1] >= y_min)
        &
        (grid_points[:, 1] <= y_max)
    )

    target[mask] += amplitude

    return target


def add_solid_square(
    target,
    grid_points,
    center,
    size,
    amplitude=1.0
):
    """
    Add a solid square target.
    """

    return add_solid_rectangle(
        target,
        grid_points,
        center,
        size,
        size,
        amplitude
    )


# ============================================================
# SMOOTH TARGETS
# ============================================================

def add_gaussian_target(
    target,
    grid_points,
    center,
    sigma,
    amplitude=1.0
):
    """
    Add a smooth Gaussian target.

    The attenuation gradually decreases away
    from the center.
    """

    cx, cy = center

    distance_squared = (
        (grid_points[:, 0] - cx) ** 2
        +
        (grid_points[:, 1] - cy) ** 2
    )

    gaussian = amplitude * np.exp(
        -distance_squared /
        (2 * sigma ** 2)
    )

    target += gaussian

    return target


# ============================================================
# MULTIPLE TARGETS
# ============================================================

def create_solid_targets(
    grid_points,
    target_centers,
    target_size=1.0,
    amplitude=1.0
):
    """
    Create multiple solid square targets.

    Parameters
    ----------
    target_centers : list of tuples
        [(x1,y1), (x2,y2), ...]
    """

    target = create_empty_target(
        grid_points
    )

    for center in target_centers:

        target = add_solid_square(
            target,
            grid_points,
            center,
            target_size,
            amplitude
        )

    return target


def create_smooth_targets(
    grid_points,
    target_centers,
    sigma=0.4,
    amplitude=1.0
):
    """
    Create multiple smooth Gaussian targets.
    """

    target = create_empty_target(
        grid_points
    )

    for center in target_centers:

        target = add_gaussian_target(
            target,
            grid_points,
            center,
            sigma,
            amplitude
        )

    return target


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from rti_model import generate_grid

    import matplotlib.pyplot as plt

    # Create grid
    X, Y, grid_points = generate_grid()

    # --------------------------------------------------------
    # Solid target
    # --------------------------------------------------------

    solid = create_solid_targets(
        grid_points,
        target_centers=[
            (3.5, 3.5)
        ],
        target_size=1.5
    )

    # --------------------------------------------------------
    # Smooth target
    # --------------------------------------------------------

    smooth = create_smooth_targets(
        grid_points,
        target_centers=[
            (3.5, 3.5)
        ],
        sigma=0.5
    )

    # Convert vectors to images
    solid_image = solid.reshape(
        X.shape
    )

    smooth_image = smooth.reshape(
        X.shape
    )

    # --------------------------------------------------------
    # Display solid target
    # --------------------------------------------------------

    plt.figure(figsize=(6, 5))

    plt.imshow(
        solid_image,
        origin="lower",
        extent=[
            0, 7,
            0, 7
        ]
    )

    plt.colorbar(
        label="Attenuation"
    )

    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")

    plt.title("Solid Target")

    plt.tight_layout()

    plt.show()

    # --------------------------------------------------------
    # Display smooth target
    # --------------------------------------------------------

    plt.figure(figsize=(6, 5))

    plt.imshow(
        smooth_image,
        origin="lower",
        extent=[
            0, 7,
            0, 7
        ]
    )

    plt.colorbar(
        label="Attenuation"
    )

    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")

    plt.title("Smooth Target")

    plt.tight_layout()

    plt.show()