import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# RTI PARAMETERS
# ============================================================

AREA_WIDTH = 7.0
AREA_HEIGHT = 7.0

GRID_SIZE = 30

ELLIPSE_WIDTH = 0.5


# ============================================================
# 1. UNIFORM SENSOR PLACEMENT
# ============================================================

def generate_uniform_sensors(
    num_sensors,
    width=AREA_WIDTH,
    height=AREA_HEIGHT
):
    """
    Generate sensors uniformly along the perimeter
    of the RTI area.
    """

    perimeter = 2 * (width + height)

    # Equal spacing along the perimeter
    distances = np.linspace(
        0,
        perimeter,
        num_sensors,
        endpoint=False
    )

    sensors = []

    for distance in distances:

        # Bottom edge
        if distance < width:

            x = distance
            y = 0.0

        # Right edge
        elif distance < width + height:

            x = width
            y = distance - width

        # Top edge
        elif distance < 2 * width + height:

            x = width - (
                distance - width - height
            )
            y = height

        # Left edge
        else:

            x = 0.0
            y = height - (
                distance - 2 * width - height
            )

        sensors.append([x, y])

    return np.array(sensors)


# ============================================================
# 2. GENERATE SENSOR LINKS
# ============================================================

def generate_links(num_sensors):
    """
    Generate every unique pair of sensors.
    """

    links = []

    for i in range(num_sensors):

        for j in range(i + 1, num_sensors):

            links.append((i, j))

    return links


# ============================================================
# 3. GENERATE RECONSTRUCTION GRID
# ============================================================

def generate_grid(
    grid_size=GRID_SIZE,
    width=AREA_WIDTH,
    height=AREA_HEIGHT
):

    x = np.linspace(
        width / (2 * grid_size),
        width - width / (2 * grid_size),
        grid_size
    )

    y = np.linspace(
        height / (2 * grid_size),
        height - height / (2 * grid_size),
        grid_size
    )

    X, Y = np.meshgrid(x, y)

    grid_points = np.column_stack(
        (X.ravel(), Y.ravel())
    )

    return X, Y, grid_points


# ============================================================
# 4. BUILD RTI WEIGHT MATRIX
# ============================================================

def build_weight_matrix(
    sensors,
    links,
    grid_points,
    ellipse_width=ELLIPSE_WIDTH
):

    num_links = len(links)
    num_pixels = len(grid_points)

    W = np.zeros(
        (num_links, num_pixels)
    )

    for link_index, (i, j) in enumerate(links):

        sensor_1 = sensors[i]
        sensor_2 = sensors[j]

        # Distance between sensors
        link_distance = np.linalg.norm(
            sensor_1 - sensor_2
        )

        # Distance from every pixel
        # to sensor 1
        d1 = np.linalg.norm(
            grid_points - sensor_1,
            axis=1
        )

        # Distance from every pixel
        # to sensor 2
        d2 = np.linalg.norm(
            grid_points - sensor_2,
            axis=1
        )

        # Elliptical region
        inside_ellipse = (
            d1 + d2
            < link_distance + ellipse_width
        )

        # Assign weight
        W[
            link_index,
            inside_ellipse
        ] = 1.0 / np.sqrt(link_distance)

    return W


# ============================================================
# 5. PLOT SENSOR NETWORK
# ============================================================

def plot_sensor_network(
    sensors,
    links,
    width=AREA_WIDTH,
    height=AREA_HEIGHT,
    title="RTI Sensor Network"
):

    plt.figure(figsize=(7, 7))

    # Plot links
    for i, j in links:

        plt.plot(
            [
                sensors[i, 0],
                sensors[j, 0]
            ],
            [
                sensors[i, 1],
                sensors[j, 1]
            ],
            linewidth=0.4,
            alpha=0.25
        )

    # Plot sensors
    plt.scatter(
        sensors[:, 0],
        sensors[:, 1],
        s=70,
        zorder=3
    )

    # Sensor numbers
    for index, (x, y) in enumerate(sensors):

        plt.text(
            x,
            y,
            str(index),
            fontsize=8,
            ha="center",
            va="bottom"
        )

    plt.xlim(
        -0.3,
        width + 0.3
    )

    plt.ylim(
        -0.3,
        height + 0.3
    )

    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")

    plt.title(title)

    plt.gca().set_aspect("equal")

    plt.grid(
        True,
        alpha=0.2
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("UNIFORM SENSOR CONFIGURATION TEST")
    print("=" * 50)

    for num_sensors in [16, 24, 32]:

        sensors = generate_uniform_sensors(
            num_sensors
        )

        links = generate_links(
            num_sensors
        )

        expected_links = (
            num_sensors *
            (num_sensors - 1)
        ) // 2

        print()
        print(
            f"Sensors  : {num_sensors}"
        )

        print(
            f"Links    : {len(links)}"
        )

        print(
            f"Expected : {expected_links}"
        )

        print(
            "First five sensor positions:"
        )

        print(
            sensors[:5]
        )

        plot_sensor_network(
            sensors,
            links,
            title=f"{num_sensors}-Node RTI Network"
        )