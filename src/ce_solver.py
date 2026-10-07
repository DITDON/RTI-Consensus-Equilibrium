import numpy as np


def data_agent(x, W, y, step=0.1):
    """
    Data-fidelity agent.

    Moves the current reconstruction x toward
    consistency with the measured RSS data.
    """

    x = np.asarray(x, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    prediction = W @ x
    residual = y - prediction

    correction = W.T @ residual

    norm = np.linalg.norm(W, ord=2) ** 2 + 1e-8

    return x + step * correction / norm


def denoiser_agent(x, denoiser, device):
    """
    CNN denoising agent.

    Input:
        x -> flattened reconstruction

    Output:
        denoised reconstruction
    """

    import torch

    x_tensor = torch.tensor(
        x,
        dtype=torch.float32,
        device=device
    )

    side = int(np.sqrt(len(x)))

    x_tensor = x_tensor.reshape(
        1, 1, side, side
    )

    with torch.no_grad():
        output = denoiser(x_tensor)

    return output.squeeze().cpu().numpy().reshape(-1)


def consensus_equilibrium(
    x0,
    W,
    y,
    denoiser,
    device,
    iterations=20,
    data_step=0.1,
    relaxation=0.5
):
    """
    Consensus Equilibrium reconstruction.

    Combines:
        1. Data fidelity
        2. CNN denoising

    Parameters
    ----------
    x0 : initial reconstruction
    W : RTI weight matrix
    y : RSS measurements
    denoiser : trained CNN denoiser
    device : cpu / cuda
    iterations : number of CE iterations
    data_step : data-agent step size
    relaxation : consensus relaxation
    """

    x = np.asarray(x0, dtype=np.float32).copy()

    history = []

    for iteration in range(iterations):

        # Data fidelity agent
        data_output = data_agent(
            x,
            W,
            y,
            step=data_step
        )

        # CNN denoiser agent
        denoised_output = denoiser_agent(
            x,
            denoiser,
            device
        )

        # Consensus update
        new_x = (
            (1 - relaxation) * x
            + relaxation * 0.5 * (
                data_output + denoised_output
            )
        )

        change = np.linalg.norm(new_x - x)

        history.append(change)

        x = new_x

        print(
            f"CE Iteration {iteration + 1:02d}/{iterations} "
            f"| Change = {change:.6f}"
        )

        if change < 1e-5:
            print("CE converged.")

            break

    return x, history