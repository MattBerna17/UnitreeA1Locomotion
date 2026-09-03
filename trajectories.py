"""
Traiettorie di riferimento per testare il tracking del robot A1, e
funzioni di supporto per calcolare la distanza (cross-track error) tra
la posizione del robot e il percorso.
"""

import numpy as np


def line_trajectory(length=6.0, n_points=600):
    """Linea retta lungo l'asse x, partendo dall'origine."""
    x = np.linspace(0, length, n_points)
    y = np.zeros_like(x)
    return np.stack([x, y], axis=1)


def sine_trajectory(length=6.0, amplitude=0.5, wavelength=2.5, n_points=600):
    """Traiettoria sinusoidale, avanzamento lungo x, oscillazione su y."""
    x = np.linspace(0, length, n_points)
    y = amplitude * np.sin(2 * np.pi * x / wavelength)
    return np.stack([x, y], axis=1)

def cosine_trajectory(length=6.0, amplitude=0.15, wavelength=3.0, n_points=600):
    """
    Traiettoria cosinusoidale con ampiezza ridotta.
    La formula A * (1 - cos(kx)) garantisce che la curva parta da y=0 
    con una tangente perfettamente parallela all'asse X, permettendo al 
    robot di partire dritto senza scatti iniziali.
    """
    x = np.linspace(0, length, n_points)
    # 1 - cos(...) fa partire la traiettoria dolce dall'origine
    y = amplitude * (1 - np.cos(2 * np.pi * x / wavelength))
    return np.stack([x, y], axis=1)


def circle_trajectory(radius=2.0, n_points=600):
    """
    Traiettoria circolare, tangente all'asse x nell'origine (il robot
    parte "dritto" e la curvatura comincia gradualmente).
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = radius * np.sin(theta)
    y = radius * (1 - np.cos(theta))
    return np.stack([x, y], axis=1)


def closest_point_on_trajectory(position_xy, trajectory_xy):
    """
    Ritorna (distanza minima, indice del punto piu' vicino) tra la
    posizione data e la traiettoria discretizzata.
    """
    diffs = trajectory_xy - position_xy[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return float(dists[idx]), idx


TRAJECTORIES = {
    "line": line_trajectory(),
    "cosine": cosine_trajectory(),
    "circle": circle_trajectory(),
}
