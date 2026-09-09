# trajectories to test the different algorithms applied to the A1 robot

import numpy as np


def line_trajectory(length=6.0, n_points=600):
    """
    Straight line from the origin
    """
    x = np.linspace(0, length, n_points)
    y = np.zeros_like(x)
    return np.stack([x, y], axis=1)


def sine_trajectory(length=6.0, amplitude=0.5, wavelength=2.5, n_points=600):
    """
    Sine trajectory
    """
    x = np.linspace(0, length, n_points)
    y = amplitude * np.sin(2 * np.pi * x / wavelength)
    return np.stack([x, y], axis=1)

def cosine_trajectory(length=6.0, amplitude=0.15, wavelength=3.0, n_points=600):
    """
    Cosine trajectory squeezed
    """
    x = np.linspace(0, length, n_points)
    y = amplitude * (1 - np.cos(2 * np.pi * x / wavelength))
    return np.stack([x, y], axis=1)


def circle_trajectory(radius=2.0, n_points=600):
    """
    Circular trajectory
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = radius * np.sin(theta)
    y = radius * (1 - np.cos(theta))
    return np.stack([x, y], axis=1)


def closest_point_on_trajectory(position_xy, trajectory_xy):
    """
    Returns (min distance, index of nearest segment start) from the position and the trajectory.
    Calculates the orthogonal distance to the line segments connecting the points
    to eliminate discretization aliasing.
    """
    A = trajectory_xy[:-1]
    B = trajectory_xy[1:]
    
    AB = B - A
    AP = position_xy - A
    
    dot_AP_AB = np.sum(AP * AB, axis=1)
    dot_AB_AB = np.sum(AB * AB, axis=1)
    
    t = np.clip(dot_AP_AB / dot_AB_AB, 0.0, 1.0)
    
    C = A + t[:, np.newaxis] * AB
    
    dists = np.linalg.norm(position_xy - C, axis=1)
    idx = int(np.argmin(dists))
    min_dist = float(dists[idx])
    
    return min_dist, idx


TRAJECTORIES = {
    "line": line_trajectory(),
    "cosine": cosine_trajectory(),
    "circle": circle_trajectory(),
}
