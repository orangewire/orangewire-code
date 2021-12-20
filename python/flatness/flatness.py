#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

"""
Create a random orthogonal basis.
The first two vectors are in the plane, the last one
is the plane normal.
"""
def random_basis():
    # Unit plane normal
    n0 = np.random.rand(3)
    n0 = n0 / np.sqrt(np.sum(n0**2))
    # Plane basis
    u0 = np.array([1, -n0[0]/n0[1], 0], dtype='float')
    u0 = u0 / np.sqrt(np.sum(u0**2))
    v0 = np.cross(n0, u0)

    return (u0, v0, n0)


"""
Create a point cloud scattered around a plane.
"""
def generate_data(num_points: int, noise_amp: float, basis):
    u, v, n = basis
    # Random coordinate in the plane
    s = np.random.uniform(-10, 10, num_points)
    t = np.random.uniform(-10, 10, num_points)
    # Random deviation along the normal axis
    b = noise_amp * np.random.uniform(-1, 1, num_points)
    # Generate points
    X = s[:, np.newaxis]*u + t[:, np.newaxis]*v + b[:, np.newaxis]*n
    # Center data
    m = np.mean(X, axis=0)
    X = X - m

    return X


"""
Find the normal to the optimal plane that best fits
the input data points.
"""
def best_fit_normal(X):
    # Perform SVD
    U, Sigma, Vh = np.linalg.svd(X, full_matrices=True)
    # Find index of smallest singular value
    jmin = np.argmin(Sigma)

    # Return the right-singular vector at that index
    return Vh[jmin]


"""
Find the extremal distances to a plane specified by its
normal.
"""
def max_dist(X, normal):
    # Dot product of every point in X with the normal
    D = np.sum(normal*X, axis=1)

    return (np.min(D), np.max(D))


"""
Helper function to plot a plane.
"""
def plot_plane(ax, origin, normal, color, alpha):
    x = np.linspace(-13, 13, 2)
    y = np.linspace(-13, 13, 2)
    px, py = np.meshgrid(x, y)
    pz = - (px * normal[0] + py * normal[1]) / normal[2]
    ax.plot_surface(px + origin[0], py + origin[1],
                    pz + origin[2], color=color, alpha=alpha)


def main():
    basis = random_basis()
    X = generate_data(100, 2, basis)
    n_opt = best_fit_normal(X)
    dmin, dmax = max_dist(X, n_opt)
    F = dmax - dmin
    n_rms = np.linalg.norm(n_opt - basis[2])

    print(f'flatness score: {F}')
    print(f'norm RMS error: {n_rms}')

    # Plot the points, best fit plane and upper and lower planes
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plot_plane(ax, [0, 0, 0], n_opt, "red", 0.5)
    plot_plane(ax, dmin*n_opt, n_opt, "orange", 0.5)
    plot_plane(ax, dmax*n_opt, n_opt, "orange", 0.5)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    plt.show()


if __name__ == '__main__':
    main()
