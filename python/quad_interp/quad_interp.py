#!/usr/bin/python3
import sys
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon

"""
Computes a quadratic form with all coefficients set to one, as an array.
Then, dotting this array with a conformant coefficient array gives a value.
"""


def qform(xx, yy):
    return np.array([xx ** 2, xx*yy, yy ** 2, xx, yy, 1])


"""
Normalize arrays of X and Y coordinates as well as an array Z of
values between 0 and 1
"""


def normalize(X, Y, Z):
    xmin = np.min(X)
    ymin = np.min(Y)
    zmin = np.min(Z)
    Xn = X - xmin
    Yn = Y - ymin
    Zn = Z - zmin
    xmax = np.max(Xn)
    ymax = np.max(Yn)
    zmax = np.max(Zn)
    Xn /= xmax
    Yn /= ymax
    Zn /= zmax
    return (Xn, Yn, Zn, xmin, xmax, ymin, ymax, zmin, zmax)


"""
Transform a point to normalized coordinates, given the extremal values
returned by the normalize() function
"""


def to_normalized_coord(X, Y, xmin, xmax, ymin, ymax):
    return ((X-xmin)/xmax, (Y-ymin)/ymax)


"""
Return the generalized quad interpolant that minimizes the quadratic coefficients
"""


def quad_interpolation(XX, YY, VV):
    # Renormalize neighbor coordinates for better stability
    Xn, Yn, Vn, xmin, xmax, ymin, ymax, zmin, zmax = normalize(XX, YY, VV)

    # Value vector, 4 values for the Lagrange multipliers,
    # 6 zeros for stationary points where the Lagrangian vanishes
    b = np.append(Vn, np.zeros(6))

    # We now form the matrix A of the linear system to be solved
    E = np.diag([1, 1, 1, 0, 0, 0])
    X = np.array([qform(Xn[0], Yn[0]),
                  qform(Xn[1], Yn[1]),
                  qform(Xn[2], Yn[2]),
                  qform(Xn[3], Yn[3])])
    A = np.block([
        [X, np.zeros([4, 4])],
        [E, X.T]
    ])

    # And solve the linear system for the quadratic form coefficients
    a = np.linalg.solve(A, b)
    # a = np.linalg.lstsq(A, b)[0]

    # np.set_printoptions(precision=3)
    # print(A)
    # print(a)

    def interp(xx, yy):
        # Transform xx and yy to normalized coordinate space
        xn, yn = to_normalized_coord(xx, yy, xmin, xmax, ymin, ymax)
        # Evaluate quadratic form and transform result to original value space
        return np.dot(a[0:6], qform(xn, yn)) * zmax + zmin

    return interp


def main(argv):
    # These are the point coordinates for which we know a value
    X = np.array([20, 21, 29, 25], dtype='float')
    Y = np.array([1.5, 7.9, 12.154, 0.238], dtype='float')

    # These are the RGB color values for each point
    r = np.array([1, 0.7, 0, 0.4], dtype='float')
    g = np.array([0.3, 1, 0.2, 0.2], dtype='float')
    b = np.array([0.1, 0, 0, 1], dtype='float')

    # Get an interpolant for each color channel
    r_interp = quad_interpolation(X, Y, r)
    g_interp = quad_interpolation(X, Y, g)
    b_interp = quad_interpolation(X, Y, b)

    # Plot stuff
    # Construct a mesh grid
    xmin = np.min(X)
    ymin = np.min(Y)
    xmax = np.max(X)
    ymax = np.max(Y)
    xls = np.linspace(xmin, xmax, 50)
    yls = np.linspace(ymin, ymax, 50)
    xv, yv = np.meshgrid(xls, yls)

    # Interpolate colors and clamp the results between 0 and 1
    r_int = np.clip(r_interp(xv, yv), 0, 1)
    g_int = np.clip(g_interp(xv, yv), 0, 1)
    b_int = np.clip(b_interp(xv, yv), 0, 1)
    C = np.dstack((r_int, g_int, b_int))

    # Plot the quadrilateral itself
    poly = Polygon(np.dstack((X, Y))[0])
    poly_x, poly_y = poly.exterior.xy

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(poly_x, poly_y, color='black')

    # Plot each known point as a colored disk
    ax.scatter(X, Y, color=np.dstack((r, g, b))[
        0], edgecolor='black', linewidth=2, s=100)

    # Display interpolated colors as an image
    ax.imshow(C, extent=[xmin, xmax, ymin, ymax], origin='lower')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xlim(xmin-1, xmax+1)
    plt.ylim(ymin-1, ymax+1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
