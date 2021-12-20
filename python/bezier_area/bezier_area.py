#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.special


"""
Compute the area of a closed Bezier curve specified by
its control points.
"""
def bezier_area(P):
    R = np.array([[0, 1],
                  [-1, 0]])

    N = P.shape[0]-1
    A = 0
    for ii in range(1, N+1):
        for jj in range(0, ii):
            B = scipy.special.binom(
                N, ii) * scipy.special.binom(N, jj) / scipy.special.binom(2*N, ii+jj)
            D = jj/(ii+jj) - (N-jj)/(2*N-ii-jj)
            S = np.dot(P[ii, :], np.matmul(R, P[jj, :]))
            A += B * D * S

    return A


"""
Compute the approximate area of a Bezier curve by sampling
it in multiple points, and running a shoelace algorithm on
the resulting polygon.
"""
def approx_bezier_area(P, num_samples):
    def shoelace(x, y):
        return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

    def bezier_interp(P, t):
        N = P.shape[0]-1
        val = np.zeros((t.shape[0], 2), dtype='float')
        for ii in range(0, N+1):
            b = scipy.special.binom(N, ii)
            c = b * np.power(t, ii) * np.power(1-t, N-ii)
            val += c[:, np.newaxis] * P[ii, :]
        return val

    N = P.shape[0]-1
    tt = np.linspace(0, 1, num_samples)
    pts = bezier_interp(P, tt)
    return shoelace(pts[:, 0], pts[:, 1])


def main():
    P = np.array([[-0.5, -0.5], [1, -1.8], [1.5, -5],
                 [0.6, -2.3], [1, 1], [-1, 1], [-0.5, -0.5]])

    Approx = approx_bezier_area(P, 100)
    print(f'the approximate area is {Approx}')

    Area = bezier_area(P)
    print(f'the exact area is {Area}')


if __name__ == '__main__':
    main()
