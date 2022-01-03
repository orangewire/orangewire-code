#!/usr/bin/python3
import sys
import math
import numpy as np
import scipy.special
from numpy.core.fromnumeric import shape
from matplotlib import pyplot as plt
from matplotlib import cm


class BezierSpline:
    def __init__(self, control_points):
        self.control_points = control_points
        self.calculate_coefficients()

    def calculate_coefficients(self):
        self.coefficients = np.zeros(self.control_points.shape)
        N = self.control_points.shape[0]
        prod = 1
        for jj in range(N):
            # Product from m=0 to j-1 of (n-m)
            prod *= N - jj if jj > 0 else 1
            # Weighted sum from i=0 to j of the control points
            ii = np.array(range(jj+1))
            factor = np.power(-1, ii+jj) * prod / (scipy.special.factorial(ii)
                                                   * scipy.special.factorial(jj - ii))
            self.coefficients[jj, :] = np.dot(
                factor, self.control_points[0:jj+1])

    def sample(self, tt):
        N = self.control_points.shape[0]
        tpow = np.power(tt[:, np.newaxis], np.array(range(N)))
        return np.matmul(tpow, self.coefficients)


class HermiteSpline:
    def __init__(self, control_points, tension, free_tangents):
        self.control_points = control_points
        self.calculate_tangents(tension, free_tangents)
        self.calculate_segments()

    def calculate_tangents(self, tension, free_tangents):
        # Initialize start and end tangents
        self.tangents = np.zeros(self.control_points.shape)
        self.tangents[0] = free_tangents[0]
        self.tangents[-1] = free_tangents[1]

        # Formula for a generic cardinal spline
        for ii in range(1, self.size() - 1):
            self.tangents[ii] = (
                1 - tension) * (self.control_points[ii+1] - self.control_points[ii-1])

    def calculate_segments(self):
        # Each spline segment is a cubic Bezier spline
        # If we have N control points, there are N-1 segments
        self.segments = np.empty(self.control_points.shape[0]-1, dtype=object)
        for ii in range(self.size()-1):
            bezier_control_points = np.array([
                self.control_points[ii],
                self.control_points[ii] + self.tangents[ii] / 3,
                self.control_points[ii+1] - self.tangents[ii+1] / 3,
                self.control_points[ii+1]
            ])
            self.segments[ii] = BezierSpline(bezier_control_points)

    def size(self):
        return self.control_points.shape[0]

    def to_local_space(self, tt):
        # Find segment index corresponding to each value of tt
        tclip = np.clip(tt, 0, 1)
        idxs = np.clip(np.floor((self.size()-1) * tclip), 0, self.size() - 2)
        # Convert tt to local parameter value
        tlocal = tt * (self.size()-1) - idxs

        # Return an array that aggregates segment indices and local parameter value
        return np.dstack((idxs, tlocal))[0]

    def sample(self, tt):
        # Group local segment parameter values by index
        loc = self.to_local_space(tt)
        unique = np.unique(loc[:, 0], return_index=True)
        seg_tt = np.split(loc[:, 1], unique[1][1:])

        # Sample each segment
        return np.concatenate([
            self.segments[int(idx)].sample(tloc)
            for idx, tloc in zip(unique[0], seg_tt)
        ])


def figure_bezier_spline():
    control_points = np.array([
        [0, 0], [0.5, 5], [5.2, 5.5], [3, 2.2]
    ])
    B = BezierSpline(control_points)

    tp = np.linspace(0, 1, 50)
    rp = B.sample(tp)

    fig, ax = plt.subplots(1)
    ax.plot(rp[:, 0], rp[:, 1])
    ax.scatter(control_points[:, 0], control_points[:, 1], color='red')
    ax.plot(control_points[0:2, 0], control_points[0:2,
            1], color='gray', linestyle='dashed')
    ax.plot(control_points[2:, 0], control_points[2:, 1],
            color='gray', linestyle='dashed')
    ax.set_title('A cubic Bezier spline')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def figure_hermite_spline():
    control_points = np.array([
        [0, 0], [0.5, 5], [5.2, 5.5], [3, 2.2]
    ])
    # free_tangents = np.zeros([2, 2])
    free_tangents = np.array([
        [0,3], [-4,0]
    ])

    H = HermiteSpline(control_points, 0, free_tangents)

    tp = np.linspace(0, 1, 100)
    rp = H.sample(tp)

    fig, ax = plt.subplots(1)
    ax.plot(rp[:, 0], rp[:, 1])
    ax.scatter(control_points[:, 0], control_points[:, 1], color='red')

    for segment in H.segments:
        ax.scatter(
            segment.control_points[:, 0], segment.control_points[:, 1], marker='+', color='green', s=100)
        ax.plot(segment.control_points[0:2, 0], segment.control_points[0:2,
                1], color='gray', linestyle='dashed')
        ax.plot(segment.control_points[2:, 0], segment.control_points[2:, 1],
                color='gray', linestyle='dashed')

    ax.set_title('A cubic Hermite spline')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def figure_hermite_tension():
    control_points = np.array([
        [0, 0], [0.5, 5], [5.2, 5.5], [3, 2.2]
    ])
    free_tangents = np.zeros([2, 2])

    fig, ax = plt.subplots(1)
    tensions = np.linspace(0, 1, 10)
    colors = cm.RdYlBu(tensions)[::-1]
    tp = np.linspace(0, 1, 100)
    for ii, tension in enumerate(tensions):
        H = HermiteSpline(control_points, tension, free_tangents)
        rp = H.sample(tp)
        ax.plot(rp[:, 0], rp[:, 1], color=colors[ii])

    ax.scatter(control_points[:, 0], control_points[:, 1], color='red')
    ax.set_title('The effect of tension')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def main(argv):
    # figure_bezier_spline()
    figure_hermite_spline()
    # figure_hermite_tension()


if __name__ == '__main__':
    main(sys.argv[1:])
