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
        self.tangents[0] = (1 - tension) * free_tangents[0]
        self.tangents[-1] = (1 - tension) * free_tangents[1]

        # Formula for a generic cardinal spline
        for ii in range(1, self.size() - 1):
            self.tangents[ii] = (1 - tension) \
                * (self.control_points[ii+1] - self.control_points[ii-1])

    def calculate_segments(self):
        # Each spline segment is a cubic Bezier spline
        # If we have N control points, there are N-1 segments
        self.segments = []
        for ii in range(self.size()-1):
            bezier_control_points = np.array([
                self.control_points[ii],
                self.control_points[ii] + self.tangents[ii] / 3,
                self.control_points[ii+1] - self.tangents[ii+1] / 3,
                self.control_points[ii+1]
            ])
            self.segments.append(BezierSpline(bezier_control_points))

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


class ArclenHermiteSpline:
    def __init__(self, spline, samples):
        self.spline = spline
        self.calculate_lengths_iterative(samples)
        self.invert_arc_length(samples)

    def calculate_lengths_iterative(self, samples):
        # Sample the Hermite spline
        tp = np.linspace(0, 1, samples)
        rp = self.spline.sample(tp)
        # Calculate Euclidean distances between consecutive pairs of points
        distances = np.zeros(tp.shape)
        distances[1:] = np.linalg.norm(rp[1:,:] - rp[:-1,:], axis=1)
        # The arc length table is the prefix sum of these distances
        self.arc_length = np.cumsum(distances)

    def invert_arc_length(self, samples):
        last_idx = 0
        self.lut = np.zeros(samples)

        # Build the lookup table iteratively
        for ii in range(samples):
            s_bar = ii / (samples - 1)
            self.lut[ii], last_idx = self.arclen_remap(s_bar, last_idx)
        
        # Repeat the last value in order to avoid an out of bounds
        # error during sampling
        self.lut = np.append(self.lut, self.lut[-1])

    def arclen_remap(self, s_bar, last_idx):
        # Arc length from normalized arc length
        ss = np.clip(s_bar, 0, 1) * self.arc_length[-1]
        # Get the index of the largest arc length value that is
        # smaller than our target value ss
        idx = self.binary_search(ss, last_idx)

        if idx == self.arc_length.shape[0]-1:
            return 1, idx

        # The distance covered in the LUT by the binary search
        # algorithm is a measure of the inverse of the arc length
        len_before = self.arc_length[idx]
        len_after = self.arc_length[idx+1]
        len_segment = len_after - len_before
        frac = (ss - len_before) / len_segment
        length = (idx + frac) / (self.arc_length.shape[0]-1)
        return length, idx

    def binary_search(self, target, last_idx):
        lb = last_idx
        ub = self.arc_length.shape[0]
        idx = lb

        while lb < ub:
            idx = lb + (ub - lb) // 2
            if self.arc_length[idx] < target:
                lb = idx + 1
            else:
                ub = idx
        
        return idx - 1 if self.arc_length[idx] > target else idx

    def sample(self, s_bar):
        sclip = np.clip(s_bar, 0, 1)
        max_idx = self.lut.shape[0] - 2
        idxs = np.floor(sclip * max_idx).astype(int)
        alpha = max_idx * sclip - idxs
        tt = (1-alpha) * self.lut[idxs] + alpha * self.lut[idxs + 1]

        return self.spline.sample(tt)


def figure_bezier_spline():
    control_points = np.array([
        [0, 0], [0.5, 5], [5.2, 5.5], [3, 2.2]
    ])
    labels = [r'$P_0$', r'$P_1$', r'$P_2$', r'$P_3$']
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

    for ii, label in enumerate(labels):
        ax.annotate(label, control_points[ii] + np.array([0.1, 0]))

    ax.set_title('A cubic Bezier spline')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def figure_hermite_spline():
    control_points = np.array([
        [0, 0], [0.5, 5], [5.2, 5.5], [3, 2.2]
    ])
    labels = [r'$P_0$', r'$P_1$', r'$P_2$', r'$P_3$']
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

    for ii, label in enumerate(labels):
        ax.annotate(label, control_points[ii] + np.array([-0.4, -0.1]))

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
    labels = [r'$P_0$', r'$P_1$', r'$P_2$', r'$P_3$']
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

    for ii, label in enumerate(labels):
        ax.annotate(label, control_points[ii] + np.array([0.2, -0.3]))

    ax.set_title('The effect of tension')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def figure_arclen():
    control_points = np.array([
        [0, 0], [0.5, 5], [5.2, 5.5], [4, 4.1]
    ])
    labels = [r'$P_0$', r'$P_1$', r'$P_2$', r'$P_3$']
    free_tangents = np.array([
        [0,3], [-4,0]
    ])

    H = HermiteSpline(control_points, 0, free_tangents)
    AH = ArclenHermiteSpline(H, 100)

    tp = np.linspace(0, 1, 100)
    rp = H.sample(tp)

    tps = np.linspace(0, 1, 15)
    rps = H.sample(tps)

    tp_arclen = np.linspace(0, 1, 15)
    rp_arclen = AH.sample(tp_arclen)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(rp[:, 0], rp[:, 1])
    axs[1].plot(rp[:, 0], rp[:, 1])
    axs[0].scatter(control_points[:, 0], control_points[:, 1], color='red')
    axs[1].scatter(control_points[:, 0], control_points[:, 1], color='red')
    axs[0].scatter(rps[:, 0], rps[:, 1], linewidth=2, marker='+', s=150, color='limegreen')
    axs[1].scatter(rp_arclen[:, 0], rp_arclen[:, 1], linewidth=2, marker='+', s=150, color='limegreen')

    for ii, label in enumerate(labels):
        axs[0].annotate(label, control_points[ii] + np.array([0.2, -0.3]))
        axs[1].annotate(label, control_points[ii] + np.array([0.2, -0.3]))

    axs[0].set_title('Hermite spline')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_aspect('equal', adjustable='box')

    axs[1].set_title('Arc length parametrized\nHermite spline')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_aspect('equal', adjustable='box')
    plt.show()


def main(argv):
    # figure_bezier_spline()
    # figure_hermite_spline()
    # figure_hermite_tension()
    figure_arclen()




if __name__ == '__main__':
    main(sys.argv[1:])
