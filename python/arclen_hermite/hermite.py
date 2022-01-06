#!/usr/bin/python3
import sys
import numpy as np
import scipy.special


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

    def to_local_space(self, xx):
        # Find segment index corresponding to each value of xx
        tclip = np.clip(xx, 0, 1)
        idxs = np.clip(np.floor((self.size()-1) * tclip), 0, self.size() - 2)
        # Convert xx to local parameter value
        tt = xx * (self.size()-1) - idxs

        # Return an array that aggregates segment indices and local parameter value
        return np.dstack((idxs, tt))[0]

    def sample(self, xx):
        # Group local segment parameter values by index
        loc = self.to_local_space(xx)
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
        distances[1:] = np.linalg.norm(rp[1:, :] - rp[:-1, :], axis=1)
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
        max_idx = self.arc_length.shape[0]-1

        if idx == max_idx:
            return 1, idx

        # The distance covered in the LUT by the binary search
        # algorithm is a measure of the inverse of the arc length
        len_before = self.arc_length[idx]
        len_after = self.arc_length[idx+1]
        len_segment = len_after - len_before
        frac = (ss - len_before) / len_segment
        xx = (idx + frac) / max_idx
        return xx, idx

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
        # Get the values xx of the curve parameter corresponding
        # to the normalized arc lengths s_bar
        sclip = np.clip(s_bar, 0, 1)
        max_idx = self.lut.shape[0] - 2
        idxs = np.floor(sclip * max_idx).astype(int)
        alpha = max_idx * sclip - idxs
        xx = (1-alpha) * self.lut[idxs] + alpha * self.lut[idxs + 1]

        # Sample the spline
        return self.spline.sample(xx)
