#!/usr/bin/python3
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from hermite import BezierSpline, HermiteSpline, ArclenHermiteSpline


def figure_bezier_spline():
    control_points = np.array([
        [0, 0], [0.5, 5], [5.2, 5.5], [3, 2.2]
    ])
    labels = [r'$P_0$', r'$P_1$', r'$P_2$', r'$P_3$']
    B = BezierSpline(control_points)

    xp = np.linspace(0, 1, 50)
    rp = B.sample(xp)

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
        [0, 3], [-4, 0]
    ])

    H = HermiteSpline(control_points, 0, free_tangents)

    xp = np.linspace(0, 1, 100)
    rp = H.sample(xp)

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
    xp = np.linspace(0, 1, 100)
    for ii, tension in enumerate(tensions):
        H = HermiteSpline(control_points, tension, free_tangents)
        rp = H.sample(xp)
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
        [0, 3], [-4, 0]
    ])

    H = HermiteSpline(control_points, 0, free_tangents)
    AH = ArclenHermiteSpline(H, 100)

    xp = np.linspace(0, 1, 100)
    rp = H.sample(xp)

    xps = np.linspace(0, 1, 15)
    rps = H.sample(xps)

    xp_arclen = np.linspace(0, 1, 15)
    rp_arclen = AH.sample(xp_arclen)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(rp[:, 0], rp[:, 1])
    axs[1].plot(rp[:, 0], rp[:, 1])
    axs[0].scatter(control_points[:, 0], control_points[:, 1], color='red')
    axs[1].scatter(control_points[:, 0], control_points[:, 1], color='red')
    axs[0].scatter(rps[:, 0], rps[:, 1], linewidth=2,
                   marker='+', s=150, color='limegreen')
    axs[1].scatter(rp_arclen[:, 0], rp_arclen[:, 1], linewidth=2,
                   marker='+', s=150, color='limegreen')

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


def figure_ifrac():
    control_points = np.array([
        [-5, 0], [-3, -1], [0, 0], [3, 1], [5, 0]
    ])
    free_tangents = np.zeros([2, 2])

    H = HermiteSpline(control_points, 0, free_tangents)
    AH = ArclenHermiteSpline(H, 10)

    xp = np.linspace(0, 1, 100)
    rp = H.sample(xp)
    x_pts = np.array([0.4, 0.45, 0.6])
    r_pts = AH.sample(x_pts)

    fig, ax = plt.subplots(1)
    dashed = 20
    curve_color = 'blue'
    ax.plot(rp[dashed:-dashed, 0], rp[dashed:-dashed, 1], color=curve_color)
    ax.plot(rp[0:dashed+1, 0], rp[0:dashed+1, 1],
            color=curve_color, linestyle='dashed')
    ax.plot(rp[-dashed-1:, 0], rp[-dashed-1:, 1],
            color=curve_color, linestyle='dashed')
    ax.scatter(r_pts[:, 0], r_pts[:, 1], linewidth=2,
               marker='+', s=150, color='limegreen')

    offset = 0.4
    x_offset = 0.02
    x_la = np.linspace(0, x_pts[2]-x_offset, 20)
    r_la = AH.sample(x_la)
    x_lb = np.linspace(0, x_pts[0]-2*x_offset, 20)
    r_lb = AH.sample(x_lb)
    x_seg = np.linspace(x_pts[0]+x_offset, x_pts[2]+x_offset)
    r_seg = AH.sample(x_seg)
    x_slb = np.linspace(x_pts[0]+2*x_offset, x_pts[1]+2*x_offset)
    r_slb = AH.sample(x_slb)
    ax.plot(r_la[:, 0], r_la[:, 1]+offset, color='black')
    ax.plot(r_lb[:, 0], r_lb[:, 1]+2*offset, color='black')
    ax.plot(r_seg[:, 0], r_seg[:, 1]-offset, color='black')
    ax.plot(r_slb[:, 0], r_slb[:, 1]-2*offset, color='black')

    pts_labels = [r'$s_i$', r'$s$', r'$s_{i+1}$']
    for ii, label in enumerate(pts_labels):
        ax.annotate(label, r_pts[ii] + np.array([0.1, -0.2]))

    len_labels = [r'$l_{after}$', r'$l_{before}$', r'$s-l_{before}$', r'$l_{segment}$']
    pos = np.array([r_la[-1,:]+np.array([-0.4,0.5]), r_lb[-1,:]+np.array([-0.4,1]), r_slb[-1,:]+np.array([0,-1]), r_seg[-1,:]+np.array([0,-0.7])])
    for ii, label in enumerate(len_labels):
        ax.annotate(label, pos[ii])

    ax.set_aspect('equal', adjustable='box')
    plt.show()


def main(argv):
    # figure_bezier_spline()
    # figure_hermite_spline()
    figure_hermite_tension()
    # figure_ifrac()
    # figure_arclen()


if __name__ == '__main__':
    main(sys.argv[1:])
