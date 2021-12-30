#!/usr/bin/python3
import sys
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt


def catenary(x, a, p, q):
    """
    The catenary function in its most generic form.
    """
    return a * np.cosh((x-p)/a) + q


def f(a, h, v, L):
    """
    This function will be used by the Newton-Raphson algorithm to find
    a value for 'a'.
    """
    return 2 * a * np.sinh(0.5 * h/a) - np.sqrt(np.power(L, 2)-np.power(v, 2))


def fprime(a, h):
    """
    The derivative of f will also be used by the Newton-Raphson algorithm to find
    a value for 'a'.
    """
    return 2 * np.sinh(0.5 * h/a) - (h/a) * np.cosh(0.5 * h/a)


def nr_first_guess(ff, start_x, start_step, alpha):
    """
    This helper function finds a good enough first guess for the value of 'a', and
    allows the Newton-Raphson algorithm to converge.
    """
    xx = start_x
    step = start_step
    yy = ff(xx)
    yy_prev = yy

    while yy * yy_prev > 0:
        yy_prev = yy
        xx += step
        step *= alpha
        yy = ff(xx)

    # Backtrack a bit
    return xx - 0.5 * step / alpha


def get_params(p1, p2, L):
    """
    Return the curve parameters 'a', 'p', 'q' as well as the integration
    constant 'c', given the input parameters.
    """
    hv = p2 - p1
    m = p1 + p2
    def f_bind(a): return f(a, *hv, L)
    def fprime_bind(a): return fprime(a, hv[0])

    # Newton-Raphson algorithm to find a value for 'a'
    a0 = nr_first_guess(f_bind, 0.1, 0.01, 1.8)
    a = optimize.newton(f_bind, a0, fprime_bind)

    # Use our formulas to compute the rest
    p = 0.5 * (m[0] - a * np.log((L+hv[1])/(L-hv[1])))
    q = 0.5 * (m[1] - L / np.tanh(0.5 * hv[0]/a))
    c = -a * np.sinh((p1[0]-p)/a)

    return a, p, q, c


def arclen_remap(s_bar, L, a, p, c):
    """
    Map a normalized length fraction between 0 and 1 to an argument
    value to feed the catenary function
    """
    return a * np.arcsinh((s_bar * L - c) / a) + p


def main(argv):
    p1 = np.array([0.5, 0.6])
    p2 = np.array([4.1, 2.5])
    min_len = np.linalg.norm(p2-p1)

    fig, axs = plt.subplots(1, 2)

    for ii in range(1, 4):
        L = ii * min_len + 0.1
        a, p, q, c = get_params(p1, p2, L)
        # print(L, a, p, q, c)

        xp = np.linspace(p1[0], p2[0], 100)
        yp = catenary(xp, a, p, q)

        xps = np.linspace(p1[0], p2[0], 15)
        yps = catenary(xps, a, p, q)

        s_bar = np.linspace(0, 1, 15)
        xps_arclen = arclen_remap(s_bar, L, a, p, c)
        yps_arclen = catenary(xps_arclen, a, p, q)

        axs[0].plot(xp, yp, linewidth=2)
        axs[0].plot(xps, yps, linewidth=10, linestyle='solid', color='black', alpha=0.15)
        axs[0].scatter(xps, yps, linewidth=2, marker='+', s=100, color='red')

        axs[1].plot(xp, yp, linewidth=2)
        axs[1].plot(xps_arclen, yps_arclen, linewidth=10, linestyle='solid', color='black', alpha=0.15)
        axs[1].scatter(xps_arclen, yps_arclen, linewidth=2, marker='+', s=100, color='red')

    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Initial parametrization')
    axs[0].set_aspect('equal', adjustable='box')

    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title('Arc length parametrization')
    axs[1].set_aspect('equal', adjustable='box')

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
