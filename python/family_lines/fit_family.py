#!/usr/bin/python3
import sys
import numpy as np
from matplotlib import pyplot as plt


def get_data():
    """
    Generate a set of lines in slope-intercept form.
    """

    # Context: The data here comes from simulations I ran a few years ago
    # to optimize a Yagi antenna. I was trying to characterize the
    # effect of changing the spacing between the dipole and the
    # reflector. Each configuration would show up as a line in an
    # inductive reactance vs. resistance diagram, that I used for
    # perfect matching purposes.
    #
    # Each line in the data matrix gathers simulation data for multiple values
    # of the spacing between the dipole and the reflector, from 41mm to 138mm:
    # Z[:,0] = values (mm) for spacing between the reflector and driven element
    # Z[:,1] = Antenna resistance for a 286mm dipole @435MHz
    # Z[:,2] = Antenna inductive reactance for a 286mm dipole @435MHz
    # Z[:,3] = Antenna resistance for a 296mm dipole @435MHz
    # Z[:,4] = Antenna inductive reactance for a 296mm dipole @435MHz
    Z = np.array([
        [41,	13.03,	-94.7,	10.83,	-46.86],
        [55,	18.64,	-83.62,	17.0,	-27.89],
        [69,	24.92,	-69.18,	24.08,	-7.561],
        [83,	32.07,	-53.72,	32.21,	12.71],
        [96,	39.59,	-39.22,	40.82,	31.02],
        [110,	48.79,	-23.93,	51.39,	49.94],
        [124,	59.34,	-9.239,	63.6,	67.85],
        [138,	71.44,	4.532,	77.64,	84.51]
    ])

    a = (Z[:, 4]-Z[:, 2])/(Z[:, 3]-Z[:, 1])
    b = Z[:, 2]-a*Z[:, 1]

    # Also return the first column as our parameter values s_i
    return Z[:,0], a, b


def to_hesse_normal_form(a, b):
    """
    Convert a slope-intercept form to a Hesse normal form
    """

    r = - b / (a * np.sqrt(1 + 1 / np.power(a, 2)))
    theta = np.arctan(-1 / a)
    return r, theta


def to_slope_intercept_form(r, theta):
    """
    Convert a Hesse normal form to a slope-intercept form
    """

    a = - 1 / np.tan(theta)
    b = r / np.sin(theta)
    return a, b


def plot_lines(ax, a, b, s_bnd, color, style, width):
    """
    Helper function to plot multiple lines specified by
    slope-intercept form
    """
    
    for aa, bb in zip(a, b):
        ax.plot(s_bnd, aa * s_bnd + bb, color=color, linestyle=style, linewidth=width)


def main(argv):
    s, a, b = get_data()
    # Fit the Hesse normal form parameters using 2nd order polynomials
    r, theta = to_hesse_normal_form(a, b)
    r_coeffs = np.polyfit(s, r, 2)
    theta_coeffs = np.polyfit(s, theta, 2)
    r_fit = np.poly1d(r_coeffs)
    theta_fit = np.poly1d(theta_coeffs)

    # For multiple values of the parameter s, calculate slope and y-intercept
    sp = np.linspace(np.min(s), np.max(s), 100)
    a_fit, b_fit = to_slope_intercept_form(r_fit(sp), theta_fit(sp))

    # Plot: fit
    fig1, axs = plt.subplots(2, 2)
    axs[0][0].plot(sp, r_fit(sp), color='green')
    axs[0][0].scatter(x=s, y=r, marker='+', s=100, color='red')
    axs[0][0].set_title('r: polynomial fit')
    axs[0][0].set_xlabel('s')
    axs[0][0].set_ylabel('r')

    axs[1][0].plot(sp, theta_fit(sp), color='green')
    axs[1][0].scatter(x=s, y=theta, marker='+', s=100, color='red')
    axs[1][0].set_title(r'$\theta$: polynomial fit')
    axs[1][0].set_xlabel('s')
    axs[1][0].set_ylabel(r'$\theta$')

    axs[0][1].plot(sp, a_fit, color='blue')
    axs[0][1].scatter(x=s, y=a, marker='+', s=100, color='red')
    axs[0][1].set_title('a: best fit')
    axs[0][1].set_xlabel('s')
    axs[0][1].set_ylabel('a')

    axs[1][1].plot(sp, b_fit, color='blue')
    axs[1][1].scatter(x=s, y=b, marker='+', s=100, color='red')
    axs[1][1].set_title('b: best fit')
    axs[1][1].set_xlabel('s')
    axs[1][1].set_ylabel('b')

    # Plot: parametrized family of lines
    s_bnd = np.array([np.min(s), np.max(s)])
    fig2, ax = plt.subplots(1)
    plot_lines(ax, a, b, np.array([-1e2,1e2]), 'orange', 'solid', 4)
    plot_lines(ax, a_fit, b_fit, np.array([-1e2,1e2]), 'black', 'dotted', 1)
    ax.set_xlim([0,80])
    ax.set_ylim([0,100])
    ax.set_title('Parametrized family')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
