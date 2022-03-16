#!/usr/bin/python3
import sys
import numpy as np
import numpy.polynomial.polynomial as P
from matplotlib import pyplot as plt


def main(argv):
    points = np.array([
        [1,1],
        [2,2],
        [3,3],
        [4,4],
        [5,5],
        [6,np.pi]
    ])

    roots = points[:,0]
    values = points[:,1]

    # np.set_printoptions(precision=3, suppress=True)

    polys = []
    for ii in range(len(roots)):
        r = roots[np.arange(len(roots))!=ii]
        poly = P.polyfromroots(r)
        yi = P.polyval(roots[ii], poly)
        polys.append(poly / yi)

    polys = np.array(polys)
    interp = np.sum(values[:, np.newaxis] * polys, axis=0)

    print(P.polyval(roots, interp))

    xp = np.linspace(0, 10, 100)

    fig, axs = plt.subplots(2,1)

    for poly in polys:
        yp = P.polyval(xp, poly)
        axs[0].plot(xp, yp)

    yp = P.polyval(xp, interp)
    axs[1].plot(xp, yp)

    axs[0].set_xlim([np.min(roots)-1, np.max(roots)+1])
    axs[0].set_ylim([-2,2])
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    axs[1].set_xlim([np.min(roots)-1, np.max(roots)+1])
    axs[1].set_ylim([-10,10])
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])