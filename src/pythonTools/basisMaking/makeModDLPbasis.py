from collections import OrderedDict
import os
import sys
sys.path.append('~/workspace/unotran/src')
import pydgm
import numpy as np
import scipy as sp
from scipy.io import FortranFile
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif'})
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['lines.linewidth'] = 1.85
rcParams['axes.labelsize'] = 20
rcParams.update({'figure.autolayout': True})
np.set_printoptions(linewidth=132)


def getSpectrum(G, geoOption):
    try:
        phi = np.load('inf_phi_g{}_{}.npy'.format(G, geoOption))
    except IOError:
        setVariables(G, geoOption)
        pydgm.solver.initialize_solver()
        pydgm.solver.solve()
        phi = np.copy(pydgm.state.phi)
        pydgm.solver.finalize_solver()
        np.save('inf_phi_g{}_{}'.format(G, geoOption), phi)
        # plot(phi[0])
    return phi[0, 0]


def setVariables(G, geoOption):
    # Set the variables for the discrete ordinance problem

    setGeometry(geoOption)
    pydgm.control.boundary_type = [1.0, 1.0]  # reflective boundarys
    pydgm.control.solver_type = 'eigen'.ljust(256)
    pydgm.control.allow_fission = True
    pydgm.control.source_value = 0.0
    pydgm.control.xs_name = '../makeXS/{0}g/{0}gXS.anlxs'.format(G).ljust(256)
    pydgm.control.angle_order = 8
    pydgm.control.angle_option = pydgm.angle.gl
    pydgm.control.eigen_print = 1
    pydgm.control.outer_print = 0
    pydgm.control.inner_print = 0
    pydgm.control.eigen_tolerance = 1e-14
    pydgm.control.outer_tolerance = 1e-14
    pydgm.control.inner_tolerance = 1e-14
    pydgm.control.max_eigen_iters = 10000
    pydgm.control.max_outer_iters = 10
    pydgm.control.max_inner_iters = 1
    pydgm.control.store_psi = True
    pydgm.control.equation_type = 'DD'
    pydgm.control.legendre_order = 0
    pydgm.control.ignore_warnings = True


def setGeometry(geoOption):
    '''Construct the geometry for the snapshot problems'''
    pydgm.control.fine_mesh = [1]
    pydgm.control.coarse_mesh = [0.0, 1.0]
    if geoOption == 'uo2':
        pydgm.control.material_map = [1]
    elif geoOption == 'mox':
        pydgm.control.material_map = [3]


def barchart(x, y):
    X = np.zeros(2 * len(y))
    Y = np.zeros(2 * len(y))
    for i in range(0, len(y)):
        X[2 * i] = x[i]
        X[2 * i + 1] = x[i + 1]
        Y[2 * i] = y[i]
        Y[2 * i + 1] = y[i]
    return X, Y


def modGramSchmidt(A):
    m, n = A.shape
    A = A.copy()
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for k in range(n):
        R[k, k] = np.linalg.norm(A[:, k:k + 1].reshape(-1), 2)
        Q[:, k:k + 1] = A[:, k:k + 1] / R[k, k]
        R[k:k + 1, k + 1:n + 1] = np.dot(Q[:, k:k + 1].T, A[:, k + 1:n + 1])
        A[:, k + 1:n + 1] = A[:, k + 1:n + 1] - np.dot(Q[:, k:k + 1], R[k:k + 1, k + 1:n + 1])

    return Q, R


def plotBasis(G):
    basis = np.loadtxt('{0}g/mdlp_{0}g'.format(G))
    vectors = np.zeros((3, G))
    for g in range(G):
        b = np.trim_zeros(basis[g], trim='f')
        if len(b) >= 3:
            b = b[:3]
        else:
            bb = np.zeros(3)
            bb[:b.shape[0]] = b
            b = bb
        vectors[:, g] = b
    plot(vectors)


def plot(A):
    colors = ['b', 'g', 'm']
    plt.clf()
    G = A.shape[1]

    groupMask, counts = getGroupBounds(G)
    print bounds, diffs

    for i, a in enumerate(A):
        for CG in range(len(diffs)):
            if i < diffs[CG]:
                ming = bounds[CG]
                maxg = bounds[CG + 1]
                plt.plot(range(ming, maxg), a[ming:maxg], c=colors[i], label='order {}'.format(i))
    plt.vlines(bounds[1:-1] - 0.5, -1, 1)
    plt.xlim([0, G - 1])
    plt.ylim([-1, 1])
    plt.xlabel('Energy group')
    plt.ylabel('Normalized basis')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3)
    plt.savefig('plots/{}_mdlp.png'.format(G))
    return


def DLP(size):
    order = size
    A = np.ones((size, order))
    if order > 1:
        for j in range(size):
            A[j, 1] = (size - 1 - (2.0 * j)) / (size - 1)
        for i in range(2, order):
            for j in range(size):
                c0 = (i - 1) * (size - 1 + i)
                c1 = (2 * i - 1) * (size - 1 - 2 * j)
                c2 = i * (size - i)
                A[j, i] = (c1 * A[j, i - 1] - c0 * A[j, i - 2]) / c2
    return modGramSchmidt(A)[0]


def getGroupBounds(G):
    if G == 44:
        # Non-contiguous
        groupBounds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]
        # Contiguous
#         groupBounds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]
    elif G == 238:
        # Non-contiguous
        groupBounds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3,
                       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4, 2, 3, 4, 2, 4, 1, 2, 3,
                       2, 2, 3, 4, 1, 1, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 3,
                       1, 2, 1, 1, 1, 1, 2, 3, 4, 4, 3, 1, 1, 1, 1, 1, 2, 1, 3, 3, 1, 2,
                       3, 3, 3, 4, 6, 3, 3, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                       2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                       3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7]
        # Contiguous
#         groupBounds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#                        1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4,
#                        4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#                        6, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9,
#                        9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11,
#                        12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
#                        13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14,
#                        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
#                        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
#                        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
#                        14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 18]
    else:
        raise NotImplementedError, 'Group {} structure has not been created'.format(G)

    counts = dict(zip(*np.unique(groupBounds, return_counts=True)))

    return groupBounds, counts


def makeBasis(G, option='uo2'):
    # Get the coarse group bounds
    groupMask, counts = getGroupBounds(G)

    # Initialize the basis lists
    basis = np.zeros((G, G))

    # Get the reference spectrum
    spectrum = getSpectrum(G, option)

    # Compute the basis for each coarse group
    for group, order in counts.items():
        # Get the mask for the currect group
        m = groupMask == group

        # Get the standard DLP
        A = DLP(order)

        # Pointwise multiply the DLP with the spectrum
        for j in range(A.shape[1] - 1):
            A[:, j + 1] = A[:, j] * spectrum[m]

        # Reorthogonalize the basis
        A, r = np.linalg.qr(A, mode='complete')

        # Make the zeroth positive
        if np.sum(A[:, 0]) < 0:
            A *= -1

        # Slice into the basis with the current group
        basis[np.ix_(m, m)] = A

    directory = '{}g'.format(G)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the basis to file
    np.savetxt('{1}g/m{0}_{1}g'.format('dlp', G), basis)


if __name__ == '__main__':
    for G in [44, 238]:
        makeBasis(G)
        # plotBasis(G)
