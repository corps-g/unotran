from collections import OrderedDict
import sys
sys.path.append('~/workspace/unotran/src')
import pydgm
import numpy as np
import scipy as sp
from scipy.io import FortranFile
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif'})
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
    return phi[0,0]

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

def barchart(x, y) :
    X = np.zeros(2*len(y))
    Y = np.zeros(2*len(y))
    for i in range(0, len(y)) :
        X[2*i]   = x[i]
        X[2*i+1] = x[i+1]
        Y[2*i]   = y[i]
        Y[2*i+1] = y[i]
    return X, Y

def modGramSchmidt(A):
    m,n= A.shape
    A= A.copy()
    Q= np.zeros((m,n))
    R= np.zeros((n,n))

    for k in range(n):
        R[k,k]= np.linalg.norm(A[:,k:k+1].reshape(-1),2)
        Q[:,k:k+1]= A[:,k:k+1]/R[k,k]
        R[k:k+1,k+1:n+1]= np.dot( Q[:,k:k+1].T, A[:,k+1:n+1] )
        A[:,k+1:n+1]= A[:, k+1:n+1] - np.dot(Q[:,k:k+1], R[k:k+1,k+1:n+1])

    return Q, R

def plotBasis(G):
    basis = np.loadtxt('dlp/mdlp_{}g'.format(G))
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

    bounds, diffs = getGroupBounds(G)
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
    A = np.ones((size,order))
    if order > 1:
        for j in range(size):
            A[j, 1] = (size - 1 - (2.0 * j)) / (size - 1)
        for i in range(2, order):
            for j in range(size):
                c0 = (i - 1) * (size - 1 + i)
                c1 = (2 * i - 1) * (size - 1 - 2 * j)
                c2 = i * (size - i)
                A[j,i] = (c1 * A[j, i-1] - c0 * A[j, i-2]) / c2
    return modGramSchmidt(A)[0]

def getGroupBounds(G):
    if G == 2:
        groupBounds = [1]
    elif G == 3:
        groupBounds = [1, 3]
    elif G == 4:
        groupBounds = [1, 3]
    elif G == 7:
        groupBounds = [1, 5]
    elif G == 8:
        groupBounds = [1, 6]
    elif G == 9:
        groupBounds = [1, 7]
    elif G == 12:
        groupBounds = [1, 4, 10]
    elif G == 14:
        groupBounds = [1, 4, 12]
    elif G == 16:
        groupBounds = [1, 3, 14]
    elif G == 18:
        groupBounds = [1, 16]
    elif G == 23:
        groupBounds = [1, 8, 21]
    elif G == 25:
        groupBounds = [1, 8, 23]
    elif G == 30:
        groupBounds = [1, 19]
    elif G == 33:
        groupBounds = [1, 32]
    elif G == 40:
        groupBounds = [1, 23, 38]
    elif G == 43:
        groupBounds = [1, 26, 40]
    elif G == 44:
        groupBounds = [1, 15, 38, 43]
    elif G == 50:
        groupBounds = [1, 17]
    elif G == 69:
        groupBounds = [1, 10, 57, 67]
    elif G == 70:
        groupBounds = [1, 10, 58, 68]
    elif G == 100:
        groupBounds = [1, 58]
    elif G == 172:
        groupBounds = [1, 40, 100, 160, 169]
    elif G == 174:
        groupBounds = [1, 52, 112, 172]
    elif G == 175:
        groupBounds = [1, 53, 113, 173]
    elif G == 238:
        groupBounds = [1, 15, 75, 135, 138, 157, 217, 226, 233, 238]
    elif G == 240:
        groupBounds = [1, 60, 120, 180]
    elif G == 315:
        groupBounds = [1, 4, 64, 124, 184, 244, 304, 312, 315]
    elif G == 1968:
        groupBounds = [1, 11, 71, 131, 191, 251, 311, 371, 431, 491, 551, 611, 671, 731, 791, 851, 911, 971, 1031, 1091, 1151, 1211, 1271, 1331, 1391, 1393, 1448, 1451, 1465, 1467, 1518, 1520, 1527, 1587, 1591, 1594, 1654, 1659, 1666, 1726, 1786, 1797, 1836, 1896, 1956, 1965]
    else:
        groupBounds = [0, G]

    groupBounds.append(G)

    groupBounds = np.array(groupBounds)
    groupBounds[-1] += 1

    diffs = groupBounds[1:] - groupBounds[:-1]
    groupBounds -= 1

    return groupBounds, diffs

def makeBasis(G, option='uo2'):
    # Get the coarse group bounds
    groupBounds, diffs = getGroupBounds(G)

    # Initialize the basis lists
    P = []
    basis = np.array([])
    spectrum = getSpectrum(G, option)

    # Compute the basis for each coarse group
    for i, order in enumerate(diffs):
        A = DLP(order)
        for j in range(A.shape[1] - 1):
            A[:, j + 1] = A[:, j] * spectrum[groupBounds[i]: groupBounds[i + 1]]

        A, r = np.linalg.qr(A, mode='complete')

        if np.sum(A[:, 0]) < 0:
            A *= -1

        P.append(A)
        basis = A if i == 0 else sp.linalg.block_diag(basis, A)

    # Save the basis to file
    np.savetxt('{0}/m{0}_{1}g'.format('dlp', G), basis)


if __name__ == '__main__':
    for G in [2, 3, 4, 7, 8, 9, 12, 14, 16, 18, 23, 25, 30, 33, 40, 43, 44, 50, 69, 70, 100, 172, 174, 175, 238, 240, 315, 1968]:
        makeBasis(G)
        plotBasis(G)

