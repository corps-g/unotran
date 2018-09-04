from collections import OrderedDict
import sys
sys.path.append('~/workspace/unotran/src')
import os
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


def getEnergy(G):
    with open('../makeXS/{0}g/{0}gXS.anlxs'.format(G), 'r') as f:
        f.readline()
        s = f.readline()
    return np.array(s.split()).astype(float) * 1e6


def makePlot(G):
    E = getEnergy(G)
    meanE = 0.5 * (E[1:] + E[:-1])
    diffE = E[:-1] - E[1:]
    snapshots = getSnapshots(G, 'full')
    snapMean = np.mean(snapshots, axis=1) * meanE / diffE
    norm = np.linalg.norm(snapMean)
    snapMean /= norm

    EE, minsnap = barchart(E, np.min(snapshots, axis=1) / norm / diffE * meanE)
    EE, maxsnap = barchart(E, np.max(snapshots, axis=1) / norm / diffE * meanE)

    full = KLT(snapshots)[:, 0] * meanE / diffE
    uo2 = KLT(getSnapshots(G, 'uo2'))[:, 0] * meanE / diffE
    mox = KLT(getSnapshots(G, 'mox'))[:, 0] * meanE / diffE
    com = KLT(np.concatenate((getSnapshots(G, 'uo2'), getSnapshots(G, 'mox'), getSnapshots(G, 'junction')), axis=1))[:, 0] * meanE / diffE

    norm = np.mean(snapMean / full)

    plt.plot(*barchart(E, full * norm), label='full', c='b', ls='-')
    plt.plot(*barchart(E, uo2 * norm), label='UO$_2$', c='g', ls='-')
    plt.plot(*barchart(E, mox * norm), label='MOX', c='r', ls='-')
    plt.plot(*barchart(E, com * norm), label='combine', c='k', ls='-')
    plt.fill_between(EE, minsnap, maxsnap, alpha=0.5)
    plt.xlim([min(E), max(E)])
    plt.ylim([1e-5, 1e0])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('energy [eV]')
    plt.ylabel('flux per unit lethargy')
    plt.legend(loc=0)
    plt.savefig('plots/{}_snapshots.png'.format(G))
    plt.clf()


def getSnapshots(G, geoOption):
    try:
        phi = np.load('phi_g{}_{}.npy'.format(G, geoOption))
    except IOError:
        setVariables(G, geoOption)
        pydgm.solver.initialize_solver()
        pydgm.solver.solve()
        phi = np.copy(pydgm.state.phi)
        pydgm.solver.finalize_solver()
        np.save('phi_g{}_{}'.format(G, geoOption), phi)
        # plot(phi[0])
    return phi[0].T


def setVariables(G, geoOption):
    # Set the variables for the discrete ordinance problem

    pydgm.control.boundary_type = [1.0, 1.0]  # reflective boundarys
    setGeometry(geoOption)
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
    pydgm.control.max_outer_iters = 100
    pydgm.control.max_inner_iters = 1
    pydgm.control.store_psi = True
    pydgm.control.equation_type = 'DD'
    pydgm.control.legendre_order = 0
    pydgm.control.ignore_warnings = True


def setGeometry(geoOption):
    '''Construct the geometry for the snapshot problems'''
    if geoOption == 'uo2':
        pydgm.control.fine_mesh = [3, 22, 3]
        pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [8, 1, 8]
    elif geoOption == 'mox':
        pydgm.control.fine_mesh = [3, 22, 3]
        pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [8, 3, 8]
    elif geoOption == 'assay1':
        pydgm.control.fine_mesh = [6, 18, 18, 18, 18, 6]
        pydgm.control.coarse_mesh = [0.0, 1.1176, 4.3688, 7.62, 10.8712, 14.1224, 15.24]
        pydgm.control.material_map = [8, 5, 6, 6, 5, 8]
    elif geoOption == 'assay2':
        pydgm.control.fine_mesh = [6, 18, 18, 18, 18, 6]
        pydgm.control.coarse_mesh = [0.0, 1.1176, 4.3688, 7.62, 10.8712, 14.1224, 15.24]
        pydgm.control.material_map = [8, 5, 5, 5, 5, 8]
    elif geoOption == 'assay3':
        pydgm.control.fine_mesh = [6, 18, 18, 18, 18, 6]
        pydgm.control.coarse_mesh = [0.0, 1.1176, 4.3688, 7.62, 10.8712, 14.1224, 15.24]
        pydgm.control.material_map = [8, 5, 7, 7, 5, 8]
    elif geoOption == 'assay4':
        pydgm.control.fine_mesh = [6, 18, 18, 18, 18, 6]
        pydgm.control.coarse_mesh = [0.0, 1.1176, 4.3688, 7.62, 10.8712, 14.1224, 15.24]
        pydgm.control.material_map = [8, 7, 7, 7, 7, 8]
    elif geoOption == 'core1':
        ass1 = [8, 5, 6, 6, 5, 8]
        ass2 = [8, 5, 5, 5, 5, 8]
        fm_ass = [6, 18, 18, 18, 18, 6]
        cm_ass = [1.1176, 3.2512, 3.2512, 3.2512, 3.2512, 1.1176]
        ass_map = [1, 2, 1, 2, 1, 2, 1]
        cm = [0.0]
        fm = []
        mm = []
        for i, ass in enumerate(ass_map):
            mm += ass1 if ass == 1 else ass2
            cm += cm_ass
            fm += fm_ass
        cm = np.cumsum(cm)

        pydgm.control.fine_mesh = fm
        pydgm.control.coarse_mesh = cm
        pydgm.control.material_map = mm
        pydgm.control.boundary_type = [0.0, 0.0]  # vacuum boundaries
    elif geoOption == 'core2':
        ass1 = [8, 5, 6, 6, 5, 8]
        ass2 = [8, 5, 7, 7, 5, 8]
        fm_ass = [6, 18, 18, 18, 18, 6]
        cm_ass = [1.1176, 3.2512, 3.2512, 3.2512, 3.2512, 1.1176]
        ass_map = [1, 2, 1, 2, 1, 2, 1]
        cm = [0.0]
        fm = []
        mm = []
        for i, ass in enumerate(ass_map):
            mm += ass1 if ass == 1 else ass2
            cm += cm_ass
            fm += fm_ass
        cm = np.cumsum(cm)

        pydgm.control.fine_mesh = fm
        pydgm.control.coarse_mesh = cm
        pydgm.control.material_map = mm
        pydgm.control.boundary_type = [0.0, 0.0]  # vacuum boundaries
    elif geoOption == 'core3':
        ass1 = [8, 5, 6, 6, 5, 8]
        ass2 = [8, 7, 7, 7, 7, 8]
        fm_ass = [6, 18, 18, 18, 18, 6]
        cm_ass = [1.1176, 3.2512, 3.2512, 3.2512, 3.2512, 1.1176]
        ass_map = [1, 2, 1, 2, 1, 2, 1]
        cm = [0.0]
        fm = []
        mm = []
        for i, ass in enumerate(ass_map):
            mm += ass1 if ass == 1 else ass2
            cm += cm_ass
            fm += fm_ass
        cm = np.cumsum(cm)

        pydgm.control.fine_mesh = fm
        pydgm.control.coarse_mesh = cm
        pydgm.control.material_map = mm
        pydgm.control.boundary_type = [0.0, 0.0]  # vacuum boundaries
    elif geoOption == 'junction':
        pydgm.control.fine_mesh = [3, 22, 3, 3, 22, 3]
        pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26, 1.35, 2.43, 2.52]
        pydgm.control.material_map = [5, 1, 5, 5, 3, 5]
    elif geoOption == 'full':
        # Define the number of pins of UO2
        nPins = 5
        # Define the fine mesh
        pydgm.control.fine_mesh = [3, 22, 3] * nPins * 2
        # Define the coarse mesh
        x = [0.0, 0.09, 1.17, 1.26]
        cm = [xx + i * x[-1] for i in range(10) for xx in x[:-1]] + [2 * nPins * 1.26]
        pydgm.control.coarse_mesh = cm
        # Define the material map
        mMap = [5, 1, 5] * nPins + [5, 3, 5] * nPins
        pydgm.control.material_map = mMap


def barchart(x, y):
    X = np.zeros(2 * len(y))
    Y = np.zeros(2 * len(y))
    for i in range(len(y)):
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


def plotBasis(G, basisType):
    basis = np.loadtxt('{1}g/klt_{0}_{1}g'.format(basisType, G))
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
    plot(vectors, basisType)


def plot(A, basisType):
    colors = ['b', 'g', 'm']
    plt.clf()
    G = A.shape[1]

    groupMask, counts = getGroupBounds(G)
    bounds -= 1
    print bounds

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
    plt.savefig('plots/{}_{}.png'.format(G, basisType))
    return


def KLT(Data, flat=False):
    A = Data.T.dot(Data)
    # Perform the SVD
    w, A = np.linalg.eig(A)
    idx = w.argsort()[::-1]  # sorts eigenvalues and vectors max->min
    w = w[idx]
    A = A[:, idx]
    A = np.real(Data.dot(A))
    # Orthogonalize the matrix
    A, r = np.linalg.qr(A, mode='complete')

    if flat:
        # ignore the smallest eigenpair to make room for the flat zeroth
        A = np.insert(A.T[:-1], 0, np.ones(len(A[0])) / np.sqrt(len(A[0])), axis=0)  # Add a flat zeroth

        # Redo the SVD if the zeroth was changed
        A, r = np.linalg.qr(A.T, mode='complete')

    if np.sum(A[:, 0]) < 0:
        A *= -1

    # Make sure that the basis is orthogonal
    np.testing.assert_array_almost_equal(A.T.dot(A), np.eye(len(A)), 12)

    return A


def getGroupBounds(G):
    if G == 44:
        groupBounds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
    elif G == 238:
        groupBounds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 3, 4, 3, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 2, 3, 3, 3, 3, 3, 5, 3, 4, 2, 3, 4, 2, 4, 1, 2, 4, 2, 2, 3, 5, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 3, 1, 2, 1, 1, 1, 1, 2, 3, 4, 5, 3, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 2, 3, 3, 4, 5, 7, 4, 3, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7]
    else:
        raise NotImplementedError, 'Group {} structure has not been created'.format(G)

    counts = dict(zip(*np.unique(groupBounds, return_counts=True)))

    return groupBounds, counts


def makeBasis(G, basisType, flat=False):
    if basisType == 'full':
        name = 'klt_full'
        data = getSnapshots(G, 'full')
    elif basisType == 'mox':
        name = 'klt_mox'
        data = getSnapshots(G, 'mox')
    elif basisType == 'uo2':
        name = 'klt_uo2'
        data = getSnapshots(G, 'uo2')
    elif basisType == 'combine':
        name = 'klt_combine'
        data = np.concatenate((getSnapshots(G, 'uo2'), getSnapshots(G, 'mox'), getSnapshots(G, 'junction')), axis=1)
    elif basisType == 'core1':
        name = 'klt_core1'
        data = getSnapshots(G, 'core1')
    elif basisType == 'core2':
        name = 'klt_core2'
        data = getSnapshots(G, 'core2')
    elif basisType == 'core3':
        name = 'klt_core3'
        data = getSnapshots(G, 'core3')
    elif basisType == 'assay_12':
        name = 'klt_assay_12'
        data = np.concatenate((getSnapshots(G, 'assay1'), getSnapshots(G, 'assay2')), axis=1)
    elif basisType == 'assay_13':
        name = 'klt_assay_13'
        data = np.concatenate((getSnapshots(G, 'assay1'), getSnapshots(G, 'assay3')), axis=1)
    elif basisType == 'assay_14':
        name = 'klt_assay_14'
        data = np.concatenate((getSnapshots(G, 'assay1'), getSnapshots(G, 'assay4')), axis=1)
    elif basisType == 'assay_all':
        name = 'klt_assay_all'
        data = np.concatenate((getSnapshots(G, 'assay1'), getSnapshots(G, 'assay2'), getSnapshots(G, 'assay3'), getSnapshots(G, 'assay4')), axis=1)
    else:
        raise NotImplementedError('The type: {} has not been implemented'.format(basisType))

    # Get the coarse group bounds
    groupMask, counts = getGroupBounds(G)

    # Initialize the basis lists
    basis = np.zeros((G, G))

    # Compute the basis for each coarse group
    for group, order in counts.items():
        # Get the mask for the currect group
        m = groupMask == group

        # Get the DLP basis for the given order
        A = KLT(data[m], flat)

        # Slice into the basis with the current group
        basis[np.ix_(m, m)] = A

    plt.plot(basis[:, :2])

    directory = '{}g'.format(G)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the basis to file
    np.savetxt('{1}g/{0}_{1}g'.format(name, G), basis)
    #np.savetxt('{0}/{1}_{2}g'.format(name, name if flat else 'f_' + name, G), basis)


if __name__ == '__main__':
    task = -1
    Gs = [44, 238]
    for G in Gs:
        #makePlot(G)
        for basisType in ['core1', 'core2', 'core3', 'assay_12', 'assay_13', 'assay_14', 'assay_all', 'full', 'mox', 'uo2', 'combine']:
            task += 1
            makeBasis(G, basisType, flat=True)
            #plotBasis(G, basisType)
