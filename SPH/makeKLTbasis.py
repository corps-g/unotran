from coarseBounds import computeBounds, Grouping
from collections import OrderedDict
import sys
import os
sys.path.append('/homes/rlreed/workspace/unotran/src')
import pydgm
import numpy as np
from scipy.linalg import eigh
import matplotlib as mpl
mpl.use('Agg')
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
import pickle


def getEnergy(G):
    with open('XS/{0}gXS.anlxs'.format(G), 'r') as f:
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

    full = np.mean(snapshots, axis=1) * meanE / diffE
    uo2 = np.mean(getSnapshots(G, 'uo2-1'), axis=1) * meanE / diffE
    mox = np.mean(getSnapshots(G, 'mox'), axis=1) * meanE / diffE
    com = np.mean(np.concatenate((getSnapshots(G, 'uo2-1'), getSnapshots(G, 'mox'), getSnapshots(G, 'junction')), axis=1), axis=1) * meanE / diffE
    pin = np.mean(np.concatenate((getSnapshots(G, 'uo2-1'), getSnapshots(G, 'mox')), axis=1), axis=1) * meanE / diffE

    norm = np.mean(snapMean / full)
    plt.clf()

    plt.plot(*barchart(E, full * norm), label='full', c='r', ls='-')
    plt.plot(*barchart(E, uo2 * norm), label='UO$_2$', c='m', ls='-')
    plt.plot(*barchart(E, mox * norm), label='MOX', c='c', ls='-')
    plt.plot(*barchart(E, com * norm), label='combine', c='b', ls='-')
    plt.plot(*barchart(E, com * norm), label='pin', c='g', ls='--')
    plt.fill_between(EE, minsnap, maxsnap, alpha=0.5)
    plt.xlim([min(E), max(E)])
    plt.ylim([1e-5, 1e0])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy [eV]')
    plt.ylabel('Flux per unit lethargy')
    plt.legend(loc=0, fancybox=True, framealpha=0.0)
    plt.savefig('plots/{}_snapshots.png'.format(G), transparent=True)
    plt.clf()


def getSnapshots(G, geoOption, ref_path):
    return np.load('{}/ref_phi_{}_{}.npy'.format(ref_path, geoOption, G))[0]


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


def plotBasis(bName, info):
    # load the basis from file
    basis = np.loadtxt(bName)
    G = info.G

    E = getEnergy(G)

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
    plot(vectors, bName, info, E)

def barchart(x, y):
    X = np.zeros(2 * len(y))
    Y = np.zeros(2 * len(y))
    for i in range(0, len(y)):
        X[2 * i] = x[i]
        X[2 * i + 1] = x[i + 1]
        Y[2 * i] = y[i]
        Y[2 * i + 1] = y[i]
    return X, Y

def plot(A, bName, info, E):
    colors = ['b', 'g', 'm']
    plt.clf()
    G = A.shape[1]

    for i, a in enumerate(A):
        bounds = [E[0]]
        ming = 0
        for CG, nGroups in info.counts.items():
            maxg = ming + nGroups
            x, y = barchart(E[ming:maxg+1], a[ming:maxg])
            plt.semilogx(x, y, c=colors[i], label='order {}'.format(i))
            bounds.append(E[maxg])
            ming += nGroups
    bounds = np.array(bounds)
    print(bounds)
    plt.vlines(bounds[1:-1], -1, 1, alpha=0.8)
    plt.ylim([-1, 1])
    plt.xlim([E[-1], E[0]])
    plt.xlabel('Energy [eV]')
    plt.ylabel('Normalized basis')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3, fancybox=True, framealpha=0.0)
    plt.savefig('{}.png'.format(bName), transparent=True)
    return


def KLT(Data):
    N = len(Data)
    A = Data.T.dot(Data)
    # Perform the SVD
    w, A = eigh(A)
    idx = w.argsort()[::-1]  # sorts eigenvalues and vectors max->min
    w = w[idx]
    A = A[:, idx]
    A = np.real(Data.dot(A))

    # Orthogonalize the matrix
    A, r = np.linalg.qr(A, mode='complete')

    # ignore the smallest eigenpair to make room for the flat zeroth
    A = np.insert(A.T[:-1], 0, np.ones(len(A[0])) / np.sqrt(len(A[0])), axis=0)  # Add a flat zeroth

    # Redo the SVD if the zeroth was changed
    A, r = np.linalg.qr(A.T, mode='complete')

    if np.sum(A[:, 0]) < 0:
        A *= -1

    # Make sure that the basis is orthogonal
    np.testing.assert_array_almost_equal(A.T.dot(A), np.eye(len(A)), 12)

    return A


def getSnapshotData(structure, basisType, ref_path):

    G = structure.G
    if '-sig' in basisType:
        sig_flag = True
        basisType = basisType.replace('-sig', '')
    else:
        sig_flag = False
    if basisType == 'klt_full':
        data = getSnapshots(G, 'full', ref_path)
    elif basisType == 'klt_mox':
        data = getSnapshots(G, 'mox', ref_path)
    elif basisType == 'klt_uo2':
        data = getSnapshots(G, 'uo2-1', ref_path)
    elif basisType == 'klt_pins_full':
        data = np.concatenate((getSnapshots(G, 'uo2-1', ref_path), getSnapshots(G, 'mox', ref_path)), axis=1)
    elif basisType == 'klt_pins_core1':
        data = np.concatenate((getSnapshots(G, 'uo2-1', ref_path), getSnapshots(G, 'uo2-2', ref_path), getSnapshots(G, 'uo2-Gd', ref_path)), axis=1)
    elif basisType == 'klt_pins_core2':
        data = np.concatenate((getSnapshots(G, 'uo2-1', ref_path), getSnapshots(G, 'uo2-2', ref_path), getSnapshots(G, 'mox', ref_path)), axis=1)
    elif basisType == 'klt_combine':
        data = np.concatenate((getSnapshots(G, 'uo2-1', ref_path), getSnapshots(G, 'mox', ref_path), getSnapshots(G, 'junction', ref_path)), axis=1)
    elif basisType == 'klt_core1':
        data = getSnapshots(G, 'core1', ref_path)
    elif basisType == 'klt_core2':
        data = getSnapshots(G, 'core2', ref_path)
    elif basisType == 'klt_assay_12':
        data = np.concatenate((getSnapshots(G, 'assay1', ref_path), getSnapshots(G, 'assay2', ref_path)), axis=1)
    elif basisType == 'klt_assay_13':
        data = np.concatenate((getSnapshots(G, 'assay1', ref_path), getSnapshots(G, 'assay3', ref_path)), axis=1)
    elif basisType == 'klt_assay_all':
        data = np.concatenate((getSnapshots(G, 'assay1', ref_path), getSnapshots(G, 'assay2', ref_path), getSnapshots(G, 'assay3', ref_path)), axis=1)
    elif basisType == 'klt_c5g7-uo2-pin':
        data = getSnapshots(G, 'c5g7-uo2-pin', ref_path)
    elif basisType == 'klt_c5g7-mox-pin':
        data = getSnapshots(G, 'c5g7-mox-pin', ref_path)
    elif basisType == 'klt_c5g7_pins':
        data = np.concatenate((getSnapshots(G, 'c5g7-uo2-pin', ref_path), getSnapshots(G, 'c5g7-mox-pin', ref_path)), axis=1)
    elif basisType == 'klt_c5g7-uo2-assay':
        data = getSnapshots(G, 'c5g7-uo2-assay-small', ref_path)
    elif basisType == 'klt_c5g7-mox-assay':
        data = getSnapshots(G, 'c5g7-mox-assay-small', ref_path)
    elif basisType == 'klt_c5g7_assays':
        data = np.concatenate((getSnapshots(G, 'c5g7-uo2-assay-small', ref_path), getSnapshots(G, 'c5g7-mox-assay-small', ref_path)), axis=1)
        nSnaps = len(data[0])
        idx = np.random.randint(nSnaps, size=nSnaps//100)
        data = data[:,idx]
    elif basisType == 'klt_c5g7_full':
        data = getSnapshots(G, 'c5g7-small', ref_path)
        nSnaps = len(data[0])
        idx = np.random.randint(nSnaps, size=nSnaps//100)
        data = data[:,idx]
    elif basisType == 'klt_inf':
        data = getSnapshots(G, 'inf', ref_path)
    else:
        raise NotImplementedError('The type: {} has not been implemented'.format(basisType))

    if sig_flag:
        data = np.concatenate((data, structure.sig_t.T), axis=1)

    return data


def makeBasis(basisType, structure, ref_path='reference'):

    bName = 'basis/{}_{}.basis'.format(basisType, structure.fname)
    print('building ' + bName)

    data = getSnapshotData(structure, basisType, ref_path)

    print('{} snapshots available'.format(data.shape[1]))

    G = structure.G
    groupMask = structure.structure

    # Initialize the basis lists
    basis = np.zeros((G, G))

    # Compute the basis for each coarse group
    for group, order in structure.counts.items():
        # Get the mask for the currect group
        m = groupMask == group

        # Get the DLP basis for the given order
        A = KLT(data[m])

        # Slice into the basis with the current group
        basis[np.ix_(m, m)] = A

    np.testing.assert_array_almost_equal(basis.T.dot(basis), np.eye(len(basis)), 12)

    # Save the basis to file
    if not os.path.exists(bName):
        np.savetxt(bName, basis)

    # plotBasis(bName, structure)


def getInfo(task, kill=False):
    Gs = [44, 238, 1968]
    geos = ['full', 'core1', 'core2', 'c5g7-uo2-pin-small', 'c5g7-mox-low-pin', 'c5g7-mox-mid-pin', 'c5g7-mox-high-pin', 'c5g7-mox-assay', 'c5g7-uo2-assay', 'c5g7']
    contigs = [1, 2, 3]
    mins = [1.5]
    ratios = [1.6]
    groups = [60]

    item = 0
    groupBounds = []
    for G in Gs:
        for geo in geos:
            # Select the basis based on the geometry
            if geo == 'inf_med':
                basisTypes = ['klt_uo2']
            elif geo == 'full':
                basisTypes = ['klt_full', 'klt_combine', 'klt_uo2', 'klt_mox', 'klt_pins_full']
            elif geo == 'core1':
                basisTypes = ['klt_assay_12', 'klt_assay_all', 'klt_core1', 'klt_pins_core1']
            elif geo == 'core2':
                basisTypes = ['klt_assay_13', 'klt_assay_all', 'klt_core2', 'klt_pins_core2']
            elif geo == 'c5g7':
                basisTypes = ['klt_c5g7_pins', 'klt_c5g7_assays', 'klt_c5g7_full']
            elif geo == 'c5g7-uo2-pin':
                basisTypes = ['klt_c5g7-uo2-pin']
            elif geo == 'c5g7-mox-low-pin':
                basisTypes = ['klt_c5g7-mox-low-pin']
            elif geo == 'c5g7-mox-mid-pin':
                basisTypes = ['klt_c5g7-mox-mid-pin']
            elif geo == 'c5g7-mox-high-pin':
                basisTypes = ['klt_c5g7-mox-high-pin']
            elif geo == 'c5g7-uo2-assay':
                basisTypes = ['klt_c5g7-uo2-assay']
            elif geo == 'c5g7-mox-assay':
                basisTypes = ['klt_c5g7-mox-assay']

            for basisType in basisTypes:
                basisOptions = [''] if 'klt' in basisType else ['']
                for basisOption in basisOptions:
                    for contig in contigs:
                        for min_cutoff in mins:
                            for ratio_cutoff in ratios:
                                for group_cutoff in groups:
                                    if item == task and not kill:
                                        return G, geo, basisType + basisOption, contig, min_cutoff, ratio_cutoff, group_cutoff
                                    else:
                                        item += 1
        groupBounds.append(item)
    if kill:
        groupBounds = [0] + groupBounds
        for i, G in enumerate(Gs):
            print('Group {:4d} cutoffs: {:>5d}-{:<5d}'.format(G, groupBounds[i] + 1, groupBounds[i+1]))
    exit()

if __name__ == '__main__':
    # getInfo(1, True)

    task = os.environ['SLURM_ARRAY_TASK_ID']
    task = int(task) - 1

    # Get the parameters for the task number
    G, geo, basisType, contig, min_cutoff, ratio_cutoff, group_cutoff = getInfo(task)

    # Build the coarse-group structure
    structure = pickle.load(open('XS/structure{}_{}_{}g_m{}_r{}_g{}.p'.format(contig, geo, G, min_cutoff, ratio_cutoff, group_cutoff), 'rb'))

    makeBasis(basisType, structure)

    # makePlot(238)

    print('complete')

