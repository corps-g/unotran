from coarseBounds import computeBounds, Grouping
from collections import OrderedDict
import os
import sys
sys.path.append('/homes/rlreed/workspace/unotran/src')
import pydgm
import numpy as np
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


def plotBasis(basis, name, info):
    G = info.G
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
    plot(vectors, name, info)


def plot(A, name, info):
    colors = ['b', 'g', 'm']
    plt.clf()
    G = A.shape[1]

    bounds = [0]
    for i, a in enumerate(A):
        ming = 0
        for CG, nGroups in info.counts.items():
            maxg = ming + nGroups
            plt.plot(range(ming, maxg)[::-1], a[ming:maxg], c=colors[i], label='order {}'.format(i))
            bounds.append(maxg)
            ming += nGroups

    bounds = np.array(bounds)[::-1]
    plt.vlines(bounds[1:-1] - 0.5, -1, 1)
    plt.xlim([0, G - 1])
    plt.ylim([-1, 1])
    plt.xlabel('Energy group')
    plt.ylabel('Normalized basis')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3)
    plt.savefig('{}.png'.format(name))
    plt.clf()
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

def testBasis(basis):
    np.testing.assert_array_almost_equal(basis.T.dot(basis), np.eye(len(basis)), 12)

def makeBasis(structure):
    '''
    Create the DLP basis given the structure provided (coarseBounds.Structure type)
    '''

    # Get the coarse group bounds
    G = structure.G
    counts = structure.counts
    groupMask = structure.structure
    bname = 'dlp_{}'.format(structure.fname)
    print('building {}'.format(bname))

    # Initialize the basis lists
    basis = np.zeros((G, G))

    # Compute the basis for each coarse group
    for group, order in counts.items():
        # Get the DLP basis for the given order
        A = DLP(order)

        # Get the mask for the currect group
        m = groupMask == group

        # Slice into the basis with the current group
        basis[np.ix_(m, m)] = A

    testBasis(basis)

    # Save the basis to file
    if not os.path.exists('basis/{}.basis'.format(bname)):
        np.savetxt('basis/{}.basis'.format(bname), basis)

    # plotBasis(basis, 'basis/{}'.format(bname), structure)


def getInfo(task, kill=False):
    Gs = [44, 238, 1968]
    geos = ['full', 'core1', 'core2', 'c5g7-uo2-pin', 'c5g7-mox-low-pin', 'c5g7-mox-mid-pin', 'c5g7-mox-high-pin', 'c5g7-mox-assay', 'c5g7-uo2-assay', 'c5g7']
    contigs = [1, 2, 3]
    mins = [1.5]
    ratios = [2.4]
    groups = [60]

    item = 0
    groupBounds = []
    for G in Gs:
        for geo in geos:
            for contig in contigs:
                for min_cutoff in mins:
                    for ratio_cutoff in ratios:
                        for group_cutoff in groups:
                            if item == task and not kill:
                                return G, geo, contig, min_cutoff, ratio_cutoff, group_cutoff
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
    G, geo, contig, min_cutoff, ratio_cutoff, group_cutoff = getInfo(task)

    # Build the coarse-group structure
    structure = pickle.load(open('XS/structure{}_{}_{}g_m{}_r{}_g{}.p'.format(contig, geo, G, min_cutoff, ratio_cutoff, group_cutoff), 'rb'))

    makeBasis(structure)

    print('complete')

