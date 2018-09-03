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
    basis = np.loadtxt('{0}g/dlp_{0}g'.format(G))
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
    bounds -= 1
    bounds = np.concatenate(([0], bounds))

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
    plt.savefig('plots/{}_dlp.png'.format(G))
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
    if G == 44:
        groupBounds = [1, 15, 38, 43]
    elif G == 238:
        groupBounds = [1, 18, 78, 79, 80, 82, 85, 86, 92, 93, 102, 108, 110, 119, 121, 135, 136, 137, 138, 139, 161, 221, 227, 233, 238]
    elif G == 1968:
        groupBounds = [1, 11, 71, 131, 191, 251, 311, 371, 431, 491, 551, 611, 671, 731, 791, 851, 911, 971, 1031, 1091, 1151, 1211, 1271, 1331, 1391, 1393, 1448, 1451, 1465, 1467, 1518, 1520, 1527, 1587, 1591, 1594, 1654, 1659, 1666, 1726, 1786, 1797, 1836, 1896, 1956, 1965]
    else:
        groupBounds = [0, G]

    groupBounds.append(G)

    groupBounds = np.array(groupBounds)
    groupBounds[-1] += 1

    diffs = groupBounds[1:] - groupBounds[:-1]
    groupBounds = groupBounds[1:]

    return groupBounds, diffs

def makeBasis(G):
    # Get the coarse group bounds
    groupBounds, diffs = getGroupBounds(G)

    # Initialize the basis lists
    P = []
    basis = np.array([])

    # Compute the basis for each coarse group
    for i, order in enumerate(diffs):
        A = DLP(order)

        P.append(A)
        basis = A if i == 0 else sp.linalg.block_diag(basis, A)

    directory = '{}g'.format(G)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the basis to file
    np.savetxt('{1}g/{0}_{1}g'.format('dlp', G), basis)


if __name__ == '__main__':
    for G in [44, 238]:
        makeBasis(G)
        plotBasis(G)

