import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family':'serif'})
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['lines.linewidth'] = 1.85
rcParams['axes.labelsize'] = 20
rcParams.update({'figure.autolayout': True})

def getXS(fname):
    '''
    Read the input file and extract the total cross sections

    input
        fname: name of the file to read

    output
        sig_t: numpy array containing the total cross sections for each material
    '''

    # Initialize the total cross section list
    sig_t = []
    with open(fname) as f:
        # Get the number of materials and number of groups
        nMat, nGroups = [int(d) for d in f.readline().split()[:2]]
        # Skip 2 lines to pass the energy bounds and velocity
        f.readline()
        f.readline()

        # Loop over the materials
        for m in range(nMat):
            # Initialize temp list for total XS for material m
            t = []
            # Skip the material name
            f.readline()
            # Get the number of legendre moments
            nLegendre = int(f.readline().split()[0])
            # Loop over groups and extract the total XS
            for g in range(nGroups):
                t.append(float(f.readline().split()[0]))
            # Skip the scattering moments
            for g in range(nGroups * nLegendre):
                f.readline()
            # Append the temp XS list to the main list
            sig_t.append(t)

    return np.array(sig_t)

def computeBounds(fname):
    '''
    Determine the coarse group bounds following the guidelines in Gibson's paper

    inputs
        fname: name of the file containing cross sections in proteus format
    '''

    def reset(xs):
        '''Reset the min and max cutoffs to the input cross section'''
        minXS = max(xs, minCutoff)
        maxXS = xs
        return minXS, maxXS

    sig_t = getXS(fname)
    nMat = len(sig_t)
    nGroup = len(sig_t[0])
    minCutoff = 1.0
    ratioCutoff = 2.0
    groupCutoff = 60
    for m in range(nMat):

        bounds = [nGroup]
        # Initialize the cutoff bounds
        minXS, maxXS = reset(sig_t[m,-1])

        for i, xs in enumerate(sig_t[m,1:][::-1]):
            group = nGroup - i
            # Check if the xs in below the min or above the max
            minXS = min(xs, minXS)
            maxXS = max(xs, maxXS)
            ratio = maxXS / minXS
            # Check for a ratio that is too large
            if (ratio > ratioCutoff and maxXS > minCutoff) or bounds[-1] - group >= groupCutoff:
                bounds.append(group)
                # Reset the cutoff bounds
                minXS, maxXS = reset(xs)
        bounds.append(0)
        bounds = bounds[::-1]
        print bounds
        break
    B = [1] + bounds[1:-1] + [nGroup + 1]
    print ['{}-{} ({})'.format(b, B[i+1] - 1, B[i+1] - b) for i, b in enumerate(B[:-1])]

    plt.semilogy(range(1, nGroup+1), sig_t[0], 'bo')
    for b in bounds[1:-1]:
        plt.axvline(b - 0.5)
    plt.ylim([0,50])
    plt.xlim([0, nGroup])
    plt.xlabel('Fine group number')
    plt.ylabel('Total cross section [cm$^{-1}$]')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    for G in [44, 238, 1968]:
        fname = 'makeXS/{0}g/{0}gXS.anlxs'.format(G)
        computeBounds(fname)
