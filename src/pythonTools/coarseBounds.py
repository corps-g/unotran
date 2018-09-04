import numpy as np
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
        minXS, maxXS = reset(sig_t[m, -1])
        for i, xs in enumerate(sig_t[m, 1:][::-1]):
            group = nGroup - i
            # Check if the xs in below the min or above the max
            minXS = min(xs, minXS)
            maxXS = max(xs, maxXS)
            ratio = maxXS / minXS
            # Check for a ratio that is too large
            if (ratio > ratioCutoff and maxXS > minCutoff) or bounds[-1] - group > groupCutoff:
                bounds.append(group + 1)
                # Reset the cutoff bounds
                minXS, maxXS = reset(xs)
        bounds.append(1)
        bounds = bounds[::-1]
        print nGroup, bounds[:-1]
        break
    B = [1] + bounds[1:-1] + [nGroup + 1]
    print ['{}-{} &({})'.format(b, B[i + 1] - 1, B[i + 1] - b) for i, b in enumerate(B[:-1])]

    plt.semilogy(range(1, nGroup + 1), sig_t[0], 'bo')
    for b in bounds[1:-1]:
        plt.axvline(b - 0.5)
    plt.ylim([0, 50])
    plt.xlim([0, nGroup])
    plt.xlabel('Fine group number')
    plt.ylabel('Total cross section [cm$^{-1}$]')
    plt.grid(True)
    plt.show()


def findBounds(sig_t):
    '''
    Find the coarse group structure given a set of cross sections

    Inputs:
        sig_t: array of the total XS for one material
    Outputs:
        bounds: Starting fine group indices for each coarse group
    '''

    def reset(xs):
        '''Reset the min and max cutoffs to the input cross section'''
        minXS = max(xs, minCutoff)
        maxXS = xs
        return minXS, maxXS

    # Get the number of groups
    nGroup = len(sig_t)

    # Define some group selecting parameters
    minCutoff = 1.0  # XS below which the ratio cutoff is ignored
    ratioCutoff = 2.0  # Ratio of largest to smallest XS in a coarse group
    groupCutoff = 60  # Maximum number of fine groups in a coarse group

    # Initialize the boundary at the lowest energy group
    bounds = [nGroup]

    # Initialize the cutoff bounds
    minXS, maxXS = reset(sig_t[-1])

    # Loop through the cross sections from greatest to least
    for i, xs in enumerate(sig_t[1:][::-1]):
        group = nGroup - i

        # Check if the xs in below the min or above the max
        minXS = min(xs, minXS)
        maxXS = max(xs, maxXS)
        ratio = maxXS / minXS

        # Check for a ratio that is too large
        if (ratio > ratioCutoff and maxXS > minCutoff) or bounds[-1] - group > groupCutoff:
            bounds.append(group + 1)
            # Reset the cutoff bounds
            minXS, maxXS = reset(xs)

    # Add the highest energy group bound
    bounds.append(1)
    # Reverse to the natural order of structures
    bounds = bounds[::-1]
    return bounds[:-1]


def nonContigBounds(fname):
    '''
    This function determins which fine groups should belong to each coarse group

    The boundaries will not necessarily be contiguous

    Inputs:
        fname: name of the cross section file in anlxs format
    '''

    # Get the XS for only material 0
    # TODO expand this to multiple materials
    sig_t = getXS(fname)[:1]

    # Number the energy groups
    groups = np.arange(len(sig_t[0]))

    # Get the minimum XS across all materials
    minXS = np.min(sig_t, axis=0)
    # Get the maximum XS across all materials
    maxXS = np.max(sig_t, axis=0)

    # Sort the maximum XS and get the ordering indicies
    mask = np.argsort(maxXS)
    # Use this to not sort the cross sections
    # mask = np.arange(len(sig_t[0]))

    # Get the coarse group bounds given the total cross sections
    bounds = np.array(findBounds(maxXS[mask]) + [len(sig_t[0]) + 1]) - 1

    sub_mask = np.zeros((len(bounds) - 1, len(sig_t[0]))).astype(bool)
    structure = np.zeros(len(sig_t[0])).astype(int)
    for i, b in enumerate(bounds[:-1]):
        sub_mask[i, sorted(mask[b:bounds[i + 1]])] = 1
        structure[mask[b:bounds[i + 1]]] = i
    print bounds
    print repr(structure)

    # Plot the cross sections
    #plt.semilogy(groups, minXS[sort_mask], 'b-', label='Min')
    for i, m in enumerate(sub_mask):
        plt.semilogy(groups[m], maxXS[m], label='Group {}'.format(i), ls='none', marker='o')
    # for b in bounds[1:-1]:
    #    plt.axvline(b - 0.5)
    #plt.ylim([1e-1, 1e1])
    #plt.xlim([0, len(sig_t[0]) - 1])
    plt.xlabel('Fine group number')
    plt.ylabel('Total cross section [cm$^{-1}$]')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    Gs = [2, 3, 4, 7, 8, 9, 12, 14, 16, 18, 23, 25, 30, 33, 40, 43, 44, 50, 69, 70, 100, 172, 174, 175, 238, 240, 315, 1968]
    Gs = [44, 238]
    for G in Gs:
        fname = 'makeXS/{0}g/{0}gXS.anlxs'.format(G)
        nonContigBounds(fname)
        # computeBounds(fname)
