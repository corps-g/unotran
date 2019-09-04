import os
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
np.set_printoptions(threshold=10000)
import pickle


class Grouping(object):
    def __init__(self, G, geoOption, contig, name, bounds, minCutoff, ratioCutoff, groupCutoff, sig_t, E=None):
        self.name = name
        self.G = G
        self.geoOption = geoOption
        self.contig = contig
        self.minCutoff = minCutoff
        self.ratioCutoff = ratioCutoff
        self.groupCutoff = groupCutoff
        self.structure = bounds
        self.mask = np.array([i for c in range(max(bounds) + 1) for i, g in enumerate(bounds) if g == c])
        self.counts = dict(zip(*np.unique(bounds, return_counts=True)))
        self.maxOrder = np.max(list(self.counts.values()))
        self.fname = '{}_{}_{}g_m{}_r{}_g{}'.format(name, geoOption, G, round(minCutoff, 2), round(ratioCutoff, 2), groupCutoff)
        self.sig_t = sig_t
        self.E = E

    def __str__(self):
        s = 'Name: {}'.format(self.name)
        s += '\nGeometry: {}'.format(self.geoOption)
        s += '\nMaxOrder: {}'.format(self.maxOrder)
        #s += '\nStructure: {}'.format(' '.join(self.structure.astype(str)))
        struct = np.concatenate(([0], np.cumsum([self.counts[i] for i in range(max(self.structure) + 1)])))
        s += '\nStructure:'
        for i in range(len(struct) - 1):
            s += '\n\tGroup {}: '.format(i)
            s += '{}-{}'.format(struct[i] + 1, struct[i+1]) if struct[i] + 1 != struct[i+1] else '{}'.format(struct[i] + 1)
        #s += '\nMask: {}'.format(' '.join(self.mask.astype(str)))
        s += '\nCounts: {}'.format(', '.join(['CG-{}={}'.format(key, item) for key, item in self.counts.items()]))
        s += '\nFileName: {}'.format(self.fname)

        return s

    def writeFile(self):
        # Get the coarse group bounds given the total cross sections
        s = '{}\n{}'.format(self.name, self.structure.tolist())

        with open('XS/structure_{}'.format(self.fname), 'w') as f:
            f.write(s)

def getMats(geoOption):
    if geoOption == 'inf':
        return [0]
    elif geoOption == 'full':
        return [0, 3, 9]
    elif geoOption == 'core1':
        return [0, 1, 2, 9]
    elif geoOption == 'core2':
        return [0, 1, 3, 9]
    elif 'c5g7' in geoOption:
        return [0, 3, 9]
        if 'uo2' in geoOption:
            return [5, 9]
        elif 'mox' in geoOption:
            if 'low' in geoOption:
                return [6, 9]
            elif 'mid' in geoOption:
                return [7, 9]
            elif 'high' in geoOption:
                return [8, 9]
            else:
                return [6, 7, 8, 9]
        else:
            return [5, 6, 7, 8, 9]
    else:
        raise NotImplementedError('{} has not been implented'.format(geoOption))


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
        E = np.array(f.readline().split()).astype(float)[::-1]
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

    return np.array(sig_t), E


def findContiguousBounds(sig_t, minCutoff, ratioCutoff, groupCutoff):
    '''
    Find the coarse group structure given a set of cross sections

    Inputs:
        sig_t: array of the total XS indexed by material [mat, G]
        minCutoff: XS below which the ratio cutoff is ignored
        ratioCutoff: Ratio of largest to smallest XS in a coarse group
        groupCutoff: Maximum number of fine groups in a coarse group
    Outputs:
        bounds: Starting fine group indices for each coarse group
    '''

    def reset(xs):
        '''Reset the min and max cutoffs to the input cross section'''
        minXS = np.maximum(xs, minCutoff)
        maxXS = xs
        return minXS, maxXS

    sig_t = np.array(sig_t)

    # Get the number of groups
    nGroup = len(sig_t[0])

    # Initialize the boundary at the lowest energy group
    bounds = [nGroup + 1]

    # Initialize the cutoff bounds
    minXS, maxXS = reset(sig_t[:,-1])

    # Loop through the cross sections from greatest to least
    for i, xs in enumerate(sig_t.T[::-1]):
        group = nGroup - i

        # Check if the xs is below the min or above the max
        minXS = np.minimum(xs, minXS)
        maxXS = np.maximum(xs, maxXS)
        ratio = maxXS - minXS

        # Check for a ratio that is too large
        if (max(ratio) > ratioCutoff and max(maxXS) > minCutoff) or bounds[-1] - group > groupCutoff:
            bounds.append(group + 1)
            # Reset the cutoff bounds
            minXS, maxXS = reset(xs)

    # Add the highest energy group bound
    bounds.append(1)
    # Reverse to the natural order of structures
    bounds = np.array(bounds[::-1])

    return np.array([i for i, b in enumerate(bounds[1:] - bounds[:-1]) for j in range(b)])

def findNoncontiguousBounds(sig_t, minCutoff=1.0, ratioCutoff=2.0, groupCutoff=60):
    '''
    Find the coarse group structure given a set of cross sections

    Inputs:
        sig_t: array of the total XS indexed by material [mat, G]
        minCutoff: XS below which the ratio cutoff is ignored
        ratioCutoff: Ratio of largest to smallest XS in a coarse group
        groupCutoff: Maximum number of fine groups in a coarse group
    Outputs:
        structure: mapping of fine group to coarse group
    '''

    G = len(sig_t[0])

    # Get the minimum and maximum of all materials
    minXS = np.min(sig_t, axis=0)
    minmap = np.argsort(minXS)[::-1]
    maxXS = np.max(sig_t, axis=0)
    maxmap = np.argsort(maxXS)[::-1]

    unsorted_groups = list(range(G))

    # Initialize a dictionary
    structure = np.zeros(G) - 1
    coarse_group_index = 0
    while len(unsorted_groups) > 0:
        # Get the largest unsorted cross section
        maximum = maxXS[maxmap[0]]
        # If above the minimum cross section cutoff
        if maximum > minCutoff:
            # Find the ratio for the current group
            #cutoff = max(ratioCutoff, maximum / minXS[maxmap[0]])
            cutoff = ratioCutoff
            # Select all groups with a smaller ratio than the cutoff
            cutmap = minmap[maximum - minXS[minmap] <= cutoff]
            if len(cutmap) == 0:
                cutmap = [maxmap[0]]
        else:
            # Select all remaining groups
            cutmap = minmap
        # Only allow groupCutoff groups into the current group
        cutmap = cutmap[:groupCutoff]
        # Assign selected groups to the coarse_group_index
        structure[cutmap] = coarse_group_index
        # Remove the assigned groups from minmap, maxmap, and unsorted_groups
        minmap = np.array([g for g in minmap if g not in cutmap])
        maxmap = np.array([g for g in maxmap if g not in cutmap])
        unsorted_groups = [g for g in unsorted_groups if g not in cutmap]
        # Increment the coarse_group_index
        coarse_group_index += 1

    return (structure * -1 + max(structure)).astype(int)

def computeBounds(G, geoOption, contig, minCutoff, ratioCutoff, groupCutoff):
    '''
    This function determines which fine groups should belong to each coarse group

    The boundaries will not necessarily be contiguous

    Inputs:
        fname: name of the cross section file in anlxs format
    '''

    # Contig options
    names = {0: 'contiguous_min',
             1: 'contiguous_max',
             2: 'contiguous_mean',
             3: 'continuous_difference',
             4: 'noncontiguous_min',
             5: 'noncontiguous_max',
             6: 'noncontiguous_mean',
             7: 'noncontiguous_difference'}

    # Select the material types for the geoOption
    mats = getMats(geoOption)

    # Load the total cross section for group G
    sig_t, E = getXS('XS/{}gXS.anlxs'.format(G))

    # Slice out only the cross sections used for the current geo
    mask = np.array([r in mats for r in range(len(sig_t))]).astype(bool)
    sig_t = sig_t[mask]

    # Number the energy groups
    groups = np.arange(G)

    # Get the minimum XS across all materials
    minXS = np.min(sig_t, axis=0)
    # Get the maximum XS across all materials
    maxXS = np.max(sig_t, axis=0)
    # Get the mean XS across all materials
    meanXS = np.mean(sig_t, axis=0)

    # Select which XS upon which to compute the bounds
    if contig == 0:  # contiguous min
        mask = findContiguousBounds([minXS], minCutoff, ratioCutoff, groupCutoff)
    elif contig == 1:  # contiguous max
        mask = findContiguousBounds([maxXS], minCutoff, ratioCutoff, groupCutoff)
    elif contig == 2:  # contiguous mean
        mask = findContiguousBounds([meanXS], minCutoff, ratioCutoff, groupCutoff)
    elif contig == 3:  # continuous difference
        mask = findContiguousBounds(sig_t, minCutoff, ratioCutoff, groupCutoff)
    elif contig == 4:  # non-contiguous min
        mask = findNoncontiguousBounds([minXS], minCutoff, ratioCutoff, groupCutoff)
    elif contig == 5:  # non-contiguous max
        mask = findNoncontiguousBounds([maxXS], minCutoff, ratioCutoff, groupCutoff)
    elif contig == 6:  # non-contiguous mean
        mask = findNoncontiguousBounds([meanXS], minCutoff, ratioCutoff, groupCutoff)
    elif contig == 7:  # non-contiguous difference
        mask = findNoncontiguousBounds(sig_t, minCutoff, ratioCutoff, groupCutoff)

    structure = Grouping(G, geoOption, contig, names[contig], mask, minCutoff, ratioCutoff, groupCutoff, sig_t, E)

    makePlot(structure, minXS, maxXS)

    return structure


def makePlot(structure, minXS, maxXS):
    # Plot the cross sections
    colors = ['b', 'g', 'r', 'c', 'm', 'orange', 'purple', 'grey']
    ls = '-' if structure.contig < 4 else '--'

    groups = np.arange(structure.G)
    mask = structure.structure

    for i, m in enumerate(mask):
        submask = mask == m
        plt.fill_between(groups[submask] + 1, minXS[submask], maxXS[submask], label='Group {}'.format(m), linestyle=ls, color=colors[m%len(colors)])
        #plt.semilogy(groups[i], minXS[i], label='Group {}'.format(m), ls=ls, marker='o', c=colors[m%len(colors)])
        #plt.semilogy(groups[i], maxXS[i], label='Group {}'.format(m), ls=ls, marker='s', c=colors[m%len(colors)])
    plt.yscale('log')
    plt.xlabel('Fine group number')
    plt.ylabel('Total cross section [cm$^{-1}$]')
    plt.xlim([0, structure.G+1])
    plt.grid(True)
    #plt.title(structure.name + ' {}'.format(len(structure.counts.keys())))
    plt.savefig('XS/structure_plot_{}.png'.format(structure.fname), transparent=True)
    plt.clf()

    E = structure.E[::-1]
    E = 0.5 * (E[1:] + E[:-1])

    for i, m in enumerate(mask):
        submask = mask == m
        plt.fill_between(E[submask], minXS[submask], maxXS[submask], label='Group {}'.format(m), linestyle=ls, color=colors[m%len(colors)])
        #plt.semilogy(groups[i], minXS[i], label='Group {}'.format(m), ls=ls, marker='o', c=colors[m%len(colors)])
        #plt.semilogy(groups[i], maxXS[i], label='Group {}'.format(m), ls=ls, marker='s', c=colors[m%len(colors)])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [MeV]')
    plt.ylabel('Total cross section [cm$^{-1}$]')
    plt.grid(True)
    #plt.title(structure.name + ' {}'.format(len(structure.counts.keys())))
    plt.savefig('XS/structure_plot_{}_E.png'.format(structure.fname), transparent=True)
    plt.clf()

def getInfo(task, kill=False):
    Gs = [44, 238, 1968]
    geos = ['inf', 'full', 'core1', 'core2', 'c5g7-small']
    contigs = [1]
    mins = [0]
    ratios = [1.3]
    groups = [60]

    item = 0
    groupBounds = []

    for geo in geos:
        for G in Gs:
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

def f(s):
    '''Convert a structure into a proper string'''
    try:
        return '{}'.format(s[0] + 1) if s[1] - 1 == s[0] else '{}-{}'.format(s[0] + 1, s[1])
    except IndexError:
        return ''


if __name__ == '__main__':
    # getInfo(1, True)

    #task = os.environ['SLURM_ARRAY_TASK_ID']
    #task = int(task) - 1


    for geo in ['full', 'core1', 'core2']:
        print(geo)
        s = '& 44-group & \multicolumn{2}{c|}{238-group} & \multicolumn{4}{c}{1968-group}\\\\\n'
        s += '\hline\n'
        s += 'add\'l. \# & (+0) & (+0) & (+20) & (+0) & (+20) & (+40) & (+60)\\\\\n'
        s += '\hline\n'

        # Build the coarse-group structure
        s44 = computeBounds(44, geo, 1, 0.0, 1.3, 60)
        c44 = np.concatenate(([0], np.cumsum([s44.counts[i] for i in range(max(s44.structure)+1)])))
        s238 = computeBounds(238, geo, 1, 0.0, 1.3, 60)
        c238 = np.concatenate(([0], np.cumsum([s238.counts[i] for i in range(max(s238.structure)+1)])))
        s1968 = computeBounds(1968, geo, 1, 0.0, 1.3, 60)
        c1968 = np.concatenate(([0], np.cumsum([s1968.counts[i] for i in range(max(s1968.structure)+1)])))

        for i in range(20):
            s += 'CG {:2d} & {:5s} & {:7s} & {:7s} & {:9s} & {:9s} & {:9s} & {:9s} \\\\\n'.format(i+1, f(c44[i:i+2]), f(c238[i:i+2]), f(c238[i+20:i+22]), f(c1968[i+0:i+2]), f(c1968[i+20:i+22]), f(c1968[i+40:i+42]), f(c1968[i+60:i+62]))
        print(s)







        #pickle.dump(structure, open('XS/structure{}_{}_{}g_m{}_r{}_g{}.p'.format(contig, geo, G, min_cutoff, ratio_cutoff, group_cutoff), 'wb'))

