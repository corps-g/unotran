import numpy as np
import sys
sys.path.append('/homes/rlreed/workspace/unotran/src')
import sph_dgm
import sph
import pickle
from structures import structure
from coarseBounds import computeBounds, Grouping
from makeDLPbasis import makeBasis as makeDLP
from makeKLTbasis import makeBasis as makeKLT
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib import rc
rc('font', **{'family': 'serif'})
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['lines.linewidth'] = 1.85
rcParams['axes.labelsize'] = 20
rcParams['legend.numpoints'] = 1
rcParams.update({'figure.autolayout': True})
import os


def buildGEO(ass_map, homogenzied=False):
    fine_map = [3, 22, 3]
    coarse_map = [0.09, 1.08, 0.09]

    if G > 40:
        material_map = [[5, 1, 5], [5, 4, 5]]
    else:
        material_map = [[9, 1, 9], [9, 4, 9]]

    npins = len(ass_map)

    if homogenzied:
        fine_map = [sum(fine_map)]
        coarse_map = [sum(coarse_map)]
        material_map = [[i + 1] for i in range(npins)]
        if version in [1, 3]:
            ass_map = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        else:
            ass_map = range(npins)

    cm = [0.0]
    fm = []
    mm = []
    for i, ass in enumerate(ass_map):
        mm += material_map[ass]
        cm += coarse_map
        fm += fine_map
    cm = np.cumsum(cm)

    return npins, fm, cm, mm


def run_reference(G, mmap, xs_name, mapping):
    print('running the reference')
    # Build the reference geometry
    nPin, fm, cm, mm = buildGEO(mmap, False)

    # Run the reference problem
    ref = sph.DGMSOLVER(G, xs_name, fm, cm, mm, nPin, mapping=mapping, vacuum=False)

    # Save the flux and cross sections
    np.save('{}/ref_phi_{}'.format(data_path, G), ref.phi)
    np.save('{}/ref_sig_t_{}'.format(data_path, G), ref.sig_t)
    np.save('{}/ref_sig_f_{}'.format(data_path, G), ref.vsig_f)
    np.save('{}/ref_phi_homo_{}'.format(data_path, G), ref.phi_homo)
    np.save('{}/ref_sig_t_homo_{}'.format(data_path, G), ref.sig_t_homo)
    np.save('{}/ref_sig_f_homo_{}'.format(data_path, G), ref.sig_f_homo)


def run_homogenized(G, mmap, xs_name, mapping, order, useSPH=True):
    print('running the homogenized problem')
    # Write the homogenized cross sections
    refXS = pickle.load(open('{}/refXS_sph_{}_o{}.p'.format(data_path, G, order), 'rb'))

    xs_name = '_homo_o{}.'.join(xs_name.split('.')).format(order)
    if useSPH:
        refXS.write_homogenized_XS(xs_name)
        extra = ''
    else:
        refXS.write_homogenized_XS(xs_name, np.ones(refXS.sig_t.shape))
        extra = '_nosph'

    # Build the reference geometry
    nPin, fm, cm, mm = buildGEO(mmap, True)

    # Run the homogenized problem
    homo = sph.DGMSOLVER(mapping.nCG, xs_name, fm, cm, mm, nPin, vacuum=False)

    # Save the flux and cross sections
    np.save('{}/homo_phi_{}{}_o{}'.format(data_path, G, extra, order), homo.phi)
    np.save('{}/homo_sig_t_{}{}_o{}'.format(data_path, G, extra, order), homo.sig_t)
    np.save('{}/homo_sig_f_{}{}_o{}'.format(data_path, G, extra, order), homo.vsig_f)
    np.save('{}/homo_phi_homo_{}{}_o{}'.format(data_path, G, extra, order), homo.phi_homo)
    np.save('{}/homo_sig_t_homo_{}{}_o{}'.format(data_path, G, extra, order), homo.sig_t_homo)
    np.save('{}/homo_sig_f_homo_{}{}_o{}'.format(data_path, G, extra, order), homo.sig_f_homo)

def run_dgm_homogenized(G, mmap, xs_name, mapping, order, method, basis, dgmstructure, homogOption):
    print('running the homogenized problem with DGM: order {}'.format(order))
    # Get the coarse group structure
    nCG = mapping.nCG
    xs_name = '_homo.'.join(xs_name.split('.'))

    # Build the basis
    if 'klt' in basis:
        makeKLT(basis, dgmstructure)
    else:
        makeDLP(dgmstructure)

    # Update structure name with basis
    dgmstructure.fname = '{}_{}'.format(basis, dgmstructure.fname)

    # Build the geometry
    nPin, fm, cm, mm = buildGEO(mmap, True)

    # Write the homogenized cross section moments
    ref_XS = pickle.load(open('{}/refXS_dgm_{}_{}_h{}.p'.format(data_path, dgmstructure.fname, method, homogOption), 'rb'))
    print(ref_XS.chi.shape)
    ref_XS.nCellPerPin = sum(fm) // nPin
    print(mm)
    XS_args = ref_XS.write_homogenized_XS(mmap=mm, order=order, method=method)
    # Run the homogenized problem with DGM
    homo = sph_dgm.DGMSOLVER(nCG, xs_name, fm, cm, mm, nPin, dgmstructure, XS=XS_args, vacuum=False, order=order, homogOption=0)

    # Save the flux and cross sections
    np.save('{}/dgm_phi_{}_{}_{}_o{}_h{}'.format(data_path, G, basis, method, order, homogOption), homo.phi)
    np.save('{}/dgm_sig_t_{}_{}_{}_o{}_h{}'.format(data_path, G, basis, method, order, homogOption), homo.sig_t)
    np.save('{}/dgm_sig_f_{}_{}_{}_o{}_h{}'.format(data_path, G, basis, method, order, homogOption), homo.vsig_f)
    np.save('{}/dgm_phi_homo_{}_{}_{}_o{}_h{}'.format(data_path, G, basis, method, order, homogOption), homo.phi_homo)
    np.save('{}/dgm_sig_t_homo_{}_{}_{}_o{}_h{}'.format(data_path, G, basis, method, order, homogOption), homo.sig_t_homo)
    np.save('{}/dgm_sig_f_homo_{}_{}_{}_o{}_h{}'.format(data_path, G, basis, method, order, homogOption), homo.sig_f_homo)


def getInfo(task, kill=False):
    Gs = [44, 238]
    geo = 'full'
    contigs = [1]
    min_cutoff = 0.0
    ratio_cutoff = 1.3
    group_cutoff = 60
    basisTypes = ['dlp', 'klt_full', 'klt_combine', 'klt_pins_full']
    methods = ['fine_mu', 'phi_mu_zeroth', 'phi_mu_all', 'rxn_t_mu_zeroth', 'rxn_f_mu_zeroth', 'rxn_t_mu_all', 'rxn_f_mu_all']
    homogOptions = [0]


    item = 0
    groupBounds = []
    for i, G in enumerate(Gs):
        flag = True
        maxOrder = min(G, group_cutoff)
        for method in methods:
            for contig in contigs:
                for homogOption in homogOptions:
                    if method == 'fine_mu' and homogOption != 0:
                        continue

                    for basisType in basisTypes:
                        for order in range(maxOrder):
                            if item == task and not kill:
                                return G, geo, contig, min_cutoff, ratio_cutoff, group_cutoff, basisType, homogOption, method, order, flag
                            else:
                                item += 1
                                flag = False
        groupBounds.append(item)
    if kill:
        groupBounds = [0] + groupBounds
        for i, G in enumerate(Gs):
            print('Group {:4d} cutoffs: {:>5d}-{:<5d}'.format(G, groupBounds[i] + 1, groupBounds[i + 1]))
    exit()


if __name__ == '__main__':
    # getInfo(1, True)

    task = os.environ['SLURM_ARRAY_TASK_ID']
    task = int(task) - 1

    version = 3
    data_path = 'data{}'.format('' if version == 1 else str(version))

    parameters = getInfo(task)
    print(parameters)
    G, geo, contig, min_cutoff, ratio_cutoff, group_cutoff, basis, homogOption, method, order, flag = parameters

    mmap = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    dgmstructure = computeBounds(G, geo, contig, min_cutoff, ratio_cutoff, group_cutoff)
    mapping = structure(G, dgmstructure)
    xs_name = 'XS/{}gXS.anlxs'.format(G)

    if order >= dgmstructure.maxOrder:
        print('{} >= {}, stopping'.format(order, dgmstructure.maxOrder))
        exit()

    if flag:
        run_reference(G, mmap, xs_name, mapping)
    run_homogenized(G, mmap, xs_name, structure(G, dgmstructure, order), order)
    run_homogenized(G, mmap, xs_name, structure(G, dgmstructure, order), order, False)
    run_dgm_homogenized(G, mmap, xs_name, mapping, order, method, basis, dgmstructure, homogOption)

    print('complete')

