import numpy as np
import sys
sys.path.append('/homes/rlreed/workspace/unotran/src')
from sph_dgm import XS, DGMSOLVER
import matplotlib.pyplot as plt
from structures import structure
from scipy.optimize import minimize
from coarseBounds import computeBounds, Grouping
from makeDLPbasis import makeBasis as makeDLP
from makeKLTbasis import makeBasis as makeKLT
import pickle
import time
import os
np.set_printoptions(linewidth=240, precision=8, suppress=True)


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


def plot(ref, homo):
    '''
    Create a plot comparing the reference array to the homogenized one
    '''

    ncells = int(len(ref.phi[0]) / len(ref.phi_homo[0]))

    rxn = ref.phi * (ref.sig_t - ref.sig_s)
    rxn_homo = ref.phi_homo * (ref.sig_t_homo - ref.sig_s_homo)
    rxn_sph = homo.phi_homo * (homo.sig_t_homo - homo.sig_s_homo)
    x = np.cumsum(ref.dx.flatten())

    plt.figure(0)
    plt.plot(x, rxn.flatten(), label='reference')
    plt.plot(x, [h for h in rxn_homo.flatten() for _ in range(ncells)], label='homogenized')
    plt.plot(x, [h for h in rxn_sph.flatten() for _ in range(ncells)], label='sph')
    plt.legend()
    plt.xlim([0.0, np.ceil(x[-1])])
    plt.xlabel('length [cm]')
    plt.ylabel('absorption rate')

    plt.figure(1)
    plt.plot(x, ref.phi.flatten(), label='reference')
    plt.plot(x, [h for h in ref.phi_homo.flatten() for _ in range(ncells)], label='homogenized')
    plt.plot(x, [h for h in homo.phi_homo.flatten() for _ in range(ncells)], label='sph')
    plt.legend()
    plt.xlim([0.0, np.ceil(x[-1])])
    plt.xlabel('length [cm]')
    plt.ylabel('scalar flux')

    plt.show()


def runSPH(G, pin_map, xs_name, dgmstructure, method, homogOption):
    '''
    Run the SPH problem

    Inputs:
        G       - Number of groups in the problem
        pin_map - List of pins in the geometry
        xs_name - Str with the cross section file in anlxs format
    '''
    # Build the reference geometry
    nPin, fm, cm, mm = buildGEO(pin_map, False)

    mmap = [m - 1 for r, m in zip(fm, mm) for _ in range(r)]

    # Create the initial SPH factors
    fname = '_homo.'.join(xs_name.split('.'))

    # Solve for the reference problem

    print('Computing reference')
    ref = DGMSOLVER(G, xs_name, fm, cm, mm, nPin, dgmstructure, homogOption=homogOption)

    iter_k = ref.iter_k
    iter_phi = ref.iter_phi
    iter_psi = ref.iter_psi

    na = np.newaxis
    rxn_ref = np.sum(ref.sig_t_homo * ref.phi_homo[:, na, :, :], axis=0)
    # rxn_ref_s = np.sum(ref.sig_s_homo * ref.phi_homo[:, na, :, na, :], axis=0)
    rxn_ref_f = np.sum(ref.sig_f_homo * ref.phi_homo[:, :, :], axis=0)

    space_map = [r for r in range(nPin) for _ in range(ref.phi.shape[2] // nPin)]

    nCellPerPin = ref.phi.shape[2] // ref.npin
    if homogOption in [3, 4]:
        nCellPerPin = 1

    ref_XS = XS(G, nCellPerPin, ref.sig_t_homo, ref.sig_f_homo, ref.chi_homo, ref.sig_s_homo)

    # Build the homogenized geometry
    nPin, fm, cm, mm = buildGEO(pin_map, True)

    nCG = max(dgmstructure.structure) + 1

    mu = np.ones((nCG, ref.npin), order='f')
    old_mu = np.copy(mu)

    print(rxn_ref)

    print('start SPH')

    for i in range(1000):
        XS_args = ref_XS.write_homogenized_XS(mu, method=method)

        # Get the homogenized solution
        homo = DGMSOLVER(nCG, fname, fm, cm, mm, nPin, dgmstructure, ref.norm, XS=XS_args, homogOption=homogOption, k=iter_k, phi=iter_phi, psi=iter_psi)
        iter_k = homo.iter_k
        iter_phi = homo.iter_phi
        iter_psi = homo.iter_psi

        rxn_homo = np.sum(homo.sig_t_homo * homo.phi_homo[:, na, :, :], axis=0)
        # rxn_homo_s = np.sum(homo.sig_s_homo * homo.phi_homo[:, na, :, na, :], axis=0)
        rxn_homo_f = np.sum(homo.sig_f_homo * homo.phi_homo[:, :, :], axis=0)
        # Compute the SPH factors

        if 'rxn_t' in method:
            mu *= np.nan_to_num(rxn_ref[0] / rxn_homo[0])
        elif 'rxn_f' in method:
            mu *= np.nan_to_num(rxn_ref_f / rxn_homo_f)
        elif 'phi' in method:
            mu = np.nan_to_num(ref.phi_homo[0] / homo.phi_homo[0])
        print(mu)

        # Compute the error in reaction rates
        print('rate')
        # err = np.linalg.norm(rxn_homo.flatten() - rxn_ref.flatten(), np.inf)
        print((rxn_homo - rxn_ref)[:3])
        # print('rate scatter')
        # print((rxn_homo_s - rxn_ref_s)[:3])
        print('rate fission')
        print((rxn_homo_f - rxn_ref_f))
        err = np.linalg.norm((mu - old_mu).flatten(), np.inf)
        old_mu = np.copy(mu)

        # Provide iteration output
        print('Iter: {} Error: {:6.4e}'.format(i + 1, err))
        if err < 1e-5:
            break

    print('SPH factors')
    print(mu)

    print('Make sure these reactions rates are the same if SPH is working properly')
    print(rxn_ref - rxn_homo)
    # print(rxn_homo)

    #pickle.dump(ref_XS, open('refXS_dgm.p', 'wb'))
    #pickle.dump(homo, open('homo_dgm.p', 'wb'))

    return ref_XS


def getInfo(task, kill=False):
    Gs = [44, 238]
    geo = 'full'
    contigs = [1]
    min_cutoff = 0.0
    ratio_cutoff = 1.3
    group_cutoff = 60
    basisTypes = ['dlp', 'klt_full', 'klt_combine', 'klt_pins_full']
    methods = ['phi_mu_zeroth', 'phi_mu_all', 'rxn_t_mu_zeroth', 'rxn_f_mu_zeroth', 'rxn_t_mu_all', 'rxn_f_mu_all']
    homogOptions = [0]

    item = 0
    groupBounds = []
    for i, G in enumerate(Gs):
        for method in methods:
            for contig in contigs:
                for homogOption in homogOptions:
                    for basisType in basisTypes:
                        if item == task and not kill:
                            return G, geo, contig, min_cutoff, ratio_cutoff, group_cutoff, basisType, homogOption, method
                        else:
                            item += 1
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

    data_path = 'data2'

    parameters = getInfo(task)
    print(parameters)
    G, geo, contig, min_cutoff, ratio_cutoff, group_cutoff, basisType, homogOption, method = parameters

    dgmstructure = computeBounds(G, geo, contig, min_cutoff, ratio_cutoff, group_cutoff)
    print(dgmstructure)
    if 'klt' in basisType:
        makeKLT(basisType, dgmstructure)
    else:
        makeDLP(dgmstructure)

    dgmstructure.fname = '{}_{}'.format(basisType, dgmstructure.fname)

    xs_name = 'XS/{}gXS.anlxs'.format(G)

    ass_map = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # Get the homogenized cross sections
    ass1 = runSPH(G, ass_map, xs_name, dgmstructure, method, homogOption)
    pickle.dump(ass1, open('{}/refXS_dgm_{}_{}_h{}.p'.format(data_path, dgmstructure.fname, method, homogOption), 'wb'))

    print('complete')

