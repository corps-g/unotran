import numpy as np
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
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
np.set_printoptions(precision=8, suppress=True)


def buildGEO(ass_map, homogenzied=False):
    fine_map = [6, 18, 18, 18, 18, 6]
    coarse_map = [1.1176, 3.2512, 3.2512, 3.2512, 3.2512, 1.1176]

    if G > 40:
        material_map = [[5, 1, 2, 2, 1, 5], [5, 1, 3, 3, 1, 5], [5, 1, 4, 4, 1, 5]]
    else:
        material_map = [[9, 1, 2, 2, 1, 9], [9, 1, 3, 3, 1, 9], [9, 1, 4, 4, 1, 9]]

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


def runSPH(G, pin_map, xs_name, dgmstructure, order):
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

    if False:
        print('Loading previous reference')
        ref = pickle.load(open('ref_dgm.p', 'rb'))
    else:
        print('Computing reference')
        ref = DGMSOLVER(G, xs_name, fm, cm, mm, nPin, dgmstructure, order=order)
        pickle.dump(ref, open('ref_dgm.p', 'wb'))
    na = np.newaxis
    rxn_ref = np.sum(ref.sig_t_homo * ref.phi_homo[:, na, :, :], axis=0)
    rxn_ref_s = np.sum(ref.sig_s_homo * ref.phi_homo[:, na, :, na, :], axis=0)
    rxn_ref_f = np.sum(ref.sig_f_homo * ref.phi_homo[:, :, :], axis=0)

    space_map = [r for r in range(nPin) for _ in range(ref.phi.shape[2] // nPin)]

    test_rxn_t = np.zeros((rxn_ref.shape), order='f')
    test_rxn_s = np.zeros((rxn_ref_s.shape), order='f')
    test_rxn_f = np.zeros((rxn_ref_f.shape), order='f')
    L = np.zeros((nPin))

    for c, r in enumerate(space_map):
        L[r] += ref.dx[c]

    for c, r in enumerate(space_map):
        test_rxn_t[:, :, r] += np.sum(ref.sig_t[:, :, :, c] * ref.phi[:, na, :, c], axis=0) * ref.dx[na, na, c] / L[na, na, r]
        test_rxn_s[:, :, :, r] += np.sum(ref.sig_s[:, :, :, :, c] * ref.phi[:, na, :, na, c], axis=0) * ref.dx[na, na, na, c] / L[na, na, na, r]
        test_rxn_f[:, r] += np.sum(ref.vsig_f[:, :, c] * ref.phi[:, :, c], axis=0) * ref.dx[na, c] / L[na, r]

    # np.testing.assert_allclose(0.0, rxn_ref - test_rxn_t, atol=1e-14)
    np.testing.assert_allclose(0.0, rxn_ref_s - test_rxn_s, atol=1e-14)
    np.testing.assert_allclose(0.0, rxn_ref_f - test_rxn_f, atol=1e-14)

    nCellPerPin = ref.phi.shape[2] // ref.npin

    ref_XS = XS(G, nCellPerPin, ref.sig_t_homo, ref.sig_f_homo, ref.chi_homo, ref.sig_s_homo)

    # Build the homogenized geometry
    nPin, fm, cm, mm = buildGEO(pin_map, True)

    nCG = max(dgmstructure.structure) + 1

    mu = np.ones((order + 1, nCG, ref.npin), order='f')
    for cg, MO in dgmstructure.counts.items():
        mu[MO:, cg, :] *= 0
    mu = np.ones((nCG, ref.npin), order='f')
    old_mu = np.copy(mu)

    print(rxn_ref)

    print('start SPH')

    for i in range(1000):
        XS_args = ref_XS.write_homogenized_XS(mu)

        # Get the homogenized solution
        homo = DGMSOLVER(nCG, fname, fm, cm, mm, nPin, dgmstructure, ref.norm, XS=XS_args, order=order)
        rxn_homo = np.sum(homo.sig_t_homo * homo.phi_homo[:, na, :, :], axis=0)
        # Compute the SPH factors
        # mu = np.nan_to_num(ref.phi_homo[0] / homo.phi_homo[0])
        mu *= np.nan_to_num(rxn_ref[0] / rxn_homo[0] - 1.0) ** 1.0 + 1.0
        print(mu)

        # Compute the error in reaction rates
        print('rate')
        # err = np.linalg.norm(rxn_homo.flatten() - rxn_ref.flatten(), np.inf)
        print(rxn_homo - rxn_ref)
        err = np.linalg.norm(mu.flatten() - old_mu.flatten(), np.inf)
        old_mu = np.copy(mu)

        # Provide iteration output
        print('Iter: {} Order: {}   Error: {:6.4e}'.format(i + 1, order, err))
        if err < 1e-6:
            break

    print('SPH factors')
    print(mu)

    print('Make sure these reactions rates are the same if SPH is working properly')
    print(rxn_ref - rxn_homo)
    # print(rxn_homo)

    pickle.dump(ref_XS, open('refXS_dgm.p', 'wb'))
    pickle.dump(homo, open('homo_dgm.p', 'wb'))

    return ref_XS


if __name__ == '__main__':
    np.set_printoptions(precision=6)

    task = int(os.environ['SLURM_ARRAY_TASK_ID'])

    G = 44
    basis = 'dlp'

    dgmstructure = computeBounds(G, 'core2', 1, 0.0, 1.0, 60)
    print(dgmstructure)
    if 'klt' in basis:
        makeKLT(basis, dgmstructure)
    else:
        makeDLP(dgmstructure)

    dgmstructure.fname = '{}_{}'.format(basis, dgmstructure.fname)

    xs_name = 'XS/{}gXS.anlxs'.format(G)

    if False:
        for o in range(dgmstructure.maxOrder):
            if o in [1]:
                continue
            # Get the homogenized cross sections
            ass1 = runSPH(G, [0], xs_name, dgmstructure, order=o)
            ass2 = runSPH(G, [2], xs_name, dgmstructure, order=o)

            pickle.dump(ass1, open('refXS_dgm_ass1_o{}.p'.format(o), 'wb'))
            pickle.dump(ass2, open('refXS_dgm_ass2_o{}.p'.format(o), 'wb'))
            pickle.dump(ass1 + ass2, open('refXS_dgm_o{}.p'.format(o), 'wb'))
    else:
        ass_map = [0, 2]
        item = -1
        o = task
        # Get the homogenized cross sections
        ass1 = runSPH(G, ass_map, xs_name, dgmstructure, order=o)
        pickle.dump(ass1, open('refXS_dgm_o{}.p'.format(o), 'wb'))

    exit()

    # Write the SPH cross sections
    uo2XS.write_homogenized_XS('XS/colorset_homogenized.anlxs')

    # Get the homogenized solution
    homo = makeColorset(G, pin_map, 'XS/colorset_homogenized.anlxs', True)

    rxn_ref = ref.phi_homo * (ref.sig_t_homo - ref.sig_s_homo)
    rxn_homo = homo.phi_homo * (homo.sig_t_homo - homo.sig_s_homo)

    print('Reference reaction rates')
    print(rxn_ref)
    print('Homogenzied reaction rates')
    print(rxn_homo)
    np.set_printoptions(precision=3, suppress=True)
    print('Error in reaction rates')
    print((rxn_homo - rxn_ref) / rxn_ref * 100)

    plot(ref, homo)
