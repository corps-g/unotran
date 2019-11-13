import numpy as np
import sys
sys.path.append('/homes/rlreed/workspace/unotran/src')
from sph import XS, DGMSOLVER
import matplotlib.pyplot as plt
from structures import structure
from coarseBounds import computeBounds, Grouping
from scipy.optimize import minimize, Bounds, fsolve, root
import pickle


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


def runSPH(G, pin_map, xs_name, mapping):
    '''
    Run the SPH problem

    Inputs:
        G       - Number of groups in the problem
        pin_map - List of pins in the geometry
        xs_name - Str with the cross section file in anlxs format
    '''

    print('compute reference reaction rates')
    # Build the reference geometry
    nPin, fm, cm, mm = buildGEO(pin_map, False)

    # Create the initial SPH factors
    old_mu = np.ones((mapping.nCG, nPin))
    fname = '_homo.'.join(xs_name.split('.')).replace('XS/', data_path + '/')

    # Solve for the reference problem
    ref = DGMSOLVER(G, xs_name, fm, cm, mm, nPin, mapping=mapping)
    iter_k = ref.iter_k
    iter_phi = ref.iter_phi
    iter_psi = ref.iter_psi

    ref_XS = XS(ref.sig_t_homo, ref.sig_f_homo, ref.chi_homo, ref.sig_s_homo)

    # Write the reference cross sections
    ref_XS.write_homogenized_XS(fname, old_mu)

    ref_rate = ref.phi_homo * ref.sig_t_homo
    print(ref_rate)

    # Build the homogenized geometry
    nPin, fm, cm, mm = buildGEO(pin_map, True)

    nCG = mapping.nCG

    print('start')

    for i in range(100000):
        # Get the homogenized solution
        homo = DGMSOLVER(nCG, fname, fm, cm, mm, nPin, ref.norm, k=iter_k, phi=iter_phi, psi=iter_psi)
        iter_k = homo.iter_k
        iter_phi = homo.iter_phi
        iter_psi = homo.iter_psi

        # Compute the SPH factors
        mu = ref.phi_homo / homo.phi_homo

        # Write the new cross sections adjusted by SPH
        ref_XS.write_homogenized_XS(fname, mu)

        # Compute the error in reaction rates
        homo_rate = homo.phi_homo * homo.sig_t_homo
        print(mu)
        err = np.sum(np.abs(homo_rate - ref_rate))
        # err = np.linalg.norm(mu - old_mu, np.inf)

        # Save the previous SPH factors
        # old_mu = np.copy(mu)

        # Provide iteration output
        print('Iter: {}    Error: {:6.4e}'.format(i + 1, err))
        if err < 1e-5:
            break

    print('SPH factors')
    print(mu)

    rxn_ref = ref.phi_homo * ref.sig_t_homo  # - np.sum(ref.sig_s_homo, axis=0))
    rxn_homo = homo.phi_homo * homo.sig_t_homo  # - np.sum(homo.sig_s_homo, axis=0))
    print('Make sure these reactions rates are the same if SPH is working properly')
    print(rxn_homo - rxn_ref)

    return ref_XS


if __name__ == '__main__':
    np.set_printoptions(precision=6)

    G = 44
    data_path = 'data2'
    assay_map = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    dgmstructure = computeBounds(G, 'full', 1, 0.0, 1.3, 60)
    O = dgmstructure.maxOrder
    xs_name = 'XS/{}gXS.anlxs'.format(G)

    for o in range(dgmstructure.maxOrder):
        # Get the homogenized cross sections
        mapping = structure(G, dgmstructure, o)
        ass1 = runSPH(G, assay_map, xs_name, mapping)
        pickle.dump(ass1, open('{}/refXS_sph_{}_o{}.p'.format(data_path, G, o), 'wb'))

