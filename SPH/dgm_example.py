import numpy as np
from sph_dgm import XS, DGMSOLVER
import matplotlib.pyplot as plt
from structures import structure
from scipy.optimize import minimize, Bounds
from coarseBounds import computeBounds, Grouping
from makeDLPbasis import makeBasis as makeDLP
from makeKLTbasis import makeBasis as makeKLT


def buildGEO(ass_map, homogenzied=False):
    fine_map = [6, 18, 18, 18, 18, 6]
    coarse_map = [1.1176, 3.2512, 3.2512, 3.2512, 3.2512, 1.1176]

    material_map = [[5, 1, 2, 2, 1, 5], [5, 1, 3, 3, 1, 5], [5, 1, 4, 4, 1, 5]]

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


def runSPH(G, pin_map, xs_name, mapping, dgmstructure):
    '''
    Run the SPH problem

    Inputs:
        G       - Number of groups in the problem
        pin_map - List of pins in the geometry
        xs_name - Str with the cross section file in anlxs format
    '''
    # Build the reference geometry
    nPin, fm, cm, mm = buildGEO(pin_map, False)

    # Create the initial SPH factors
    fname = '_homo.'.join(xs_name.split('.'))

    # Solve for the reference problem
    ref = DGMSOLVER(G, xs_name, fm, cm, mm, nPin, dgmstructure, mapping=mapping)

    # print(np.sum(ref.phi_homo * ref.sig_t_homo[:,np.newaxis,:,:], axis=0))

    mu = np.ones(ref.phi_homo.shape)

    nCellPerPin = ref.phi.shape[2] // ref.npin

    ref_XS = XS(G, nCellPerPin, ref.sig_t_homo, ref.sig_f_homo, ref.chi_homo, ref.sig_s_homo)

    # Build the homogenized geometry
    nPin, fm, cm, mm = buildGEO(pin_map, True)

    nCG = mapping.nCG

    print('start')

    for i in range(1000):
        XS_args = ref_XS.write_homogenized_XS(mu)

        # Get the homogenized solution
        homo = DGMSOLVER(nCG, fname, fm, cm, mm, nPin, dgmstructure, ref.norm, mapping=mapping, XS=XS_args)

        # Compute the SPH factors
        mu = ref.phi_homo / homo.phi_homo

        # Compute the error in reaction rates
        err = np.sum(np.abs(np.sum(homo.phi_homo[:,np.newaxis,:,:] * homo.sig_t_homo, axis=0) - np.sum(ref.phi_homo[:,np.newaxis,:,:] * ref.sig_t_homo, axis=0)))
        # err = np.linalg.norm(mu - old_mu, np.inf)

        # Provide iteration output
        print('Iter: {}    Error: {:6.4e}'.format(i + 1, err))
        if err < 1e-7:
            break

    print('SPH factors')
    print(mu)

    rxn_ref = ref.phi_homo * ref.sig_t_homo# - np.sum(ref.sig_s_homo, axis=0))
    rxn_homo = homo.phi_homo * homo.sig_t_homo# - np.sum(homo.sig_s_homo, axis=0))
    print('Make sure these reactions rates are the same if SPH is working properly')
    print(rxn_ref)
    print(rxn_homo)

    return ref_XS


def makeColorset(G, pin_map, xs_name, homog=False, norm=None):
    nPin, fm, cm, mm = buildGEO(pin_map, homog)
    S = DGMSOLVER(G, xs_name, fm, cm, mm, nPin, norm)

    return S


if __name__ == '__main__':
    np.set_printoptions(precision=6)

    G = 44
    basis = 'klt_pins_core2'

    dgmstructure = computeBounds(G, 'core2', 1, 0.0, 1.0, 60)
    print(dgmstructure)
    makeKLT(basis, dgmstructure)

    dgmstructure.fname = '{}_{}'.format(basis, dgmstructure.fname)

    mapping = structure(G, 2)
    xs_name = 'XS/{}gXS.anlxs'.format(G)

    # Get the reference solution
    ass_map = [0, 2]
    #ref = makeColorset(G, ass_map, xs_name, False)


    # Get the homogenized cross sections
    ass1 = runSPH(G, ass_map, xs_name, mapping, dgmstructure)
    #ass2 = runSPH(G, [1], xs_name)
    #core = uo2low + uo2high

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
