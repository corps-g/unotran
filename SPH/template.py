import numpy as np
from sph import XS, DGMSOLVER
import matplotlib.pyplot as plt


def buildGEO(pin_map, homogenzied=False):
    fine_map = [150, 200, 150]
    coarse_map = [0.0, 0.45, 1.05, 1.5]
    material_map = [[4, 1, 4], [4, 2, 4], [4, 3, 4], [4, 4, 4]]  # High UO2 | Low UO2 | RCC | Water

    npins = len(pin_map)

    if homogenzied:
        fine_map = [500]
        coarse_map = [coarse_map[0], coarse_map[-1]]
        material_map = [[i + 1] for i in range(npins)]
        pin_map = range(npins)

    fm = fine_map * npins
    cm = [xx + i * coarse_map[-1] for i in range(npins) for xx in coarse_map[:-1]] + [npins * coarse_map[-1]]
    mm = []
    for p in pin_map:
        mm += material_map[p]

    return npins, fm, cm, mm


def plot(ref, homo):
    '''
    Create a plot comparing the reference array to the homogenized one
    '''

    ncells = len(ref.phi[0]) / len(ref.phi_homo[0])

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


def runSPH(G, pin_map, xs_name):
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
    old_mu = np.ones((G, nPin))
    fname = '_homo.'.join(xs_name.split('.'))

    # Solve for the reference problem

    # Write the reference cross sections

    # Build the homogenized geometry

    for i in range(1000):
        # Get the homogenized solution

        # Compute the SPH factors

        # Write the new cross sections adjusted by SPH

        # Compute the error between successive iterations

        # Save the previous SPH factors

        # Provide iteration output
        print 'Iter: {}    Error: {}'.format(i + 1, err)
        if err < 1e-6:
            break

    print 'SPH factors'
    print mu

    rxn_ref = ref.phi_homo * (ref.sig_t_homo - ref.sig_s_homo)
    rxn_homo = homo.phi_homo * (homo.sig_t_homo - homo.sig_s_homo)
    print 'Make sure these reactions rates are the same if SPH is working properly'
    print rxn_ref
    print rxn_homo

    return ref_XS


def makeColorset(G, pin_map, xs_name, homog=False, norm=None):
    nPin, fm, cm, mm = buildGEO(pin_map, homog)

    S = DGMSOLVER(G, xs_name, fm, cm, mm, nPin, norm)

    return S


if __name__ == '__main__':
    np.set_printoptions(precision=6)
    xs_name = 'XS/colorset.anlxs'
    G = 1

    # Get the reference solution
    pin_map = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ref = makeColorset(1, pin_map, xs_name, False)

    # Get the homogenized cross sections
    uo2low = runSPH()   
    uo2high = runSPH()
    uo2XS = uo2low + uo2high

    # Write the SPH cross sections

    # Get the homogenized solution

    print 'Reference reaction rates'
    print rxn_ref
    print 'Homogenzied reaction rates'
    print rxn_homo
    np.set_printoptions(precision=6, suppress=True)
    print 'Error in reaction rates'
    print (rxn_homo - rxn_ref) / rxn_ref * 100

    plot(ref, homo)
