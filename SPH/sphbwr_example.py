import numpy as np
import sys
sys.path.append('/homes/rlreed/workspace/unotran/src')
from coarseBounds import computeBounds, Grouping
import pickle
from makeDLPbasis import makeBasis as makeDLP
from makeKLTbasis import makeBasis as makeKLT
import sph
import sph_dgm
import pydgm


def buildGEO(ass_map):
    fine_map = [1]
    coarse_map = [1.26]

    material_map = [[1], [2]]
    material_map = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

    npins = len(ass_map)

    cm = [0.0]
    fm = []
    mm = []
    for i, ass in enumerate(ass_map):
        mm += material_map[ass]
        cm += coarse_map
        fm += fine_map
    cm = np.cumsum(cm)

    return npins, fm, cm, mm

def makeDGMXS(G, refXS, dgmstructure, basisType):
    if 'klt' in basisType:
        makeKLT(basisType, dgmstructure)
    else:
        makeDLP(dgmstructure)

    dgmstructure.fname = '{}_{}'.format(basisType, dgmstructure.fname)

    fname = '_homo.'.join(xs_name.split('.'))
    refXS.write_homogenized_XS(fname)

    nPin, fm, cm, mm = buildGEO(pin_map)

    dgm = sph_dgm.DGMSOLVER(G, fname, fm, cm, mm, nPin, dgmstructure, solveFlag=False)
    pydgm.dgmsolver.initialize_dgmsolver()
    dgm.extractInfo()

    pydgm.dgmsolver.finalize_dgmsolver()
    pydgm.control.finalize_control()

    nCellPerPin = dgm.phi.shape[2] // dgm.npin

    return sph_dgm.XS(G, nCellPerPin, dgm.sig_t, dgm.vsig_f, dgm.chi, dgm.sig_s)

if __name__ == '__main__':
    np.set_printoptions(precision=6)

    G = 44

    dgmstructure = computeBounds(G, 'full', 1, 0.0, 1.3, 60)
    fname = dgmstructure.fname
    xs_name = 'XS/{}gXS.anlxs'.format(G)
    pin_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    data_path = 'data2'

    # Get the homogenized cross sections
    refXS = pickle.load(open('{}/refXS_sph_space_{}.p'.format(data_path, G), 'rb'))

    for basis in ['dlp', 'klt_full', 'klt_combine', 'klt_pins_full']:
        dgmstructure.fname = fname
        XS = makeDGMXS(G, refXS, dgmstructure, basis)
        pickle.dump(XS, open('{}/refXS_dgm_{}_{}_h{}.p'.format(data_path, dgmstructure.fname, 'fine_mu', 0), 'wb'))

