import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile
from matplotlib import rc
rc('font',**{'family':'serif'})
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['lines.linewidth'] = 1.85
rcParams['axes.labelsize'] = 20
rcParams.update({'figure.autolayout': True})

def computeError(ref, phi):
    return np.linalg.norm(ref - phi) / np.linalg.norm(ref)

def getErrors(kind):

    index = np.s_[:,:,0] if kind == 'phi' else np.s_[:,:,:]


    with FortranFile('src/10pinreference_{}.bin'.format(kind), 'r') as f:
        ref = f.read_reals(dtype=np.float).reshape(280,-1,8)[index]

    full_err = []
    mox_err = []
    uo2_err = []
    comp_err = []

    for i in range(85):
        with FortranFile('klt_full_7/10pin_dgm_7g_full_{}_{}.bin'.format(i+1, kind), 'r') as f:
            full_err.append(computeError(ref, f.read_reals(dtype=np.float).reshape(280,-1,8)[index]))
        with FortranFile('klt_mox_7/10pin_dgm_7g_mox_{}_{}.bin'.format(i+1, kind), 'r') as f:
            mox_err.append(computeError(ref, f.read_reals(dtype=np.float).reshape(280,-1,8)[index]))
        with FortranFile('klt_uo2_7/10pin_dgm_7g_uo2_{}_{}.bin'.format(i+1, kind), 'r') as f:
            uo2_err.append(computeError(ref, f.read_reals(dtype=np.float).reshape(280,-1,8)[index]))
        with FortranFile('klt_combine_7/10pin_dgm_7g_combine_{}_{}.bin'.format(i+1, kind), 'r') as f:
            comp_err.append(computeError(ref, f.read_reals(dtype=np.float).reshape(280,-1,8)[index]))

    np.savetxt('full_err_{}'.format(kind), np.array(full_err))
    np.savetxt('mox_err_{}'.format(kind), np.array(mox_err))
    np.savetxt('uo2_err_{}'.format(kind), np.array(uo2_err))
    np.savetxt('comp_err_{}'.format(kind), np.array(comp_err))

def makePlots(kind):
    full = np.loadtxt('full_err_{}'.format(kind))
    mox = np.loadtxt('mox_err_{}'.format(kind))
    uo2 = np.loadtxt('uo2_err_{}'.format(kind))
    comp = np.loadtxt('comp_err_{}'.format(kind))

    x = range(85)
    plt.semilogy(x, full*100, 'r-', label='Full')
    plt.semilogy(x, mox*100, 'b-', label='MOX')
    plt.semilogy(x, uo2*100, 'b--', label='UO$_2$')
    plt.semilogy(x, comp*100, 'g-', label='Combined')
    plt.xlabel('Expansion Order')
    plt.ylabel('Scalar Flux relative error (%)')
    plt.legend()
    plt.xlim((0, 84))
    plt.ylim((1e-2, 1e2))
    plt.grid(True)
    plt.savefig('errors_{}.pdf'.format(kind))
    plt.clf()

if __name__ == '__main__':
    for kind in ['phi', 'psi']:
        getErrors(kind)
        makePlots(kind)
