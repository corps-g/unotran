import numpy as np
from coarseBounds import computeBounds, Grouping
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

def compare(G, methods, basis, homogOption, contig):

    # Reference values
    ref_sig_f = np.load('data/ref_sig_f_{}.npy'.format(G))
    ref_phi = np.load('data/ref_phi_{}.npy'.format(G))
    ref_fission_density = np.sum(ref_sig_f * ref_phi, axis=0)
    ref_fission_density = np.sum(ref_fission_density.reshape((10,-1)), axis=1)

    # Non DGM values
    homo_sig_f = np.load('data/homo_sig_f_{}.npy'.format(G))
    homo_phi = np.load('data/homo_phi_{}.npy'.format(G))
    homo_fission_density = np.sum(homo_sig_f * homo_phi, axis=0)
    homo_fission_density = np.sum(homo_fission_density.reshape((10,-1)), axis=1)

    homo_nosph_sig_f = np.load('data/homo_sig_f_{}_nosph.npy'.format(G))
    homo_nosph_phi = np.load('data/homo_phi_{}_nosph.npy'.format(G))
    homo_nosph_fission_density = np.sum(homo_nosph_sig_f * homo_nosph_phi, axis=0)
    homo_nosph_fission_density = np.sum(homo_nosph_fission_density.reshape((10,-1)), axis=1)

    # Non DGM values
    homo_full_sig_f = np.load('data/homo_full_sig_f_{}.npy'.format(G))
    homo_full_phi = np.load('data/homo_full_phi_{}.npy'.format(G))
    homo_full_fission_density = np.sum(homo_full_sig_f * homo_full_phi, axis=0)
    homo_full_fission_density = np.sum(homo_full_fission_density.reshape((10,-1)), axis=1)

    homo_nosph_full_sig_f = np.load('data/homo_full_sig_f_{}_nosph.npy'.format(G))
    homo_nosph_full_phi = np.load('data/homo_full_phi_{}_nosph.npy'.format(G))
    homo_nosph_full_fission_density = np.sum(homo_nosph_full_sig_f * homo_nosph_full_phi, axis=0)
    homo_nosph_full_fission_density = np.sum(homo_nosph_full_fission_density.reshape((10,-1)), axis=1)

    fd_ref = ref_fission_density / np.linalg.norm(ref_fission_density)
    fd_homo = homo_fission_density / np.linalg.norm(homo_fission_density)
    fd_homo_nosph = homo_nosph_fission_density / np.linalg.norm(homo_nosph_fission_density)
    fd_homo_full = homo_full_fission_density / np.linalg.norm(homo_full_fission_density)
    fd_homo_nosph_full = homo_full_fission_density / np.linalg.norm(homo_nosph_full_fission_density)
    err_homo = np.abs(fd_homo - fd_ref) / fd_ref * 100
    err_homo_nosph = np.abs(fd_homo_nosph - fd_ref) / fd_ref * 100
    err_homo_full = np.abs(fd_homo_full - fd_ref) / fd_ref * 100
    err_homo_nosph_full = np.abs(fd_homo_nosph_full - fd_ref) / fd_ref * 100

    MO = dgmstructure.maxOrder

    plt.figure(figsize=(6.4,4.8))
    for i, method in enumerate(methods):

        dgm_err = np.zeros((10, MO))
        dof = []

        for o in range(dgmstructure.maxOrder):
            dof.append(sum([min(o + 1, orders) for cg, orders in dgmstructure.counts.items()]))
            # DGM values
            try:
                dgm_sig_f = np.load('data/dgm_sig_f_{}_{}_{}_o{}_h{}.npy'.format(G, basis, method, o, homogOption))
                dgm_phi = np.load('data/dgm_phi_{}_{}_{}_o{}_h{}.npy'.format(G, basis, method, o, homogOption))
            except FileNotFoundError:
                print('missing data/dgm_phi_{}_{}_{}_o{}_h{}.npy'.format(G, basis, method, o, homogOption))
                dgm_sig_f = np.zeros(homo_sig_f.shape)
                dgm_phi = np.zeros(homo_phi.shape)
            dgm_fission_density = np.sum(np.sum(dgm_sig_f * dgm_phi, axis=0), axis=0)
            dgm_fission_density = np.sum(dgm_fission_density.reshape((10,-1)), axis=1)
            fd_dgm = dgm_fission_density / np.linalg.norm(dgm_fission_density)
            dgm_err[:, o] = np.abs(fd_dgm - fd_ref) / fd_ref * 100
        plt.semilogy(dof, np.max(dgm_err, axis=0), style[i], label=mName[i])
    plt.semilogy(dof, np.ones(len(dof)) * np.max(err_homo, axis=0), color='b', ls=':', label='Standard SPH')
    plt.semilogy(dof, np.ones(len(dof)) * np.max(err_homo_nosph, axis=0), color='b', ls='--', label='Standard No SPH')
    plt.semilogy(dof, np.ones(len(dof)) * np.max(err_homo_full, axis=0), color='k', ls=':', label='Spatial SPH')
    plt.semilogy(dof, np.ones(len(dof)) * np.max(err_homo_nosph_full, axis=0), color='k', ls='--', label='Spatial No SPH')
    plt.ylim([1e-1, 1e2])
    plt.legend(loc=0, ncol=3)
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Max Rel FD error [%]')
    plt.savefig('{}/DGM_SPH_FD_err_{}_{}_c{}_h{}.pdf'.format(plot_path, G, basis, contig, homogOption))
    plt.clf()


def compare2(G, method, bases, homogOption, contig):

    # Reference values
    ref_sig_f = np.load('data/ref_sig_f_{}.npy'.format(G))
    ref_phi = np.load('data/ref_phi_{}.npy'.format(G))
    ref_fission_density = np.sum(ref_sig_f * ref_phi, axis=0)
    ref_fission_density = np.sum(ref_fission_density.reshape((10,-1)), axis=1)

    # Non DGM values
    homo_sig_f = np.load('data/homo_sig_f_{}.npy'.format(G))
    homo_phi = np.load('data/homo_phi_{}.npy'.format(G))
    homo_fission_density = np.sum(homo_sig_f * homo_phi, axis=0)
    homo_fission_density = np.sum(homo_fission_density.reshape((10,-1)), axis=1)

    homo_nosph_sig_f = np.load('data/homo_sig_f_{}_nosph.npy'.format(G))
    homo_nosph_phi = np.load('data/homo_phi_{}_nosph.npy'.format(G))
    homo_nosph_fission_density = np.sum(homo_nosph_sig_f * homo_nosph_phi, axis=0)
    homo_nosph_fission_density = np.sum(homo_nosph_fission_density.reshape((10,-1)), axis=1)

    # Non DGM values
    homo_full_sig_f = np.load('data/homo_full_sig_f_{}.npy'.format(G))
    homo_full_phi = np.load('data/homo_full_phi_{}.npy'.format(G))
    homo_full_fission_density = np.sum(homo_full_sig_f * homo_full_phi, axis=0)
    homo_full_fission_density = np.sum(homo_full_fission_density.reshape((10,-1)), axis=1)

    homo_nosph_full_sig_f = np.load('data/homo_full_sig_f_{}_nosph.npy'.format(G))
    homo_nosph_full_phi = np.load('data/homo_full_phi_{}_nosph.npy'.format(G))
    homo_nosph_full_fission_density = np.sum(homo_nosph_full_sig_f * homo_nosph_full_phi, axis=0)
    homo_nosph_full_fission_density = np.sum(homo_nosph_full_fission_density.reshape((10,-1)), axis=1)

    fd_ref = ref_fission_density / np.linalg.norm(ref_fission_density)
    fd_homo = homo_fission_density / np.linalg.norm(homo_fission_density)
    fd_homo_nosph = homo_nosph_fission_density / np.linalg.norm(homo_nosph_fission_density)
    fd_homo_full = homo_full_fission_density / np.linalg.norm(homo_full_fission_density)
    fd_homo_nosph_full = homo_full_fission_density / np.linalg.norm(homo_nosph_full_fission_density)
    err_homo = np.abs(fd_homo - fd_ref) / fd_ref * 100
    err_homo_nosph = np.abs(fd_homo_nosph - fd_ref) / fd_ref * 100
    err_homo_full = np.abs(fd_homo_full - fd_ref) / fd_ref * 100
    err_homo_nosph_full = np.abs(fd_homo_nosph_full - fd_ref) / fd_ref * 100

    MO = dgmstructure.maxOrder

    plt.figure(figsize=(6.4,4.8))
    for i, basis in enumerate(bases):

        dgm_err = np.zeros((10, MO))
        dof = []

        for o in range(dgmstructure.maxOrder):
            dof.append(sum([min(o + 1, orders) for cg, orders in dgmstructure.counts.items()]))
            # DGM values
            # DGM values
            try:
                dgm_sig_f = np.load('data/dgm_sig_f_{}_{}_{}_o{}_h{}.npy'.format(G, basis, method, o, homogOption))
                dgm_phi = np.load('data/dgm_phi_{}_{}_{}_o{}_h{}.npy'.format(G, basis, method, o, homogOption))
            except FileNotFoundError:
                print('missing data/dgm_phi_{}_{}_{}_o{}_h{}.npy'.format(G, basis, method, o, homogOption))
                dgm_sig_f = np.zeros(homo_sig_f.shape)
                dgm_phi = np.zeros(homo_phi.shape)
            dgm_fission_density = np.sum(np.sum(dgm_sig_f * dgm_phi, axis=0), axis=0)
            dgm_fission_density = np.sum(dgm_fission_density.reshape((10,-1)), axis=1)
            fd_dgm = dgm_fission_density / np.linalg.norm(dgm_fission_density)
            dgm_err[:, o] = np.abs(fd_dgm - fd_ref) / fd_ref * 100

        plt.semilogy(dof, np.max(dgm_err, axis=0), style[i], label=bName[i])
    plt.semilogy(dof, np.ones(len(dof)) * np.max(err_homo, axis=0), color='b', ls=':', label='Standard SPH')
    plt.semilogy(dof, np.ones(len(dof)) * np.max(err_homo_nosph, axis=0), color='b', ls='--', label='Standard No SPH')
    plt.semilogy(dof, np.ones(len(dof)) * np.max(err_homo_full, axis=0), color='k', ls=':', label='Spatial SPH')
    plt.semilogy(dof, np.ones(len(dof)) * np.max(err_homo_nosph_full, axis=0), color='k', ls='--', label='Spatial No SPH')
    plt.ylim([1e-1, 1e2])
    plt.legend(loc=0, ncol=3)
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Max Rel FD error [%]')
    plt.savefig('{}/DGM_SPH_FD_err_{}_{}_c{}_h{}.pdf'.format(plot_path, G, method, contig, homogOption))
    plt.clf()

def makeTable(G, methods, basis, homogOption, contig, order):
    # Reference values
    ref_sig_f = np.load('data/ref_sig_f_{}.npy'.format(G))
    ref_phi = np.load('data/ref_phi_{}.npy'.format(G))
    ref_fission_density = np.sum(ref_sig_f * ref_phi, axis=0)
    ref_fission_density = np.sum(ref_fission_density.reshape((10,-1)), axis=1)

    # Non DGM values
    homo_sig_f = np.load('data/homo_sig_f_{}.npy'.format(G))
    homo_phi = np.load('data/homo_phi_{}.npy'.format(G))
    homo_fission_density = np.sum(homo_sig_f * homo_phi, axis=0)
    homo_fission_density = np.sum(homo_fission_density.reshape((10,-1)), axis=1)

    homo_nosph_sig_f = np.load('data/homo_sig_f_{}_nosph.npy'.format(G))
    homo_nosph_phi = np.load('data/homo_phi_{}_nosph.npy'.format(G))
    homo_nosph_fission_density = np.sum(homo_nosph_sig_f * homo_nosph_phi, axis=0)
    homo_nosph_fission_density = np.sum(homo_nosph_fission_density.reshape((10,-1)), axis=1)

    # Non DGM values
    homo_full_sig_f = np.load('data/homo_full_sig_f_{}.npy'.format(G))
    homo_full_phi = np.load('data/homo_full_phi_{}.npy'.format(G))
    homo_full_fission_density = np.sum(homo_full_sig_f * homo_full_phi, axis=0)
    homo_full_fission_density = np.sum(homo_full_fission_density.reshape((10,-1)), axis=1)

    homo_nosph_full_sig_f = np.load('data/homo_full_sig_f_{}_nosph.npy'.format(G))
    homo_nosph_full_phi = np.load('data/homo_full_phi_{}_nosph.npy'.format(G))
    homo_nosph_full_fission_density = np.sum(homo_nosph_full_sig_f * homo_nosph_full_phi, axis=0)
    homo_nosph_full_fission_density = np.sum(homo_nosph_full_fission_density.reshape((10,-1)), axis=1)

    fd_ref = ref_fission_density / np.linalg.norm(ref_fission_density)
    fd_homo = homo_fission_density / np.linalg.norm(homo_fission_density)
    fd_homo_nosph = homo_nosph_fission_density / np.linalg.norm(homo_nosph_fission_density)
    fd_homo_full = homo_full_fission_density / np.linalg.norm(homo_full_fission_density)
    fd_homo_nosph_full = homo_full_fission_density / np.linalg.norm(homo_nosph_full_fission_density)
    err_homo = (fd_homo - fd_ref) / fd_ref * 100
    err_homo_nosph = (fd_homo_nosph - fd_ref) / fd_ref * 100
    err_homo_full = (fd_homo_full - fd_ref) / fd_ref * 100
    err_homo_nosph_full = (fd_homo_nosph_full - fd_ref) / fd_ref * 100

    MO = dgmstructure.maxOrder

    data = np.zeros((len(methods) + 5, len(ref_fission_density)))
    data[0] = fd_ref
    data[1] = err_homo
    data[2] = err_homo_nosph
    data[3] = err_homo_full
    data[4] = err_homo_nosph_full

    o=2
    for i, method in enumerate(methods):
        # DGM values
        try:
            dgm_sig_f = np.load('data/dgm_sig_f_{}_{}_{}_o{}_h{}.npy'.format(G, basis, method, o, homogOption))
            dgm_phi = np.load('data/dgm_phi_{}_{}_{}_o{}_h{}.npy'.format(G, basis, method, o, homogOption))
        except FileNotFoundError:
            print('missing data/dgm_phi_{}_{}_{}_o{}_h{}.npy'.format(G, basis, method, o, homogOption))
            dgm_sig_f = np.zeros(homo_sig_f.shape)
            dgm_phi = np.zeros(homo_phi.shape)
        dgm_fission_density = np.sum(np.sum(dgm_sig_f * dgm_phi, axis=0), axis=0)
        dgm_fission_density = np.sum(dgm_fission_density.reshape((10,-1)), axis=1)
        fd_dgm = dgm_fission_density / np.linalg.norm(dgm_fission_density)
        dgm_err = (fd_dgm - fd_ref) / fd_ref * 100

        data[5 + i] = dgm_err

    np.set_printoptions(precision=3, suppress=True)

    s =  '\\begin{{tabular}}{{c|{}}}\n'.format('|'.join(['r']*len(data)))
    s += 'Pin & Ref & SE-$\\omega$ & SE & S-$\\omega$ & S &' + ' & '.join(mName) + ' \\\\\n'
    s += '\\hline\n'
    for i in range(len(data[0])):
        s += '{} & '.format(i) + ' & '.join(['{:5.3f}'.format(f) for f in data.T[i]]) + ' \\\\\n'
    s += '\\hline\n'
    s += '\\end{tabular}\n'

    print(s)

if __name__ == '__main__':

    G = 44
    plot_path = 'plots'
    bases = ['dlp', 'klt_full', 'klt_combine', 'klt_pins_full']
    bName = ['DLP', 'POD_full', 'POD_combine', 'POD_pins']
    methods = ['rxn_t_mu_zeroth', 'rxn_f_mu_zeroth', 'rxn_t_mu_all', 'rxn_f_mu_all', 'phi_mu_zeroth', 'phi_mu_all', 'fine_mu']
    mName = ['$R_t$ - $\omega_0$', '$R_f$ - $\omega_0$', '$R_t$ - $\omega$', '$R_f$ - $\omega$', '$\phi$ - $\omega_0$', '$\phi$ - $\omega$', '$\omega_g$']
    style = ['g^--', 'mv-', 'c>:', 'yp-.', 'rs:', 'bd-.', 'ko-']



    for homogOption in [0]:
        for contig in [1]:
            print('h={} contig={}'.format(homogOption, contig))
            dgmstructure = computeBounds(G, 'full', contig, 0.0, 1.3, 60)

            for b in bases:
                compare(G, methods, b, homogOption, contig)
            for method in methods:
                compare2(G, method, bases, homogOption, contig)

    makeTable(G, methods, 'klt_combine', 0, 1, 0)

