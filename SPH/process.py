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

def compute_error(phi_name, sig_name, fd_ref, dgm=False):
    try:
        sig_f = np.load(sig_name)
        phi = np.load(phi_name)
        if dgm:
            fd_homo = np.sum(np.sum(sig_f * phi, axis=0), axis=0)
        else:
            fd_homo = np.sum(sig_f * phi, axis=0)
        fd_homo = np.sum(fd_homo.reshape((10,-1)), axis=1)
        fd_homo /= np.linalg.norm(fd_homo)
        return np.abs(fd_homo - fd_ref) / fd_ref * 100
    except FileNotFoundError:
        print('missing {}'.format(phi_name))
        return np.ones(fd_ref.shape) * np.inf


def compare(G, methods, basis, homogOption, contig):
    '''
    Create a plot that compares the different methods for a single basis
    '''

    # Reference values
    ref_sig_f = np.load('{}/ref_sig_f_{}.npy'.format(data_path, G))
    ref_phi = np.load('{}/ref_phi_{}.npy'.format(data_path, G))
    fd_ref = np.sum(ref_sig_f * ref_phi, axis=0)
    fd_ref = np.sum(fd_ref.reshape((10,-1)), axis=1)
    fd_ref /= np.linalg.norm(fd_ref)

    # Create a container for table data
    data = np.zeros((len(methods) + 7, len(fd_ref)))
    data[0] = fd_ref

    # Create a container for the errors for SPH corrected, non-DGM cases
    sph_err = np.zeros((10, dgmstructure.maxOrder))

    # Create a container for the errors for non-DGM cases
    non_err = np.zeros((10, dgmstructure.maxOrder))

    # Create a container for the degrees of freedom
    dof = []
    for o in range(dgmstructure.maxOrder):
        # Add the number of DOF for the current order
        dof.append(sum([min(o + 1, orders) for cg, orders in dgmstructure.counts.items()]))
        # Compute SPH error
        sph_err[:,o] = compute_error('{}/homo_phi_{}_o{}.npy'.format(data_path, G, o), '{}/homo_sig_f_{}_o{}.npy'.format(data_path, G, o), fd_ref)
        # Compute Non-SPH error
        non_err[:,o] = compute_error('{}/homo_phi_{}_nosph_o{}.npy'.format(data_path, G, o), '{}/homo_sig_f_{}_nosph_o{}.npy'.format(data_path, G, o), fd_ref)

    # Add data for the table
    data[1] = sph_err[:,0]
    data[2] = sph_err[:,2]
    data[3] = sph_err[:,-1]
    data[4] = non_err[:,0]
    data[5] = non_err[:,2]
    data[6] = non_err[:,-1]

    # Create the figure canvas
    plt.figure(figsize=(6.4,4.8))

    # Add plots for the sph corrected results
    plt.semilogy(dof, np.max(sph_err, axis=0), 'b:', label='SPH corrected')

    # Add plot for the non-sph corrected results
    plt.semilogy(dof, np.max(non_err, axis=0), 'b--', label='Non-SPH corrected')

    # Compute the DGM error for each method
    for i, method in enumerate(methods):
        # Initialize a continer for the error
        dgm_err = np.zeros((10, dgmstructure.maxOrder))

        for o in range(dgmstructure.maxOrder):
            # Compute the DMG error for order o
            phi_name = '{}/dgm_sig_f_{}_{}_{}_o{}_h{}.npy'.format(data_path, G, basis, method, o, homogOption)
            sig_name = '{}/dgm_phi_{}_{}_{}_o{}_h{}.npy'.format(data_path, G, basis, method, o, homogOption)
            dgm_err[:, o] = compute_error(phi_name, sig_name, fd_ref, dgm=True)
        # Add the method to the plot
        plt.semilogy(dof, np.max(dgm_err, axis=0), style[i], label=mName[i])

        # Add data for the table
        data[7 + i] = dgm_err[:,2]

    plt.ylim([1e-1, 1e2])
    plt.legend(loc=0, ncol=3, fancybox=True, framealpha=0.0)
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Max Rel FD error [%]')
    plt.savefig('{}/DGM_SPH_FD_err_{}_{}_c{}_h{}.pdf'.format(plot_path, G, basis, contig, homogOption), transparent=True)
    plt.clf()

    np.set_printoptions(precision=3, suppress=True)

    headers = ['Pin', 'Ref', 'SPH-0', 'SPH-2', 'SPH-Full', 'NON-0', 'NON-2', 'NON-Full']

    s =  '\\begin{{tabular}}{{c|{}}}\n'.format('|'.join(['r']*len(data)))
    s += ' & '.join(headers + mName) + ' \\\\\n'
    s += '\\hline\n'
    for i in range(len(data[0])):
        s += '{} & '.format(i) + ' & '.join(['{:5.3f}'.format(f) for f in data.T[i]]) + ' \\\\\n'
    s += '\\hline\n'
    s += '\\end{tabular}\n'

    if basis == 'klt_combine':
        print(s)


if __name__ == '__main__':

    G = 44
    for version in [1,2,3]:
        print('version = ', version)
        plot_path = 'plots{}'.format('' if version == 1 else str(version))
        data_path = 'data{}'.format('' if version == 1 else str(version))
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
                    print(b)
                    compare(G, methods, b, homogOption, contig)

