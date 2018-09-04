import numpy as np
import binascii
import h5py
import matplotlib.pyplot as plt
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

def makeDB(name):
    name1 = '{}_res.m'.format(name)
    f = open(name1, 'r').readlines()
    material_names = []
    db = {}
    db['filename'] = name
    for line in f:
        if 'GC_UNIVERSE_NAME' in line:
            key = line.split()[-2][1:-1]
            db[key] = {}
            material_names.append(key)
        else:
            if 'INF_TOT' in line:
                sig = np.array([float(i) for i in line.split()[6:-1]])
                db[key]['sig_t'] = sig[::2]
                db[key]['sig_t_err'] = sig[1::2]
            elif 'INF_NSF' in line:
                db[key]['vsig_f'] = np.array([float(i) for i in line.split()[6:-1:2]])
            elif 'INF_S' in line:
                if line[5] in '01234567':
                    n = line[5]
                    D = np.array([float(i) for i in line.split()[6:-1:2]])
                    db[key]['sig_s{}'.format(n)] = D.reshape((int(np.sqrt(len(D))),-1))
            elif 'INF_CHIT' in line:
                db[key]['chi'] = np.array([float(i) for i in line.split()[6:-1:2]])
            elif 'INF_FLX' in line:
                db[key]['phi'] = np.array([float(i) for i in line.split()[6:-1:2]])
            elif 'INF_KINF' in line:
                db[key]['k'] = line.split()[-3]
            elif 'MACRO_E' in line:
                db[key]['E'] = np.array([float(i) for i in line.split()[6:-1]])
            elif 'INF_INVV' in line:
                db[key]['vel'] = np.array([float(i) for i in line.split()[6:-1:2]]) ** -1
            elif 'INF_NUBAR' in line:
                db[key]['nu'] = np.array([float(i) for i in line.split()[6:-1:2]])
            elif 'INF_DIFFCOEF' in line:
                db[key]['D'] = np.array([float(i) for i in line.split()[6:-1:2]])
    db['name_order'] = material_names
    return db

def combineScattering(db, nLegendre):
    Ng = len(db['sig_s0'])
    D = np.zeros((nLegendre, Ng, Ng))
    for l in range(nLegendre):
        D[l,:,:] = db['sig_s{}'.format(l)]
        del db['sig_s{}'.format(l)]
    db['sig_s'] = D
    return db

def writeCrossSections(db, fuel=True, nLegendre=8):
    np.seterr(divide='ignore', invalid='ignore')
    s = ''
    #=======================================
    # CARD TYPE 04: BASIC PARAMETERS
    #=======================================
    s += '{} {} 1.0 0.0 0.602214179\n'.format(nLegendre, 1 if fuel else 0)
    # <NumLegendre> <DataPresent> <Energy_Fission> <Energy_Capture> <Gram_Atom_Weight> [Optional Temp]
    #=======================================
    # CARD TYPE 05: GROUP CROSS SECTIONS
    #=======================================
    sig_f = np.nan_to_num(db['vsig_f'] / db['nu'])
    for group in range(len(db['E'])-1):
        s += '{:14.12e} '.format(db['sig_t'][group])
        if fuel:
            s += '{:14.12e} '.format(sig_f[group])
            s += '{:14.12e} '.format(db['vsig_f'][group])
            s += '{:14.12e}\n'.format(db['chi'][group])
        else:
            s += '\n'
    db = combineScattering(db, nLegendre)
    M = db['sig_s']
    for sg in M:
        for sgg in sg:
            for sggl in sgg:
                s += '{:15.12e} '.format(sggl)
            if len(sgg) > 1:
                s += '\n'
    return s

def writeVoid(n, nLegendre=8):
    #=======================================
    # CARD TYPE 03: ISOTOPE/COMPOSITION NAME
    #=======================================
    s = 'Void\n'
    #=======================================
    # CARD TYPE 04: BASIC PARAMETERS
    #=======================================
    s += '{} 0 1.0 0.0 0.602214179\n'.format(nLegendre)
    # <NumLegendre> <DataPresent> <Energy_Fission> <Energy_Capture> <Gram_Atom_Weight> [Optional Temp]
    #=======================================
    # CARD TYPE 05: GROUP CROSS SECTIONS
    #=======================================
    for group in range(n):
        s += '{:14.12e}\n'.format(0.0)
    for l in range(nLegendre):
        for ss in range(n):
            for sss in range(n):
                s += '{:14.12e} '.format(0.0)
            s += '\n'
    return s

def makeFile(db, binary=0, nLegendre=8):
    keys = db['name_order']
    #=======================================
    # CARD TYPE 00: BASIC DATA
    #=======================================
    s =  '{} {} 0\n'.format(len(keys) + 1, len(db[keys[0]]['E']) - 1)
    #=======================================
    # CARD TYPE 01: GROUP BOUNDARIES
    #=======================================
    for e in db[keys[0]]['E']:
        s += '{:14.12e} '.format(e)
    s += '\n'
    #=======================================
    # CARD TYPE 02: GROUP VELOCITIES
    #=======================================
    for v in db[keys[0]]['vel']:
        s += '{:14.12e} '.format(v)
    s += '\n'
    for key in keys:
        #=======================================
        # CARD TYPE 03: ISOTOPE/COMPOSITION NAME
        #=======================================
        s += key + '\n'
        fuel = True
        s += writeCrossSections(db[key], fuel, nLegendre)
    s += writeVoid(len(db[keys[0]]['E'])-1, nLegendre)

    if binary == 0 or binary == 2:
        with open('{}.anlxs'.format(db['filename']), 'w') as f:
            f.write(s)
    if binary == 1 or binary == 2:
        with open('{}.xsb'.format(db['filename']), 'wb') as f:
            f.write(bytearray(binascii.a2b_qp(s)))

def makeHDF5(db):
    mats = ['material' + str(i) for i in range(6)]

    with h5py.File(db['filename'] + '.h5', 'w') as f:
        h = f.create_group('input')
        h['db_data'] = []
        h['dbl_data'] = []
        h['int_data'] = []
        h['str_data'] = []
        h['vec_dbl_data'] = []
        h['vec_int_data'] = []

        g = f.create_group('material')
        g.attrs['number_groups'] = 7
        g.attrs['number_materials'] = 6
        for i, name in enumerate(db['name_order']):
            chi = db[name]['chi'][:]
            D = db[name]['D'][:]
            sig_t = db[name]['sig_t']
            sig_s = db[name]['sig_s'][0]
            vsig_f = db[name]['vsig_f']
            sig_a = sig_t - np.sum(sig_s, axis=1)

            h = g.create_group('material' + str(i))
            h['chi'] = chi[:]
            h['diff_coef'] = D
            h['nu'] = np.ones(len(vsig_f))[:]
            h['sigma_a'] = sig_a[:]
            h['sigma_f'] = vsig_f[:]
            h['sigma_s'] = sig_s[:]
            h['sigma_t'] = sig_t[:]

def printout(name, ST):
    s = '{}_test = reshape((/'.format(name)
    for i, item in enumerate(ST):
        s += '{:10.8f}, '.format(item)
        if i % 7 == 6 and i != len(ST) - 1:
            s += '&\n'
            for j in range(18+len(name)):
                s += ' '
    s = s[:-2] + '/), &\n'
    for j in range(18+len(name)):
        s += ' '
    s += 'shape({}_test))'.format(name)

def printoutS(ST):
    s = '{sig_s_test = reshape((/'
    D = np.zeros((7,7,7))
    for m, sig_s_m in enumerate(ST):
        for g, sig_s_mg in enumerate(sig_s_m[0]):
            for gp, sig_s_mgp in enumerate(sig_s_mg):
                D[g,gp,m] = sig_s_mgp
    printout('sig_s', D.flatten('F').tolist())

    return

def printCrossSections(db):
    ''' Print cross sections for unotran unit tests'''
    np.set_printoptions(suppress=True, precision = 6)
    np.seterr(divide='ignore', invalid='ignore')
    keys = db['name_order']
    # sig_t
    ST = np.array([db[key]['sig_t'] for key in keys]).flatten().tolist()
    printout('sig_t', ST)
    # vsig_f
    ST = np.array([db[key]['vsig_f'] for key in keys]).flatten().tolist()
    printout('vsig_f', ST)
    # sig_f
    ST = np.nan_to_num(np.array([db[key]['vsig_f'] for key in keys]) / np.array([db[key]['nu'] for key in keys])).flatten().tolist()
    printout('sig_f', ST)
    # chi
    ST = np.array([db[key]['chi'] for key in keys]).flatten().tolist()
    printout('chi', ST)
    # sig_s
    ST = np.array([db[key]['sig_s'] for key in keys])#.flatten().tolist()
    printoutS(ST)

def barchart(x, y) :
    X = np.zeros(2 * len(y))
    Y = np.zeros(2 * len(y))
    for i in range(len(y)):
        X[2 * i] = x[i]
        X[2 * i + 1] = x[i + 1]
        Y[2 * i] = y[i]
        Y[2 * i + 1] = y[i]
    return X, Y

def plotXS(db):
    E = db['0']['E']
    sig = db['0']['sig_t']
    EE, SIG = barchart(E, sig)
    err = db['0']['sig_t_err']
    print err
    _, sig_min = barchart(E, sig - err)
    _, sig_max = barchart(E, sig + err)

    plt.plot(EE, SIG)
    plt.fill_between(EE, sig_min, sig_max, alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def main(name):
    db = makeDB(name)
    makeFile(db, 0)
    makeHDF5(db)
    #printCrossSections(db)
    plotXS(db)

if __name__ == '__main__':
    Gs = [2, 44, 238]
    names = ['uo2', 'moxlow', 'moxmid', 'moxhigh', 'BWR1', 'BWR2', 'BWR3']
    for G in Gs:
        for name in names:
            print G, name
            main('{}g/'.format(G) + name + '-{}.inp'.format(G))

