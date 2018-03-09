import numpy as np

np.set_printoptions(precision=16, suppress=True)

def createMaterial():

    basis = np.loadtxt('basis')

    with open('test.anlxs', 'r') as f:
        number_materials, number_groups, debugFlag = [int(i) for i in f.readline().split()]
        ebounds = np.array(f.readline().split()).astype(float)
        velocity = np.array(f.readline().split()).astype(float)
        materialName = f.readline()
        number_legendre, data, eFiss, eCap, AW = [int(float(i)) for i in f.readline().split()]
        
        sig_t = np.zeros((number_materials, number_groups))
        sig_f = np.zeros((number_materials, number_groups))
        nu_sig_f = np.zeros((number_materials, number_groups))
        chi = np.zeros((number_materials, number_groups))
        sig_s = np.zeros((number_materials, number_groups, number_groups, number_legendre))
        
        for mat in range(number_materials):
            if mat != 0:
                materialName = f.readline()
                number_legendre, data, eFiss, eCap, AW = [int(float(i)) for i in f.readline().split()]
            for g in range(number_groups):
                if data == 1:
                    D = [float(i) for i in f.readline().split()]
                    sig_t[mat, g] = D[0]
                    sig_f[mat, g] = D[1]
                    nu_sig_f[mat, g] = D[2]
                    chi[mat, g] = D[3]
                else:
                    sig_t[mat, g] = float(f.readline())
                    sig_f[mat, g] = 0.0
                    nu_sig_f[mat, g] = 0.0
                    chi[mat, g] = 0.0
            for l in range(number_legendre):
                for g in range(number_groups):
                    D = [float(i) for i in f.readline().split()]
                    sig_s[mat, :, g, l] = D
                    
    # Get phi moments
    phi = np.ones(number_groups)
    phi_m = np.array([basis[:,0].dot(phi), basis[:,4].dot(phi)])
    
    # sig_t_moments
    sig_t_m = np.array([sig_t[0].dot(basis[:,0]) / 2, sig_t[0].dot(basis[:,4]) / 1.7320508075688776])    

    # delta_moments
    delta_m = np.array([basis[:,g].dot(sig_t[0] - sig_t_m[0 if g < 4 else 1]) / (phi_m[0 if g < 4 else 1]) for g in range(7)])
           
    # sig_s_moments 
    sig_s_m = np.zeros((4, 2, 2, 8))
         
    for order in range(4):
        for g in range(number_groups):
            cg = 0 if g < 4 else 1
            o = order + cg * 4
            for gp in range(number_groups):
                cgp = 0 if gp < 4 else 1
                b = basis[g, o] if o < 7 else 0.0
                #print('order={}, g={}, cg={}, o={}, gp={}, cgp={}, b={}'.format(order, g, cg, o, gp, cgp, b))
                for l in range(number_legendre):
                    sig_s_m[order, cg,cgp,l] += b * sig_s[0,g,gp,l] / phi_m[cgp]
    print([sig_s_m[order].flatten().tolist() for order in range(4)])
    # nu_sig_f_moments
    nu_sig_f_m = np.array([sum(nu_sig_f[0,:4]) / phi_m[0], sum(nu_sig_f[0,4:]) / phi_m[1]])

    # chi_moments
    chi_m = basis.T.dot(chi[0])
    print(sig_s_m)
            
createMaterial()
