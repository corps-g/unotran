import pydgm
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=16, linewidth=132)

class Pin(object):
    def __init__(self, m):
        assert 1 <= m <= 7
        self.mm = [7, 7, 7, 7, 7, 7, 7,
                   7, 7, 7, m, 7, 7, 7,
                   7, 7, m, m, m, 7, 7,
                   7, m, m, m, m, m, 7,
                   7, 7, m, m, m, 7, 7,
                   7, 7, 7, m, 7, 7, 7,
                   7, 7, 7, 7, 7, 7, 7]

        self.mm = np.array(self.mm).reshape(7, 7)

class Assembly(object):
    def __init__(self, assay_type):
        G = 4
        F = 5
        if assay_type == 0:
            self.mm = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       0,0,0,0,0,G,0,0,G,0,0,G,0,0,0,0,0,
                       0,0,0,G,0,0,0,0,0,0,0,0,0,G,0,0,0,
                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       0,0,G,0,0,G,0,0,G,0,0,G,0,0,G,0,0,
                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       0,0,G,0,0,G,0,0,F,0,0,G,0,0,G,0,0,
                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       0,0,G,0,0,G,0,0,G,0,0,G,0,0,G,0,0,
                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       0,0,0,G,0,0,0,0,0,0,0,0,0,G,0,0,0,
                       0,0,0,0,0,G,0,0,G,0,0,G,0,0,0,0,0,
                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif assay_type == 1:
            self.mm = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,
                       1,2,2,2,2,G,2,2,G,2,2,G,2,2,2,2,1,
                       1,2,2,G,2,3,3,3,3,3,3,3,2,G,2,2,1,
                       1,2,2,2,3,3,3,3,3,3,3,3,3,2,2,2,1,
                       1,2,G,3,3,G,3,3,G,3,3,G,3,3,G,2,1,
                       1,2,2,3,3,3,3,3,3,3,3,3,3,3,2,2,1,
                       1,2,2,3,3,3,3,3,3,3,3,3,3,3,2,2,1,
                       1,2,G,3,3,G,3,3,F,3,3,G,3,3,G,2,1,
                       1,2,2,3,3,3,3,3,3,3,3,3,3,3,2,2,1,
                       1,2,2,3,3,3,3,3,3,3,3,3,3,3,2,2,1,
                       1,2,G,3,3,G,3,3,G,3,3,G,3,3,G,2,1,
                       1,2,2,2,3,3,3,3,3,3,3,3,3,2,2,2,1,
                       1,2,2,G,2,3,3,3,3,3,3,3,2,G,2,2,1,
                       1,2,2,2,2,G,2,2,G,2,2,G,2,2,2,2,1,
                       1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        elif assay_type == 2:
            self.mm = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]

        mesh = np.zeros((17 * 7, 17 * 7)).astype(int)
        self.mm = np.array(self.mm).reshape(17, 17)
        for i in range(17):
            for j in range(17):
                mesh[7*i:7*(i+1), 7*j:7*(j+1)] = Pin(self.mm[i,j]+1).mm

        self.mm = mesh

class Core(object):
    def __init__(self):
        mmap = [0,1,2,
                1,0,2,
                2,2,2]
        mmap = np.array(mmap).reshape(3,3)

        self.mm = np.zeros((3 * 17 * 7, 3 * 17 * 7)).astype(int)

        for i in range(3):
            for j in range(3):
                self.mm[17*7*i:17*7*(i+1), 17*7*j:17*7*(j+1)] = Assembly(mmap[i,j]).mm

# Set the variables
x = np.array([0.0, 0.101497, 0.196658, 0.413329, 0.846671, 1.06334, 1.1585, 1.26])
cx = np.zeros(3 * 17 * 7 + 1)
for i in range(3 * 17):
    cx[7*i: 7 * (i+1)] = 1.26 * i + x[:7]
cx[-1] = 3 * 17 * x[-1]

pydgm.control.spatial_dimension = 2
pydgm.control.fine_mesh_x = np.ones(len(cx) - 1).astype(int)
pydgm.control.fine_mesh_y = np.ones(len(cx) - 1).astype(int)
pydgm.control.coarse_mesh_x = cx
pydgm.control.coarse_mesh_y = cx
pydgm.control.material_map = Core().mm.flatten()
# Set Quadrature points
pydgm.control.angle_order = 8
pydgm.control.angle_option = pydgm.angle.gl
# Set boundary conditions
pydgm.control.boundary_north = 1.0
pydgm.control.boundary_south = 0.0
pydgm.control.boundary_east = 0.0
pydgm.control.boundary_west = 1.0
# Set other options
pydgm.control.allow_fission = True
pydgm.control.eigen_print = 1
pydgm.control.outer_print = 0
pydgm.control.eigen_tolerance = 1e-12
pydgm.control.outer_tolerance = 1e-13
pydgm.control.max_eigen_iters = 100000
pydgm.control.max_outer_iters = 1
pydgm.control.store_psi = False
pydgm.control.solver_type = 'eigen'.ljust(256)
pydgm.control.source_value = 0.0
pydgm.control.equation_type = 'DD'
pydgm.control.scatter_leg_order = 0
pydgm.control.ignore_warnings = True
pydgm.control.xs_name = 'test/c5g7.anlxs'.ljust(256)

# Initialize the dependancies
pydgm.solver.initialize_solver()

sig_f = np.array([[0.00721206, 0.000819301, 0.0064532, 0.0185648, 0.0178084, 0.0830348, 0.216004], [0.00762704, 0.000876898, 0.00569835, 0.0228872, 0.0107635, 0.232757, 0.248968], [0.00825446, 0.00132565, 0.00842156, 0.032873, 0.0159636, 0.323794, 0.362803], [0.00867209, 0.00162426, 0.0102716, 0.0390447, 0.0192576, 0.374888, 0.430599], [4.79002e-09, 5.82564e-09, 4.63719e-07, 5.24406e-06, 1.4539e-07, 7.14972e-07, 2.08041e-06], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
nu = np.array([[2.78145, 2.47443, 2.43383, 2.4338, 2.4338, 2.4338, 2.4338], [2.85209, 2.89099, 2.85486, 2.86073, 2.85447, 2.86415, 2.8678], [2.88498, 2.91079, 2.86574, 2.87063, 2.86714, 2.86658, 2.87539], [2.90426, 2.91795, 2.86986, 2.87491, 2.87175, 2.86752, 2.87808], [2.76283, 2.46239, 2.4338, 2.4338, 2.4338, 2.4338, 2.4338], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

pydgm.material.nu_sig_f = (sig_f * nu).T

# Solve the problem
pydgm.solver.solve()

# Print the output

for g in range(7):
    plt.imshow(pydgm.state.phi[0,g].reshape(3*17*7,-1))
    plt.savefig('c5g7_phi_{}.png'.format(g))
    plt.clf()
print(repr(pydgm.state.keff))
np.save('c5g7_phi', pydgm.state.phi)
