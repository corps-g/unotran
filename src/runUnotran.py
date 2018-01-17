import pydgm
import numpy as np
np.set_printoptions(precision=16)

G = 4
geo = 1
bound = 'R'

# Set the variables
if geo == 0:
    pydgm.control.fine_mesh = [10]
    pydgm.control.coarse_mesh = [0.0, 10.0]
    pydgm.control.material_map = [1]
elif geo == 1:
    pydgm.control.fine_mesh = [3, 10, 3]
    pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
    pydgm.control.material_map = [2, 1, 2]
pydgm.control.angle_order = 2
pydgm.control.angle_option = pydgm.angle.gl
if bound == 'V':
    pydgm.control.boundary_type = [0.0, 0.0]
elif bound == 'R':
    pydgm.control.boundary_type = [1.0, 1.0]
pydgm.control.allow_fission = True
pydgm.control.energy_group_map = [2]
pydgm.control.outer_print = False
pydgm.control.inner_print = False
pydgm.control.outer_tolerance = 1e-14
pydgm.control.inner_tolerance = 1e-14
pydgm.control.lamb = 1.0
pydgm.control.use_dgm = False
pydgm.control.store_psi = True
pydgm.control.solver_type = 'eigen'.ljust(256)
pydgm.control.source_value = 0.0
pydgm.control.equation_type = 'DD'
pydgm.control.legendre_order = 0
pydgm.control.ignore_warnings = True
if G == 2:
    pydgm.control.xs_name = 'test/2gXS.anlxs'.ljust(256)
    pydgm.control.dgm_basis_name = '2gbasis'.ljust(256)
    pydgm.control.energy_group_map = [1]
elif G == 4:
    pydgm.control.xs_name = 'test/4gXS.anlxs'.ljust(256)
    pydgm.control.dgm_basis_name = '4gbasis'.ljust(256)
    pydgm.control.energy_group_map = [2]
elif G == 7:
    pydgm.control.xs_name = 'test/testXS.anlxs'.ljust(256)
    pydgm.control.dgm_basis_name = 'basis'.ljust(256)
    pydgm.control.energy_group_map = [4]

# Initialize the dependancies
pydgm.solver.initialize_solver()

# Solve the problem
pydgm.solver.solve()

# Print the output
print pydgm.state.phi.flatten('F').tolist()
#print pydgm.state.psi.flatten('F').tolist()
print pydgm.state.d_keff
