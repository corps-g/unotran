import pydgm
import numpy as np
np.set_printoptions(precision=16)

G = 1
geo = 3
bound = 'V'
solver = 'fixed'

# Set the variables
if geo == 0:
    pydgm.control.fine_mesh = [10]
    pydgm.control.coarse_mesh = [0.0, 10.0]
    pydgm.control.material_map = [1]
elif geo == 1:
    pydgm.control.fine_mesh = [3, 10, 3]
    pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
    pydgm.control.material_map = [5, 1, 5]
elif geo == 2:
    pydgm.control.fine_mesh = [3, 22, 3]
    pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
    pydgm.control.material_map = [1, 1, 1]
elif geo == 3:
    pydgm.control.fine_mesh = [200, 200, 200]
    pydgm.control.coarse_mesh = [0, 100, 200, 300]
    pydgm.control.material_map = [1, 2, 1]

pydgm.control.angle_order = 4
pydgm.control.angle_option = pydgm.angle.gl
if bound == 'V':
    pydgm.control.boundary_type = [0.0, 0.0]
elif bound == 'R':
    pydgm.control.boundary_type = [1.0, 1.0]

pydgm.control.allow_fission = False
pydgm.control.recon_print = True
pydgm.control.eigen_print = True
pydgm.control.outer_print = True
pydgm.control.inner_print = False
pydgm.control.recon_tolerance = 1e-14
pydgm.control.eigen_tolerance = 1e-14
pydgm.control.outer_tolerance = 1e-14
pydgm.control.inner_tolerance = 1e-14
pydgm.control.max_recon_iters = 1000
pydgm.control.max_eigen_iters = 1
pydgm.control.max_outer_iters = 10000
pydgm.control.max_inner_iters = 100
pydgm.control.lamb = 1.0
pydgm.control.use_dgm = False
pydgm.control.store_psi = True
pydgm.control.solver_type = solver.ljust(256)
pydgm.control.source_value = 0.0 if solver == 'eigen' else 1.0
pydgm.control.equation_type = 'DD'
pydgm.control.legendre_order = 1
pydgm.control.ignore_warnings = True
if G == 2:
    pydgm.control.xs_name = '2gXS.anlxs'.ljust(256)
    pydgm.control.dgm_basis_name = 'test/2gbasis'.ljust(256)
    pydgm.control.energy_group_map = [1]
elif G == 4:
    pydgm.control.xs_name = '4gXS.anlxs'.ljust(256)
    pydgm.control.dgm_basis_name = 'test/4gbasis'.ljust(256)
    pydgm.control.energy_group_map = [2]
elif G == 7:
    pydgm.control.xs_name = '7gXS.anlxs'.ljust(256)
    pydgm.control.dgm_basis_name = 'test/7gbasis'.ljust(256)
    pydgm.control.energy_group_map = [4]
elif G == 1:
    pydgm.control.xs_name = 'aniso.anlxs'.ljust(256)

if pydgm.control.use_dgm:
    # Initialize the dependancies
    pydgm.dgmsolver.initialize_dgmsolver()

    # Solve the problem
    pydgm.dgmsolver.dgmsolve()
else:
    # Initialize the dependancies
    pydgm.solver.initialize_solver()

    # Solve the problem
    pydgm.solver.solve()


# Print the output
print pydgm.state.phi
np.save('aniso_phi', pydgm.state.phi)
#print pydgm.state.phi.flatten('F').tolist()
#print pydgm.state.psi.flatten('F').tolist()
#print repr(pydgm.state.d_keff)
