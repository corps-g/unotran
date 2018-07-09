from __future__ import print_function
import pydgm
import numpy as np
np.set_printoptions(precision=4, linewidth=120)

def main():
    ref_k, ref_phi, ref_psi = run_reference()
    dgm_k, dgm_phi, dgm_psi = one_iter_dgm(ref_k, ref_phi, ref_psi)
    reg_k, reg_phi, reg_psi = one_iter(ref_k, ref_phi, ref_psi)
    
    print('Compare DGM to reference')
    print('k = {}, ref_k = {}, ratio = {}'.format(dgm_k, ref_k, dgm_k / ref_k))
    print('phi = {}'.format(dgm_phi))
    print('ratio = {}'.format(dgm_phi / ref_phi))
    
    print('Compare NON-DGM to reference')
    print('k = {}, ref_k = {}, ratio = {}'.format(reg_k, ref_k, reg_k / ref_k))
    print('phi = {}'.format(reg_phi))
    print('ratio = {}'.format(reg_phi / ref_phi))
    
    

def set_problem_variables(DGM=False):
    pydgm.control.solver_type = 'eigen'.ljust(256)
    pydgm.control.source_value = 0.0
    pydgm.control.allow_fission = True
    pydgm.control.xs_name = 'test/7gXS.anlxs'.ljust(256)
    pydgm.control.xs_name = 'pythonTools/makeXS/7g/7gXS.anlxs'.ljust(256)
    pydgm.control.fine_mesh = [2, 1, 2]
    pydgm.control.coarse_mesh = [0.0, 5.0, 6.0, 11.0]
    pydgm.control.material_map = [1, 5, 3] # UO2 | water | MOX
    pydgm.control.boundary_type = [0.0, 0.0]  # Vacuum | Vacuum
    pydgm.control.angle_order = 4
    pydgm.control.angle_option = pydgm.angle.gl
    pydgm.control.recon_print = 0
    pydgm.control.eigen_print = 0
    pydgm.control.outer_print = 0
    pydgm.control.inner_print = 0
    pydgm.control.recon_tolerance = 1e-15
    pydgm.control.eigen_tolerance = 1e-15
    pydgm.control.outer_tolerance = 1e-15
    pydgm.control.inner_tolerance = 1e-15
    pydgm.control.lamb = 1.0
    pydgm.control.store_psi = True
    pydgm.control.equation_type = 'DD'
    pydgm.control.legendre_order = 0
    pydgm.control.ignore_warnings = True
    pydgm.control.max_recon_iters = 1000
    pydgm.control.max_eigen_iters = 1000
    pydgm.control.max_outer_iters = 100
    pydgm.control.max_inner_iters = 100
    pydgm.control.use_dgm = DGM
    
def run_reference():
    # Run the reference problem
    set_problem_variables(False)

    # Initialize the dependancies
    pydgm.solver.initialize_solver()
    pydgm.solver.solve()

    # Save reference values
    ref_keff = pydgm.state.d_keff * 1
    ref_phi = pydgm.state.d_phi * 1
    ref_psi = pydgm.state.d_psi * 1
    
    # Clean up the problem
    pydgm.solver.finalize_solver()
    pydgm.control.finalize_control()
    
    return ref_keff, ref_phi, ref_psi
    
def one_iter_dgm(init_k=None, init_phi=None, init_psi=None):
    # Run the reference problem
    set_problem_variables(True)
    # DLP basis
    pydgm.control.dgm_basis_name = 'test/7gbasis'.ljust(256)
    pydgm.control.energy_group_map = [4]  # 0, 1, 2, 3 | 4, 5, 6
    # Delta basis
    pydgm.control.dgm_basis_name = 'test/7gdelta'.ljust(256)
    pydgm.control.energy_group_map = range(1, 7)  # 0 | 1 | 2 | 3 | 4 | 5 | 6
    
    pydgm.control.max_recon_iters = 1
    pydgm.control.max_eigen_iters = 1
    pydgm.control.max_outer_iters = 1
    pydgm.control.max_inner_iters = 1

    # Initialize the dependancies
    pydgm.dgmsolver.initialize_dgmsolver()
    
    # Set the initial values    
    if init_k is not None:
        pydgm.state.d_keff = init_k
    if init_phi is not None:
        pydgm.state.d_phi = init_phi
    if init_psi is not None:
        pydgm.state.d_psi = init_psi
        
    # Get the moments from the fluxes
    pydgm.dgmsolver.compute_flux_moments()
    # Load the initial moments into the mg solver
    pydgm.state.d_phi = pydgm.dgm.phi_m_zero
    pydgm.state.d_psi = pydgm.dgm.psi_m_zero
        
    # Compute the XS and other moments
    order = 0
    pydgm.dgmsolver.compute_incoming_flux(order, init_psi)
    pydgm.dgmsolver.compute_xs_moments()
    pydgm.dgmsolver.slice_xs_moments(order)
        
    # Solve the problem
    pydgm.dgmsolver.dgmsolve()

    # Save the values
    keff = pydgm.state.d_keff * 1
    phi = pydgm.state.d_phi * 1
    psi = pydgm.state.d_psi * 1
    
    # Clean up the problem
    pydgm.dgmsolver.finalize_dgmsolver()
    pydgm.control.finalize_control()
    
    return keff, phi, psi
    
def one_iter(init_k=None, init_phi=None, init_psi=None):
    # Run the reference problem
    set_problem_variables()
    pydgm.control.max_recon_iters = 1
    pydgm.control.max_eigen_iters = 1
    pydgm.control.max_outer_iters = 1
    pydgm.control.max_inner_iters = 1

    # Initialize the dependancies
    pydgm.solver.initialize_solver()
    
    # Set the initial values    
    if init_k is not None:
        pydgm.state.d_keff = init_k
    if init_phi is not None:
        pydgm.state.phi = init_phi
    if init_psi is not None:
        pydgm.state.psi = init_psi
        
    # Solve the problem
    pydgm.solver.solve()

    # Save the values
    keff = pydgm.state.d_keff * 1
    phi = pydgm.state.d_phi * 1
    psi = pydgm.state.d_psi * 1
    
    # Clean up the problem
    pydgm.solver.finalize_solver()
    pydgm.control.finalize_control()
    
    return keff, phi, psi
    
if __name__ == '__main__':
    main()
