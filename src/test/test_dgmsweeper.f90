program test_sweeper
  call innerSolver()

end program test_sweeper

subroutine innerSolver()
  use control
  use dgmsweeper
  use material, only : create_material, number_legendre, number_groups, finalize_material
  use angle, only : initialize_angle, p_leg, number_angles, initialize_polynomials, finalize_angle
  use mesh, only : create_mesh, number_cells, finalize_mesh
  use state, only : initialize_state, phi, source, psi, finalize_state, output_state, d_keff
  use dgm, only : number_coarse_groups, initialize_moments, initialize_basis, finalize_moments, expansion_order, compute_source_moments

  implicit none

  integer :: i, c, a, t1, t2, t3, t4, testCond
  double precision, allocatable :: phi_m(:,:,:), psi_m(:,:,:), incoming(:,:)

  call initialize_control('test/eigen_test_options', .true.)
  xs_name = 'test/testXS.anlxs'
  source_value = 0.0
  boundary_type = [1.0, 1.0]
  legendre_order = 0
  dgm_basis_name = 'basis'
  use_DGM = .true.
  use_recondensation = .false.
  energy_group_map = [4]
  inner_print = .false.
  max_inner_iters = 500
  ignore_warnings = .true.

  ! initialize the mesh
  call create_mesh()
  ! read the material cross sections
  call create_material()
  ! initialize the angle quadrature
  call initialize_angle()
  ! get the basis vectors
  call initialize_polynomials(number_legendre)
  ! allocate the solutions variables
  call initialize_state()
  ! Initialize DGM moments
  call initialize_moments()
  call initialize_basis()
  call compute_source_moments()

  allocate(incoming(number_coarse_groups, number_angles))
  allocate(phi_m(0:number_legendre, number_coarse_groups, number_cells))
  allocate(psi_m(number_coarse_groups, number_angles * 2, number_cells))
  phi_m = 0
  psi_m = 0

  d_keff = 2.424946538
  do c=1, number_cells
    phi(0,:,c) = [2.9285415251, 3.8947396818, 0.1766345696, 8.42E-05, 0.0, 0.0, 0.0]
    do a=1, number_angles * 2
      psi(:,a,c) = phi(0,:,c)
    end do
  end do

  call compute_flux_moments()

  ! Order 0
  call getIncoming(0, incoming)
  call compute_xs_moments(order=0)
  call inner_solve(0, incoming, phi_m, psi_m)
  t1 = testCond(norm2(phi_m(0,:,1) - [3.5, 0.0]) < 1e-5)
  ! Order 1
  call getIncoming(1, incoming)
  call compute_xs_moments(order=1)
  call inner_solve(1, incoming, phi_m, psi_m)
  t2 = testCond(norm2(phi_m(0,:,1) - [-2.7958624554, 0.0]) < 1e-5)
  ! Order 2
  call getIncoming(2, incoming)
  call compute_xs_moments(order=2)
  call inner_solve(2, incoming, phi_m, psi_m)
  t3 = testCond(norm2(phi_m(0,:,1) - [-0.5713742514, 0.0]) < 1e-5)
  ! Order 3
  call getIncoming(3, incoming)
  call compute_xs_moments(order=3)
  call inner_solve(3, incoming, phi_m, psi_m)
  t4 = testCond(norm2(phi_m(0,:,1) - [1.8393577552, 0.0]) < 1e-5)

  if (t1 == 0) then
    print *, 'DGM sweeper order 0 failed'
  else if (t2 == 0) then
    print *, 'DGM sweeper order 1 failed'
  else if (t3 == 0) then
    print *, 'DGM sweeper order 2 failed'
  else if (t4 == 0) then
    print *, 'DGM sweeper order 3 failed'
  else
    print *, 'all tests passed for DGM sweeper'
  end if

  contains

  subroutine getIncoming(i, incoming)

    integer, intent(in) :: i
    double precision, intent(out) :: incoming(:,:)
    integer :: a, g, cg

    incoming = 0
    do a = 1, number_angles
      do g = 1, number_groups
        cg = energyMesh(g)
        ! Angular flux
        incoming(cg, a) = incoming(cg, a) +  basis(g, i) * psi(g, a, 1)
      end do
    end do

  end subroutine getIncoming

end subroutine innerSolver

integer function testCond(condition)
  logical, intent(in) :: condition
  if (condition) then
    write(*,"(A)",advance="no") '.'
    testCond = 1
  else
    write(*,"(A)",advance="no") 'F'
    testCond = 0
  end if

end function testCond
