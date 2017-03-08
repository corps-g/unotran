program test_sweeper

  use material, only: create_material, sig_s, sig_t, vsig_f, chi, number_groups, number_legendre
  use mesh, only: create_mesh, dx, number_cells, mMap
  use angle, only: initialize_angle, initialize_polynomials, number_angles, p_leg, wt, mu
  use state, only: initialize_state, phi, psi, source
  use sweeper

  implicit none

  ! initialize types
  integer :: fineMesh(1), materialMap(1), t1=1, t2=1, t3=1, t4=1, testCond
  double precision :: courseMesh(2), norm, error
  double precision :: psi_test(1,4,7),source_test(1,4,7),phi_test(1,0:7,7)
  ! Define problem parameters
  character(len=10) :: filename = 'test.anlxs'
  
  phi_test = 0
  psi_test = 0
  source_test = 0
  fineMesh = [1]
  materialMap = [1]
  courseMesh = [0.0, 1.0]
  
  ! Make the mesh
  call create_mesh(fineMesh, courseMesh, materialMap)
  
  ! Read the material cross sections
  call create_material(filename)
  
  ! Create the cosines and angle space
  call initialize_angle(2, 1)
  ! Create the set of polynomials used for the anisotropic expansion
  call initialize_polynomials(number_legendre)
    
  ! Create the state variable containers
  call initialize_state()
  
  phi = 1.0
  psi = 1.0
  source = 1.0
  
  t1 = testCond(number_cells .eq. 1)
  t1 = testCond(number_angles .eq. 2)
  t1 = testCond(number_groups .eq. 7)
  t1 = testCond(number_legendre .eq. 7)
  
  psi = 1
  
  if (t1 .eq. 0) then
    print *, 'sweeper: phistar update failed'
  else if (t2 .eq. 0) then
    print *, 'sweeper: source update failed'
  else if (t3 .eq. 0) then
    print *, 'sweeper: source initialization failed'
  else if (t4 .eq. 0) then
    print *, 'sweeper: phistar initialization failed'
  else
    print *, 'all tests passed for sweeper'
  end if

end program test_sweeper

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
