program test_state
  use control
  use material, only : create_material
  use angle, only : initialize_angle, initialize_polynomials
  use mesh, only : create_mesh
  use state

  implicit none

  ! initialize types
  integer :: fineMesh(1), materialMap(1), t1=1, t2=1, t3=1, t4=1, testCond
  double precision :: coarseMesh(2), norm, error, boundary(2)
  double precision :: phi_test(7,1), psi_test(7,4,1),source_test(7,4,1)
  
  phi_test = 1
  psi_test = 1
  source_test = 0

  call initialize_control('test/state_test_options', .true.)

  ! Make the mesh
  call create_mesh()
  
  ! Read the material cross sections
  call create_material()
  
  ! Create the cosines and angle space
  call initialize_angle()

  ! Create the set of polynomials used for the anisotropic expansion
  call initialize_polynomials(number_legendre)
    
  ! Create the state variable containers
  call initialize_state()
  
  t1 = testCond(norm2(phi(0,:,:)-phi_test) < 1e-12)
  
  t2 = testCond(norm2(source-source_test) < 1e-12)
  
  call finalize_state()

  store_psi = .true.

  call initialize_state()
  
  t3 = testCond(norm2(psi-psi_test) < 1e-12)
  
  if (t1 == 0) then
    print *, 'state: phi initialization failed'
  else if (t2 == 0) then
    print *, 'state: source initialization failed'
  else if (t3 == 0) then
    print *, 'state: psi initialization failed'
  else
    print *, 'all tests passed for state'
  end if

  call finalize_state()
  call finalize_control()

end program test_state

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
