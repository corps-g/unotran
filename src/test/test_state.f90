program test_state

  use material, only : create_material
  use angle, only : initialize_angle, initialize_polynomials
  use mesh, only : create_mesh
  use state

  implicit none

  ! initialize types
  integer :: fineMesh(1), materialMap(1), t1=1, t2=1, t3=1, t4=1, testCond
  double precision :: courseMesh(2), norm, error
  double precision :: phi_test(7,1), psi_test(7,4,1),source_test(7,4,1)
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
  call initialize_state(.false., 'dd')
  
  t1 = testCond(norm2(phi(0,:,:)-phi_test) .lt. 1e-7)
  
  t2 = testCond(norm2(source-source_test) .lt. 1e-7)
  
  call finalize_state()
  call initialize_state(.true., 'dd')
  
  t3 = testCond(norm2(psi-psi_test) .lt. 1e-7)
  
  if (t1 .eq. 0) then
    print *, 'state: phi initialization failed'
  else if (t2 .eq. 0) then
    print *, 'state: source initialization failed'
  else if (t3 .eq. 0) then
    print *, 'state: psi initialization failed'
  else
    print *, 'all tests passed for state'
  end if

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
