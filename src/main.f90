program main
  use material, only : create_material, number_legendre
  use angle, only : initialize_angle, p_leg, number_angles, initialize_polynomials
  use mesh, only : create_mesh
  use state, only : initialize_state
  implicit none
  
  ! initialize types
  integer :: fineMesh(3), l
  integer :: materialMap(3)
  double precision :: courseMesh(4)
  
  ! Define problem parameters
  character(len=10) :: filename = 'test.anlxs'
  fineMesh = [3, 10, 3]
  materialMap = [6,1,6]
  courseMesh = [0.0, 1.0, 2.0, 3.0]
  
  ! Make the mesh
  call create_mesh(fineMesh, courseMesh, materialMap)
  
  ! Read the material cross sections
  call create_material(filename)
  
  ! Create the cosines and angle space
  call initialize_angle(10, 1)
  ! Create the set of polynomials used for the anisotropic expansion
  call initialize_polynomials(number_legendre)
    
  ! Create the state variable containers
  call initialize_state()

  
end program main
