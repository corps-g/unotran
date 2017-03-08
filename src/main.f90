program main
  use material, only : create_material, number_legendre, number_groups
  use angle, only : initialize_angle, p_leg, number_angles, initialize_polynomials
  use mesh, only : create_mesh, number_cells
  use state, only : initialize_state, phi, source
  use sweeper, only : sweep
  implicit none
  
  ! initialize types
  integer :: fineMesh(3), l, c, a, g, counter
  integer :: materialMap(3)
  double precision :: courseMesh(4), norm, error
  
  ! Define problem parameters
  character(len=10) :: filename = 'test.anlxs'
  fineMesh = [3, 22, 3]
  materialMap = [6,1,6]
  courseMesh = [0.0, 0.09, 1.17, 1.26]
  
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

  source = 1.0
  ! Source only in first group
  !source(:,:,1) = 1.0

  error = 1.0
  norm = 0.0
  counter = 1
  do while (error .gt. 1e-6)
    call sweep()
    error = abs(norm - norm2(phi))
    norm = norm2(phi)
    print *, error, counter
    counter = counter + 1
  end do
  print *, phi(:,0,:)
  
end program main
