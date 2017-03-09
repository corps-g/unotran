program main
  !use material, only : create_material, number_legendre, number_groups
  !use angle, only : initialize_angle, p_leg, number_angles, initialize_polynomials
  !use mesh, only : create_mesh, number_cells
  !use state, only : initialize_state, phi, source
  !use sweeper, only : sweep
  use solver
  implicit none
  
  ! initialize types
  integer :: fineMesh(6), l, c, a, g, counter
  integer :: materialMap(6)
  double precision :: courseMesh(7), norm, error
  
  ! Define problem parameters
  character(len=10) :: filename = 'test.anlxs'
  fineMesh = [3, 22, 3, 3, 22, 3]
  materialMap = [6,1,6, 6, 4, 6]
  courseMesh = [0.0, 0.09, 1.17, 1.26, 1.35, 2.43, 2.52]
  
  call initialize_solver(fineMesh, courseMesh, materialMap,filename, 10, 1)

  !source = 1.0
  ! Source only in first group
  source(:,:,:) = 1.0

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
