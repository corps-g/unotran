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

  do c = 1, number_cells
    do a = 1, number_angles * 2
      do g = 1, number_groups
        if (g .eq. 1) then
          source(c,a,g) = 1.0
        else
          source(c,a,g) = 0.0
        end if
      end do
    end do
  end do
  
  error = 1.0
  norm = 0.0
  counter = 1
  do while (error .gt. 1e-6)
    call sweep()
    error = abs(norm - norm2(phi))
    norm = norm2(phi)
    print *, error, counter
    print *, phi
    counter = counter + 1
  end do
  print *, phi
  
end program main
