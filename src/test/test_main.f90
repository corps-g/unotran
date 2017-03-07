program test_sweeper
use material, only : create_material, number_legendre, number_groups
  use angle, only : initialize_angle, p_leg, number_angles, initialize_polynomials
  use mesh, only : create_mesh, number_cells
  use state, only : initialize_state, phi, source
  use sweeper, only : sweep
  implicit none
  
  ! initialize types
  integer :: fineMesh(3), l, c, a, g, counter, testCond, t1
  integer :: materialMap(3)
  double precision :: courseMesh(4), norm, error, phi_test(28,7)
  
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
    counter = counter + 1
  end do
  
  phi_test = 0.0
  phi_test(:,1) = [0.11370736914050290,0.11370736914050290,0.11370736914050290,0.18196063754411643,&
                   0.18196063754411643,0.18196063754411643,0.18196063754411643,0.18196063754411643,&
                   0.18196063754411643,0.18196063754411643,0.18196063754411643,0.18196063754411643,&
                   0.18196063754411643,0.18196063754411643,0.18196063754411643,0.18196063754411643,&
                   0.18196063754411643,0.18196063754411643,0.18196063754411643,0.18196063754411643,&
                   0.18196063754411643,0.18196063754411643,0.18196063754411643,0.18196063754411643,&
                   0.18196063754411643,0.11370740631584182,0.11370740631584182,0.11370740631584182]
                   
  t1 = testCond(norm2(phi-phi_test) .lt. 1e-7)
  
  if (t1 .eq. 0) then
    print *, 'main: phi comparison failed'
  else
    print *, 'all tests passed for main'
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
