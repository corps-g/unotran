program test_dgmsolver

  use solver

  implicit none

  ! initialize types
  integer :: testCond, t1=1, t2=1
  double precision :: phi_test(0:0,2,1), psi_test(2,4,1)

  phi_test = reshape((/1.1420149990909008,0.37464706668551212&
                       /), shape(phi_test))

  psi_test = reshape((/0.81304488744042813,0.29884810509581583,1.31748796916478740,0.41507830480599001,&
                       1.31748796916478740,0.41507830480599001,0.81304488744042813,0.29884810509581583&
                       /), shape(psi_test))

  ! initialize the variables necessary to solve the problem
  call initialize_solver(fineMesh=[1], courseMesh=[0.0_8,1.0_8], materialMap=[1], fileName='2gXS.anlxs', &
                         store=.true., angle_order=2, angle_option=1, boundary=[0.0_8, 0.0_8], &
                         energyMap=[1], basisName='2gbasis')


  ! set source
  source(:,:,:) = 1.0

  ! set phi and psi to the converged solution
  phi = phi_test
  psi = psi_test

  ! compute the moments
  call compute_moments()

  call dgmsweep(1.0_8)

  t1 = testCond(norm2(phi - phi_test) .lt. 1e-6)
  t2 = testCond(norm2(psi - psi_test) .lt. 1e-6)

  if (t1 .eq. 0) then
    print *, 'DGM solver: phi failed'
  else if (t2 .eq. 0) then
    print *, 'DGM solver: psi failed'
  else
    print *, 'all tests passed for DGM solver'
  end if

end program test_dgmsolver

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
