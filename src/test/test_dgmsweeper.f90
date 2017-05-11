program test_dgmsweeper

  use solver, only : initialize_solver
  use dgmsweeper

  implicit none

  ! initialize types
  integer :: testCond, t1=1, t2=1, o, a, an, amin, amax, astep
  logical :: octant
  double precision :: source_test(7,4), phi_test(0:7,7,1), psi_test(7,4,1), S(7,4), Q(1,7)

  source_test = reshape((/1.26182329,  1.6959436 ,  1.50956709,  1.50099137,  1.37433726,  1.33575837,  1.34199138,&
                          1.34257405,  1.76680291,  1.51043675,  1.50101764,  1.3745595 ,  1.33587244,  1.34058049,&
                          1.34257405,  1.76680291,  1.51043675,  1.50101764,  1.3745595 ,  1.33587244,  1.34058049,&
                          1.26182329,  1.69594359,  1.50956709,  1.50099137,  1.37433726,  1.33575837,  1.34199138 &
                          /), shape(source_test))

  phi_test = reshape((/1.22317204e+00,   1.00000000e-8,  -1.83146441e-01,   6.93889390e-18,&
                       1.00000000e-8,   3.46944695e-18,   1.15120620e-01,   1.00000000e-8,&
                       2.51692701e+00,   1.00000000e-8,  -3.44437502e-01,  -1.38777878e-17,&
                       1.00000000e-8,   6.93889390e-18,   2.16503573e-01,   1.38777878e-17,&
                       1.17362029e+00,  -1.38777878e-17,  -1.44487042e-01,   6.93889390e-18,&
                       1.00000000e-8,   3.46944695e-18,   9.08204267e-02,  -6.93889390e-18,&
                       1.03618471e+00,   1.38777878e-17,  -1.21336492e-01,   1.00000000e-8,&
                       1.00000000e-8,   1.00000000e-8,   7.62686520e-02,  -6.93889390e-18,&
                       9.36353101e-01,   1.00000000e-8,  -1.08537817e-01,  -6.93889390e-18,&
                       1.00000000e-8,   3.46944695e-18,   6.82237704e-02,   1.00000000e-8,&
                       8.90725516e-01,   1.00000000e-8,  -9.54244969e-02,   6.93889390e-18,&
                       1.00000000e-8,  -3.46944695e-18,   5.99811123e-02,   1.00000000e-8,&
                       3.97532152e-01,  -1.00000000e-8,  -2.84358664e-02,   1.00000000e-8,&
                       1.00000000e-8,  -1.00000000e-8,   1.78739732e-02,   3.46944695e-18&
                       /), shape(phi_test))

  psi_test = reshape((/0.66243843,  1.46237369,  0.73124892,  0.66469264,&
                       0.60404633,  0.59856738,  0.31047097,  1.52226789,&
                       3.07942675,  1.4095816 ,  1.23433893,  1.11360585,&
                       1.04656294,  0.44397067,  1.52226789,  3.07942675,&
                       1.4095816 ,  1.23433893,  1.11360585,  1.04656294,&
                       0.44397067,  0.66243843,  1.46237369,  0.73124892,&
                       0.66469264,  0.60404633,  0.59856738,  0.31047097&
                       /), shape(psi_test))

  ! initialize the variables necessary to solve the problem
  call initialize_solver(fineMesh=[1], courseMesh=[0.0_8,1.0_8], materialMap=[1], fileName='test.anlxs', &
                         store=.true., angle_order=2, angle_option=1, energyMap=[1,2,3,4,5,6], basisName='deltaBasis')


  ! set source
  source = 1.0

  ! set phi and psi to the converged solution
  phi = phi_test
  psi = psi_test

  ! compute the moments
  call compute_moments()

  do o = 1, 2  ! Sweep over octants
    ! Sweep in the correct direction in the octant
    octant = o .eq. 1
    amin = merge(1, number_angles, octant)
    amax = merge(number_angles, 1, octant)
    astep = merge(1, -1, octant)
    do a = amin, amax, astep
      an = merge(a, 2 * number_angles - a + 1, octant)
      Q = updateSource(1, an)
      S(:,an) = Q(1,:)
    end do
  end do

  t1 = testCond(norm2(S - source_test) .lt. 1e-6)

  if (t1 .eq. 0) then
    print *, 'DGM sweeper: update source failed'
  else
    print *, 'all tests passed for DGM sweeper'
  end if

end program test_dgmsweeper

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
