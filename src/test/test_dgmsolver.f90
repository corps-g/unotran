program test_dgmsolver

  use solver

  implicit none

  ! initialize types
  integer :: testCond, t1=1, t2=1
  double precision :: phi_test(0:7,7,1), psi_test(7,4,1)

  phi_test = reshape((/1.22317204e+00,   0.00000000e+00,  -1.83146441e-01,   6.93889390e-18,&
                       1.67334134e-17,   3.46944695e-18,   1.15120620e-01,   0.00000000e+00,&
                       2.51692701e+00,   0.00000000e+00,  -3.44437502e-01,  -1.38777878e-17,&
                       2.67648075e-17,   6.93889390e-18,   2.16503573e-01,   1.38777878e-17,&
                       1.17362029e+00,  -1.38777878e-17,  -1.44487042e-01,   6.93889390e-18,&
                       8.66786473e-18,   3.46944695e-18,   9.08204267e-02,  -6.93889390e-18,&
                       1.03618471e+00,   1.38777878e-17,  -1.21336492e-01,   0.00000000e+00,&
                       6.17941730e-18,   0.00000000e+00,   7.62686520e-02,  -6.93889390e-18,&
                       9.36353101e-01,   0.00000000e+00,  -1.08537817e-01,  -6.93889390e-18,&
                       5.32192793e-18,   3.46944695e-18,   6.82237704e-02,   0.00000000e+00,&
                       8.90725516e-01,   0.00000000e+00,  -9.54244969e-02,   6.93889390e-18,&
                       3.21227858e-18,  -3.46944695e-18,   5.99811123e-02,   0.00000000e+00,&
                       3.97532152e-01,  -6.93889390e-18,  -2.84358664e-02,   0.00000000e+00,&
                      -1.91309015e-18,  -1.73472348e-18,   1.78739732e-02,   3.46944695e-18&
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
                         store=.true., angle_order=2, angle_option=1, energyMap=[4], basisName='basis')


  ! set source
  source = 1.0

  ! set phi and psi to the converged solution
  phi = phi_test
  psi = psi_test

  ! compute the moments
  call compute_moments()
  call dgmsweep(0.5_8)
  print *, phi - phi_test

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
