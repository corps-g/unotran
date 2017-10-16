program test_sweeper
  use control
  use mesh, only : dx
  use angle, only : mu
  use state, only : d_psi, d_phi
  use solver, only : initialize_solver, finalize_solver
  use sweeper

  implicit none

  ! Test DD Equation

  integer :: testCond, t1, t2, t3, t4
  double precision :: Ps=0.0, incoming=0.0

  ! Define problem parameters
  call initialize_control('test/equation_test_options', .true.)

  call initialize_solver()

  call computeEQ(1.0_8, incoming, 1.0_8, dx(1) / (2 * abs(mu(1))), dx(1), mu(1), Ps)

  t1 = testCond(abs(incoming - 0.734680275209795978) < 1e-12)
  t2 = testCond(abs(Ps - 0.367340137604897989) < 1e-12)

  call finalize_solver()
  call finalize_control()

  ! Test SC Equation
  PS = 0.0
  incoming = 0.0

  ! Define problem parameters
  call initialize_control('test/equation_test_options', .true.)

  equation_type = 'SC'

  call initialize_solver()

  call computeEQ(1.0_8, incoming, 1.0_8, dx(1) / (2 * abs(mu(1))), dx(1), mu(1), Ps)

  t3 = testCond(abs(incoming - 0.686907416523104323) < 1e-12)
  t4 = testCond(abs(Ps - 0.408479080928661801) < 1e-12)

  if (t1 == 0) then
    print *, 'sweeper: DD test (outgoing) Failed'
  else if (t2 == 0) then
    print *, 'sweeper: DD test (cell) Failed'
  else if (t3 == 0) then
    print *, 'sweeper: SC test (outgoing) Failed'
  else if (t4 == 0) then
    print *, 'sweeper: SC test (cell) Failed'
  else
    print *, 'all tests passed for sweeper'
  end if

  call finalize_solver()
  call finalize_control()

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
