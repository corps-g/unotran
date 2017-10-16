program test_angle
  use control
  use angle

  implicit none

  integer :: l = 7, t1=1, t2=1, t3=1, t4=1, testCond
  double precision :: mu_test(7), wt_test(7)

  mu_test = [0.9862838086968123,0.9284348836635735,0.8272013150697650,0.6872929048116855,0.5152486363581541,0.3191123689278897,0.1080549487073437]
  wt_test = [0.0351194603317519,0.0801580871597602,0.1215185706879032,0.1572031671581935,0.1855383974779378,0.2051984637212956,0.2152638534631578]

  call initialize_control('test/reg_test_options', .true.)
  angle_order = 7
  angle_option = 1
  call initialize_angle()
  t1 = testCond(norm2(mu-mu_test) < 1e-12)
  t2 = testCond(norm2(wt-wt_test) < 1e-12)
  call finalize_angle()
  call finalize_control()

  mu_test = [0.974553956171379190, 0.87076559279969723, 0.70292257568869854, 0.50000000000000000, 0.29707742431130146, 0.12923440720030277, 0.025446043828620812]
  wt_test = [0.064742483084434838, 0.13985269574463835, 0.19091502525255952, 0.20897959183673470, 0.19091502525255952, 0.13985269574463835, 0.064742483084434838]
  
  call initialize_control('test/reg_test_options', .true.)
  angle_order = 7
  angle_option = 2
  call initialize_angle()
  t3 = testCond(norm2(mu-mu_test) < 1e-12)
  t4 = testCond(norm2(wt-wt_test) < 1e-12)
  call finalize_angle()
  call finalize_control()
  
  if (t1 == 0) then
    print *, 'angle: GL mu failed'
  else if (t2 == 0) then
    print *, 'angle: GL wt failed'
  else if (t3 == 0) then
    print *, 'angle: DGL mu failed'
  else if (t4 == 0) then
    print *, 'angle: DGL wt failed'
  else
    print *, 'all tests passed for angle'
  end if

end program test_angle

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
