program test_solver
  call test1()
  call test2()
  call test3()
  call test4()
  call test5()
  call test6()
end program test_solver

! Test against detran with vacuum conditions
subroutine test1()
  use control
  use solver

  implicit none
  
  ! initialize types
  integer :: testCond, t1
  double precision :: phi_test(7,28)
  
  ! Define problem parameters
  call initialize_control('test/reg_test_options', .true.)
  xs_name = 'test/testXS.anlxs'
  allow_fission = .true.
  
  call initialize_solver()

  phi_test = reshape([  2.43576516357,  4.58369841267,  1.5626711117,  1.31245374786,  1.12046360588,  &
                        0.867236739559, 0.595606769942, 2.47769600029, 4.77942918468,   1.71039215, &
                        1.45482285016, 1.2432932006, 1.00395695756, 0.752760077886,    2.5169315, &
                        4.97587877605, 1.84928362206, 1.58461198915, 1.35194171606,  1.118058716, &
                        0.810409774028, 2.59320903064, 5.23311939144, 1.97104981208,  1.694307587, &
                        1.44165108478, 1.20037776749, 0.813247000713, 2.70124816247,  5.529676584, &
                        2.07046672497, 1.78060787735, 1.51140559905, 1.25273492166, 0.8089175035, &
                        2.79641531407, 5.78666661987, 2.15462056117, 1.85286457556,  1.569442902, &
                        1.29575568046, 0.813378868173, 2.87941186529, 6.00774402332,  2.225613264, &
                        1.91334927054, 1.61778278393, 1.33135223233, 0.819884638625,   2.95082918, &
                        6.19580198153, 2.28505426287, 1.96372778133, 1.65788876716,  1.360806309, &
                        0.82640046716, 3.01116590643, 6.35316815516, 2.3341740765,  2.005220713, &
                        1.69082169384, 1.38498482157, 0.83227605712, 3.06083711902,  6.481707642, &
                        2.37390755792, 2.0387192711, 1.71734879522, 1.40447856797, 0.8372837595, &
                        3.10018046121, 6.58289016281, 2.40495581492, 2.06486841143,  1.738020841, &
                        1.4196914264, 0.841334574174, 3.12946077671, 6.6578387922,  2.427832061, &
                        2.08412599358, 1.75322625595, 1.43089755336, 0.844390210178,  3.148873662, &
                        6.70736606465, 2.44289487705, 2.09680407626, 1.76322843396,   1.43827772, &
                        0.846433375617, 3.15854807829, 6.73199979835, 2.4503712844,  2.103096645, &
                        1.76819052183, 1.44194181402, 0.847456401368, 3.15854807829,  6.731999798, &
                        2.4503712844, 2.10309664539, 1.76819052183, 1.44194181402, 0.8474564014, &
                        3.14887366178, 6.70736606465, 2.44289487705, 2.09680407626,  1.763228434, &
                        1.43827772036, 0.846433375617, 3.12946077671, 6.6578387922,  2.427832061, &
                        2.08412599358, 1.75322625595, 1.43089755336, 0.844390210178,  3.100180461, &
                        6.58289016281, 2.40495581492, 2.06486841143, 1.73802084073,  1.419691426, &
                        0.841334574174, 3.06083711902, 6.48170764214, 2.37390755792,  2.038719271, &
                        1.71734879522, 1.40447856797, 0.837283759523, 3.01116590643,  6.353168155, &
                        2.3341740765, 2.00522071328, 1.69082169384, 1.38498482157, 0.8322760571, &
                        2.95082917982, 6.19580198153, 2.28505426287, 1.96372778133,  1.657888767, &
                        1.36080630864, 0.82640046716, 2.87941186529, 6.00774402332,  2.225613264, &
                        1.91334927054, 1.61778278393, 1.33135223233, 0.819884638625,  2.796415314, &
                        5.78666661987, 2.15462056117, 1.85286457556, 1.56944290225,   1.29575568, &
                        0.813378868173, 2.70124816247, 5.52967658354, 2.07046672497,  1.780607877, &
                        1.51140559905, 1.25273492166, 0.808917503514, 2.59320903064,  5.233119391, &
                        1.97104981208, 1.69430758654, 1.44165108478, 1.20037776749, 0.8132470007, &
                        2.51693149995, 4.97587877605, 1.84928362206, 1.58461198915,  1.351941716, &
                        1.11805871638, 0.810409774028, 2.47769600029, 4.77942918468,   1.71039215, &
                        1.45482285016, 1.2432932006, 1.00395695756, 0.752760077886,  2.435765164, &
                        4.58369841267, 1.5626711117, 1.31245374786, 1.12046360588, 0.8672367396, &
                        0.595606769942 ],shape(phi_test))
                 
  call solve()

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-5)
  
  if (t1 == 0) then
    print *, 'solver: vacuum test failed'
  else
    print *, 'all tests passed for solver vacuum'
  end if

  call finalize_solver()
  call finalize_control()

end subroutine test1

! test against detran with reflective conditions
subroutine test2()
  use control
  use solver

  implicit none

  ! initialize types
  integer :: testCond, t1
  double precision :: phi_test(7,28)

  ! Define problem parameters
  call initialize_control('test/reg_test_options', .true.)
  xs_name = 'test/testXS.anlxs'
  boundary_type = [1.0, 1.0]
  material_map = [1, 1, 1]

  call initialize_solver()

  call solve()

  phi_test = reshape((/ 94.51265887,  106.66371692,   75.39710228,   17.95365148,&
                         6.2855009 ,    3.01584797,    1.21327705,   94.51265887,&
                       106.66371692,   75.39710228,   17.95365148,    6.2855009 ,&
                         3.01584797,    1.21327705,   94.51265887,  106.66371692,&
                        75.39710228,   17.95365148,    6.2855009 ,    3.01584797,&
                         1.21327705,   94.51265887,  106.66371692,   75.39710228,&
                        17.95365148,    6.2855009 ,    3.01584797,    1.21327705,&
                        94.51265887,  106.66371692,   75.39710228,   17.95365148,&
                         6.2855009 ,    3.01584797,    1.21327705,   94.51265887,&
                       106.66371692,   75.39710228,   17.95365148,    6.2855009 ,&
                         3.01584797,    1.21327705,   94.51265887,  106.66371692,&
                        75.39710228,   17.95365148,    6.2855009 ,    3.01584797,&
                         1.21327705,   94.51265887,  106.66371692,   75.39710228,&
                        17.95365148,    6.2855009 ,    3.01584797,    1.21327705,&
                        94.51265887,  106.66371692,   75.39710228,   17.95365148,&
                         6.2855009 ,    3.01584797,    1.21327705,   94.51265887,&
                       106.66371692,   75.39710228,   17.95365148,    6.2855009 ,&
                         3.01584797,    1.21327705,   94.51265887,  106.66371692,&
                        75.39710228,   17.95365148,    6.2855009 ,    3.01584797,&
                         1.21327705,   94.51265887,  106.66371692,   75.39710228,&
                        17.95365148,    6.2855009 ,    3.01584797,    1.21327705,&
                        94.51265887,  106.66371692,   75.39710228,   17.95365148,&
                         6.2855009 ,    3.01584797,    1.21327705,   94.51265887,&
                       106.66371692,   75.39710228,   17.95365148,    6.2855009 ,&
                         3.01584797,    1.21327705,   94.51265887,  106.66371692,&
                        75.39710228,   17.95365148,    6.2855009 ,    3.01584797,&
                         1.21327705,   94.51265887,  106.66371692,   75.39710228,&
                        17.95365148,    6.2855009 ,    3.01584797,    1.21327705,&
                        94.51265887,  106.66371692,   75.39710228,   17.95365148,&
                         6.2855009 ,    3.01584797,    1.21327705,   94.51265887,&
                       106.66371692,   75.39710228,   17.95365148,    6.2855009 ,&
                         3.01584797,    1.21327705,   94.51265887,  106.66371692,&
                        75.39710228,   17.95365148,    6.2855009 ,    3.01584797,&
                         1.21327705,   94.51265887,  106.66371692,   75.39710228,&
                        17.95365148,    6.2855009 ,    3.01584797,    1.21327705,&
                        94.51265887,  106.66371692,   75.39710228,   17.95365148,&
                         6.2855009 ,    3.01584797,    1.21327705,   94.51265887,&
                       106.66371692,   75.39710228,   17.95365148,    6.2855009 ,&
                         3.01584797,    1.21327705,   94.51265887,  106.66371692,&
                        75.39710228,   17.95365148,    6.2855009 ,    3.01584797,&
                         1.21327705,   94.51265887,  106.66371692,   75.39710228,&
                        17.95365148,    6.2855009 ,    3.01584797,    1.21327705,&
                        94.51265887,  106.66371692,   75.39710228,   17.95365148,&
                         6.2855009 ,    3.01584797,    1.21327705,   94.51265887,&
                       106.66371692,   75.39710228,   17.95365148,    6.2855009 ,&
                         3.01584797,    1.21327705,   94.51265887,  106.66371692,&
                        75.39710228,   17.95365148,    6.2855009 ,    3.01584797,&
                         1.21327705,   94.51265887,  106.66371692,   75.39710228,&
                        17.95365148,    6.2855009 ,    3.01584797,    1.21327705 &
                       /),shape(phi_test))

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-5))

  if (t1 == 0) then
    print *, 'solver: reflection test failed'
  else
    print *, 'all tests passed for solver reflect'
  end if

  call finalize_solver()
  call finalize_control

end subroutine test2

! Eigenvalue test with vacuum conditions
subroutine test3()
  use control
  use solver

  implicit none

  ! initialize types
  integer :: testCond, t1, t2
  double precision :: phi_test(7,10), keff_test

  ! Define problem parameters
  call initialize_control('test/eigen_test_options', .true.)
  source_value = 0.0
  xs_name = 'test/testXS.anlxs'
  boundary_type = [0.0, 0.0]
  outer_print = .true.

  call initialize_solver()

  keff_test = 0.034011456

  phi_test = reshape([0.000962111160119, 0.00507059397278, 0.000204973381238, 3.71770054415e-07,          0.0, &
                      0.0, 0.0, 0.00100474244796, 0.00538313118299, 0.0002228086194, &
                      4.10015457546e-07, 0.0, 0.0, 0.0, 0.001037208163, &
                      0.0056207710964, 0.000236346938805, 4.38985068221e-07, 0.0,          0.0, &
                      0.0, 0.00105908804368, 0.0057807505412, 0.000245450209592, 4.584355529e-07, &
                      0.0, 0.0, 0.0, 0.00107009913943, 0.005861208405, &
                      0.000250025289558, 4.68202305486e-07, 0.0, 0.0,          0.0, &
                      0.00107009913943, 0.0058612084051, 0.000250025289558, 4.68202305486e-07,          0.0, &
                      0.0, 0.0, 0.00105908804368, 0.0057807505412, 0.0002454502096, &
                      4.58435552887e-07, 0.0, 0.0, 0.0, 0.001037208163, &
                      0.0056207710964, 0.000236346938805, 4.38985068221e-07, 0.0,          0.0, &
                      0.0, 0.00100474244796, 0.00538313118299, 0.000222808619351, 4.100154575e-07, &
                      0.0, 0.0, 0.0, 0.000962111160119, 0.005070593973, &
                      0.000204973381238, 3.71770054415e-07, 0.0, 0.0,          0.0],shape(phi_test))

  call solve()

  print *, phi(0,:,:)

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-5)
  t2 = testCond(abs(d_keff - keff_test) < 1e-6)

  if (t1 == 0) then
    print *, 'solver: eigen vacuum solver phi failed'
  else if (t2 == 0) then
    print *, 'solver: eigen vacuum solver keff failed'
  else
    print *, 'all tests passed for eigen solver vacuum'
  end if

  call finalize_solver()
  call finalize_control()

end subroutine test3

! Eigenvalue test with reflective conditions for 1 group
subroutine test4()
  use control
  use solver

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3
  double precision :: phi_test, keff_test, psi_test

  ! Define problem parameters
  call initialize_control('test/eigen_test_options', .true.)

  call initialize_solver()

  call solve()

  keff_test = 0.714285714

  phi_test = 0.14285714
  psi_test = 0.07142857

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-6))
  t2 = testCond(all(abs(psi(:,:,:) - psi_test) < 1e-6))
  t3 = testCond(abs(d_keff - keff_test) < 1e-6)

  if (t1 == 0) then
    print *, 'solver: eigen 1g reflection solver phi failed'
  else if (t2 == 0) then
    print *, 'solver: eigen 1g reflection solver psi failed'
  else if (t3 == 0) then
    print *, 'solver: eigen 1g reflection solver keff failed'
  else
    print *, 'all tests passed for eigen 1g reflection solver'
  end if

  call finalize_solver()
  call finalize_control

end subroutine test4

! Eigenvalue test with reflective conditions for 2 groups
subroutine test5()
  use control
  use solver

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, g
  double precision :: phi_test(2), keff_test, psi_test(2)

  ! Define problem parameters
  call initialize_control('test/eigen_test_options', .true.)
  xs_name = 'test/2gXS.anlxs'

  call initialize_solver()

  call solve()

  keff_test = 0.840336134

  phi_test = [0.14285714, 0.02521008]
  psi_test = [0.07142857, 0.01260504]

  do g = 1, 2
    phi(0,g,:) = phi(0,g,:) - phi_test(g)
    psi(g,:,:) = psi(g,:,:) - psi_test(g)
  end do


  t1 = testCond(all(abs(phi) < 1e-6))
  t2 = testCond(all(abs(psi) < 1e-6))
  t3 = testCond(abs(d_keff - keff_test) < 1e-6)

  if (t1 == 0) then
    print *, 'solver: eigen 2g reflection solver phi failed'
  else if (t2 == 0) then
    print *, 'solver: eigen 2g reflection solver psi failed'
  else if (t3 == 0) then
    print *, 'solver: eigen 2g reflection solver keff failed'
  else
    print *, 'all tests passed for eigen 2g reflection solver'
  end if

  call finalize_solver()
  call finalize_control

end subroutine test5

! Eigenvalue test with reflective conditions
subroutine test6()
  use control
  use solver

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, g
  double precision :: phi_test(7), keff_test, psi_test(7)

  ! Define problem parameters
  call initialize_control('test/eigen_test_options', .true.)
  xs_name = 'test/testXS.anlxs'

  call initialize_solver()

  call solve()

  keff_test = 2.424946538

  phi_test = [2.02901618e-01, 2.69843872e-01, 1.22379825e-02, 5.83535529e-06, 0.0, 0.0, 0.0]
  psi_test = [1.01450809e-01, 1.34921936e-01, 6.11899126e-03, 2.91767764e-06, 0.0, 0.0, 0.0]

  do g = 1, 7
    phi(0,g,:) = phi(0,g,:) - phi_test(g)
    psi(g,:,:) = psi(g,:,:) - psi_test(g)
  end do


  t1 = testCond(all(abs(phi(0,:,:)) < 1e-6))
  t2 = testCond(all(abs(psi) < 1e-6))
  t3 = testCond(abs(d_keff - keff_test) < 1e-6)

  if (t1 == 0) then
    print *, 'solver: eigen 7g reflection solver phi failed'
  else if (t2 == 0) then
    print *, 'solver: eigen 7g reflection solver psi failed'
  else if (t3 == 0) then
    print *, 'solver: eigen 7g reflection solver keff failed'
  else
    print *, 'all tests passed for eigen 7g reflection solver'
  end if

  call finalize_solver()
  call finalize_control

end subroutine test6

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
