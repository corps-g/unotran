program test_dgmsolver
  call test1()
  call test2()
  call test3()
  call test4()
  call test5()
end program test_dgmsolver

! Test the 2 group dgm solution
subroutine test1()
  use dgmsolver

  implicit none

  ! initialize types
  integer :: testCond, t1=1, t2=1
  double precision :: phi_test(0:0,2,1), psi_test(2,4,1)
  double precision :: phi_new(0:0,2,1), psi_new(2,4,1)

  phi_test = reshape([1.1420149990909008,0.37464706668551212], shape(phi_test))

  psi_test = reshape([0.81304488744042813,0.29884810509581583,1.31748796916478740,0.41507830480599001,&
                      1.31748796916478740,0.41507830480599001,0.81304488744042813,0.29884810509581583&
                      ], shape(psi_test))

  ! initialize the variables necessary to solve the problem
  call initialize_dgmsolver(fineMesh=[1], courseMesh=[0.0_8,1.0_8], materialMap=[1], fileName='2gXS.anlxs', &
                         store=.true., angle_order=2, angle_option=1, boundary=[0.0_8, 0.0_8], &
                         energyMap=[1], basisName='2gbasis')


  ! set source
  source(:,:,:) = 1.0

  ! set phi and psi to the converged solution
  phi = phi_test
  psi = psi_test

  call sweep(phi_new, psi_new, incoming, 1e-8_8, .false., .false., 1.0_8)

  t1 = testCond(norm2(phi - phi_test) < 1e-6)
  t2 = testCond(norm2(psi - psi_test) < 1e-6)

  if (t1 == 0) then
    print *, 'DGM solver 1: phi failed'
  else if (t2 == 0) then
    print *, 'DGM solver 1: psi failed'
  else
    print *, 'all tests passed for DGM solver 1'
  end if

  call finalize_dgmsolver()

end subroutine test1

! Test against detran with vacuum conditions
subroutine test2()
  use dgmsolver

  implicit none

  ! initialize types
  integer :: fineMesh(3), materialMap(3), l, c, a, g, counter, testCond, t1, t2
  double precision :: courseMesh(4), norm, error, phi_test(7,28), boundary(2)

  ! Define problem parameters
  character(len=17) :: filename = 'test/testXS.anlxs'
  fineMesh = [3, 22, 3]
  materialMap = [6,1,6]
  courseMesh = [0.0, 0.09, 1.17, 1.26]
  boundary = [0.0, 0.0]

  call initialize_dgmsolver(fineMesh=fineMesh, courseMesh=courseMesh, materialMap=materialMap, fileName=filename, &
                         angle_order=10, angle_option=1, boundary=boundary, print_level=.false., &
                         energyMap=[4], basisName='basis')

  source = 1.0

  phi_test = reshape([  2.43576516357,  4.58369841267,  1.5626711117,  1.31245374786,  1.12046360588,  &
                        0.867236739559,  0.595606769942,  2.47769600029,  4.77942918468,  1.71039214967,  &
                        1.45482285016,  1.2432932006,  1.00395695756,  0.752760077886,  2.51693149995,  &
                        4.97587877605,  1.84928362206,  1.58461198915,  1.35194171606,  1.11805871638,  &
                        0.810409774028,  2.59320903064,  5.23311939144,  1.97104981208,  1.69430758654,  &
                        1.44165108478,  1.20037776749,  0.813247000713,  2.70124816247,  5.52967658354,  &
                        2.07046672497,  1.78060787735,  1.51140559905,  1.25273492166,  0.808917503514,  &
                        2.79641531407,  5.78666661987,  2.15462056117,  1.85286457556,  1.56944290225,  &
                        1.29575568046,  0.813378868173,  2.87941186529,  6.00774402332,  2.22561326404,  &
                        1.91334927054,  1.61778278393,  1.33135223233,  0.819884638625,  2.95082917982,  &
                        6.19580198153,  2.28505426287,  1.96372778133,  1.65788876716,  1.36080630864,  &
                        0.82640046716,  3.01116590643,  6.35316815516,  2.3341740765,  2.00522071328,  &
                        1.69082169384,  1.38498482157,  0.83227605712,  3.06083711902,  6.48170764214,  &
                        2.37390755792,  2.0387192711,  1.71734879522,  1.40447856797,  0.837283759523,  &
                        3.10018046121,  6.58289016281,  2.40495581492,  2.06486841143,  1.73802084073,  &
                        1.4196914264,  0.841334574174,  3.12946077671,  6.6578387922,  2.4278320614,  &
                        2.08412599358,  1.75322625595,  1.43089755336,  0.844390210178,  3.14887366178,  &
                        6.70736606465,  2.44289487705,  2.09680407626,  1.76322843396,  1.43827772036,  &
                        0.846433375617,  3.15854807829,  6.73199979835,  2.4503712844,  2.10309664539,  &
                        1.76819052183,  1.44194181402,  0.847456401368,  3.15854807829,  6.73199979835,  &
                        2.4503712844,  2.10309664539,  1.76819052183,  1.44194181402,  0.847456401368,  &
                        3.14887366178,  6.70736606465,  2.44289487705,  2.09680407626,  1.76322843396,  &
                        1.43827772036,  0.846433375617,  3.12946077671,  6.6578387922,  2.4278320614,  &
                        2.08412599358,  1.75322625595,  1.43089755336,  0.844390210178,  3.10018046121,  &
                        6.58289016281,  2.40495581492,  2.06486841143,  1.73802084073,  1.4196914264,  &
                        0.841334574174,  3.06083711902,  6.48170764214,  2.37390755792,  2.0387192711,  &
                        1.71734879522,  1.40447856797,  0.837283759523,  3.01116590643,  6.35316815516,  &
                        2.3341740765,  2.00522071328,  1.69082169384,  1.38498482157,  0.83227605712,  &
                        2.95082917982,  6.19580198153,  2.28505426287,  1.96372778133,  1.65788876716,  &
                        1.36080630864,  0.82640046716,  2.87941186529,  6.00774402332,  2.22561326404,  &
                        1.91334927054,  1.61778278393,  1.33135223233,  0.819884638625,  2.79641531407,  &
                        5.78666661987,  2.15462056117,  1.85286457556,  1.56944290225,  1.29575568046,  &
                        0.813378868173,  2.70124816247,  5.52967658354,  2.07046672497,  1.78060787735,  &
                        1.51140559905,  1.25273492166,  0.808917503514,  2.59320903064,  5.23311939144,  &
                        1.97104981208,  1.69430758654,  1.44165108478,  1.20037776749,  0.813247000713,  &
                        2.51693149995,  4.97587877605,  1.84928362206,  1.58461198915,  1.35194171606,  &
                        1.11805871638,  0.810409774028,  2.47769600029,  4.77942918468,  1.71039214967,  &
                        1.45482285016,  1.2432932006,  1.00395695756,  0.752760077886,  2.43576516357,  &
                        4.58369841267,  1.5626711117,  1.31245374786,  1.12046360588,  0.867236739559,  &
                        0.595606769942 ],shape(phi_test))

  phi(0,:,:) = phi_test

  call dgmsolve(1e-8_8, 1.0_8)

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-5)

  if (t1 == 0) then
    print *, 'dgmsolver: vacuum test 1 failed'
  else
    print *, 'all tests passed for dgmsolver vacuum 1'
  end if

  call finalize_dgmsolver()

end subroutine test2

! test against detran with reflective conditions
subroutine test3()
  use dgmsolver

  implicit none

  ! initialize types
  integer :: fineMesh(3), materialMap(3), l, c, a, g, counter, testCond, t1, t2
  double precision :: courseMesh(4), norm, error, phi_test(7,28), boundary(2)

  ! Define problem parameters
  character(len=17) :: filename = 'test/testXS.anlxs'
  fineMesh = [3, 22, 3]
  materialMap = [1,1,1]
  courseMesh = [0.0, 0.09, 1.17, 1.26]
  boundary = [1.0, 1.0]

  call initialize_dgmsolver(fineMesh=fineMesh, courseMesh=courseMesh, materialMap=materialMap, fileName=filename, &
                         angle_order=10, angle_option=1, boundary=boundary, print_level=.false., fission_option=.false.,&
                         energyMap=[4], basisName='basis')

  source = 1.0

  call dgmsolve(1e-8_8, 0.5_8)

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
    print *, 'dgmsolver: reflection test 1 failed'
  else
    print *, 'all tests passed for dgmsolver reflect 1'
  end if

  call finalize_dgmsolver()

end subroutine test3

! Test against detran with vacuum conditions and 1 spatial cell
subroutine test4()
  use dgmsolver

  implicit none

  ! initialize types
  integer :: fineMesh(1), materialMap(1), l, c, a, g, counter, testCond, t1, t2
  double precision :: courseMesh(2), norm, error, phi_test(7,1), boundary(2), psi_test(7,4,1)

  ! Define problem parameters
  character(len=17) :: filename = 'test/testXS.anlxs'
  fineMesh = [1]
  materialMap = [1]
  courseMesh = [0.0, 1.0]
  boundary = [0.0, 0.0]

  call initialize_dgmsolver(fineMesh=fineMesh, courseMesh=courseMesh, materialMap=materialMap, fileName=filename, &
                         angle_order=2, angle_option=1, boundary=boundary, print_level=.false., fission_option=.false., &
                         energyMap=[4], basisName='basis')

  phi_test = reshape([  1.1076512516190389,1.1095892550819531,1.0914913168898499,1.0358809957845283,&
                        0.93405352272848619,0.79552760081182894,0.48995843862242699 ],shape(phi_test))

  psi_test = reshape([ 0.62989551954274092,0.65696484059125337,0.68041606804080934,0.66450867626366705,&
                       0.60263096140806338,0.53438683380855967,0.38300537939872042,1.3624866129734592,&
                       1.3510195477616225,1.3107592451448140,1.2339713438183100,1.1108346317861364,&
                       0.93482033401344933,0.54700730201708170,1.3624866129734592,1.3510195477616225,&
                       1.3107592451448140,1.2339713438183100,1.1108346317861364,0.93482033401344933,&
                       0.54700730201708170,0.62989551954274092 ,0.65696484059125337,0.68041606804080934,&
                       0.66450867626366705,0.60263096140806338,0.53438683380855967,0.38300537939872042 ], shape(psi_test))

  phi(0,:,:) = phi_test
  psi(:,:,:) = psi_test
  source(:,:,:) = 1.0

  call dgmsolve(1e-8_8, 0.1_8)

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-5)

  if (t1 == 0) then
    print *, 'dgmsolver: vacuum test 2 failed'
  else
    print *, 'all tests passed for dgmsolver vacuum 2'
  end if

  call finalize_dgmsolver()

end subroutine test4

! test against detran with reflective conditions
subroutine test5()
  use dgmsolver

  implicit none

  ! initialize types
  integer :: fineMesh(1), materialMap(1), l, c, a, g, counter, testCond, t1, t2
  double precision :: courseMesh(2), norm, error, phi_test(7,1), psi_test(7,4,1), boundary(2)

  ! Define problem parameters
  character(len=17) :: filename = 'test/testXS.anlxs'
  fineMesh = [1]
  materialMap = [1]
  courseMesh = [0.0, 1.0]
  boundary = [1.0, 1.0]

  call initialize_dgmsolver(fineMesh=fineMesh, courseMesh=courseMesh, materialMap=materialMap, fileName=filename, &
                         angle_order=2, angle_option=1, boundary=boundary, print_level=.false., fission_option=.false.,&
                         energyMap=[4], basisName='basis')

  source = 1.0

  phi_test = reshape([ 94.51265887,  106.66371692,   75.39710228,   17.95365148,&
                        6.2855009 ,    3.01584797,    1.21327705],shape(phi_test))

  psi_test = reshape([94.512658438949273,106.66371642824166,75.397102078564259,17.953651480951333,&
                      6.2855008963667123,3.0158479735464110,1.2132770548341645,94.512658454844384,&
                      106.66371644092482,75.397102082192546,17.953651480951343,6.2855008963667141,&
                      3.0158479735464105,1.2132770548341649,94.512658454844384,106.66371644092482,&
                      75.397102082192546,17.953651480951343,6.2855008963667141,3.0158479735464105,&
                      1.2132770548341649,94.512658438949273,106.66371642824166,75.397102078564259,&
                      17.953651480951333,6.2855008963667123,3.0158479735464110,1.2132770548341645], shape(psi_test))

  phi(0,:,:) = phi_test
  psi(:,:,:) = psi_test
  source(:,:,:) = 1.0

  call dgmsolve(1e-8_8, 0.1_8)

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-5))

  if (t1 == 0) then
    print *, 'dgmsolver: reflection test 2 failed'
  else
    print *, 'all tests passed for dgmsolver reflect 2'
  end if

  call finalize_dgmsolver()

end subroutine test5

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
