program test_solver
  call vacuum1()
  call reflect1()
  call reflect2()
  call eigenV1g()
  call eigenV2g()
  call eigenV7g()
  call eigenR1g()
  call eigenR2g()
  call eigenR7g()
  call eigenR1gPin()
  call eigenR2gPin()
  call eigenR7gPin()
end program test_solver

! Test against detran with vacuum conditions
subroutine vacuum1()
  use control
  use solver
  use angle, only : wt

  implicit none
  
  ! initialize types
  integer :: testCond, t1, t2, a, c
  double precision :: phi_test(7,28), psi_test(7,20,28)
  
  ! Define problem parameters
  call initialize_control('test/reg_test_options', .true.)
  xs_name = 'test/testXS.anlxs'
  allow_fission = .true.
  
  call initialize_solver()

  phi_test = reshape([2.43576516357, 4.58369841267, 1.5626711117, 1.31245374786,  1.120463606, &
                      0.867236739559, 0.595606769943, 2.47769600028, 4.77942918468,   1.71039215, &
                      1.45482285016, 1.2432932006, 1.00395695756, 0.752760077886,    2.5169315, &
                      4.97587877605, 1.84928362206, 1.58461198915, 1.35194171606,  1.118058716, &
                      0.810409774027, 2.59320903064, 5.23311939144, 1.97104981208,  1.694307587, &
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
                      6.70736606465, 2.44289487705, 2.09680407625, 1.76322843396,   1.43827772, &
                      0.846433375616, 3.15854807829, 6.73199979835, 2.4503712844,  2.103096645, &
                      1.76819052183, 1.44194181402, 0.847456401367, 3.15854807829,  6.731999798, &
                      2.4503712844, 2.10309664539, 1.76819052183, 1.44194181402, 0.8474564014, &
                      3.14887366178, 6.70736606465, 2.44289487705, 2.09680407625,  1.763228434, &
                      1.43827772036, 0.846433375616, 3.12946077671, 6.6578387922,  2.427832061, &
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
                      1.11805871638, 0.810409774027, 2.47769600028, 4.77942918468,   1.71039215, &
                      1.45482285016, 1.2432932006, 1.00395695756, 0.752760077886,  2.435765164, &
                      4.58369841267, 1.5626711117, 1.31245374786, 1.12046360588, 0.8672367396, &
                      0.595606769943], shape(phi_test))

  call solve()

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-6)
  
  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  t2 = testCond(all(abs(phi_test(:,:) / phi(0,:,:) - 1.0) < 1e-5))

  if (t1 == 0) then
    print *, 'solver: vacuum test phi failed'
  else if (t2 == 0) then
    print *, 'solver: vacuum test psi failed'
  else
    print *, 'all tests passed for solver vacuum'
  end if

  call finalize_solver()
  call finalize_control()

end subroutine vacuum1

! test against detran with reflective conditions
subroutine reflect1()
  use control
  use solver
  use angle, only : wt

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, a, an, c
  double precision :: phi_test(7,28), psi_test(7, 20, 28)

  ! Define problem parameters
  call initialize_control('test/reg_test_options', .true.)
  xs_name = 'test/testXS.anlxs'
  boundary_type = [1.0, 1.0]
  material_map = [1, 1, 1]

  call initialize_solver()

  call solve()

  phi_test = reshape([94.5126588705, 106.66371692, 75.3971022838, 17.9536514809,  6.285500896, &
                      3.01584797355, 1.21327705483, 94.5126588705, 106.66371692,  75.39710228, &
                      17.9536514809, 6.28550089637, 3.01584797355, 1.21327705483,  94.51265887, &
                      106.66371692, 75.3971022838, 17.9536514809, 6.28550089637,  3.015847974, &
                      1.21327705483, 94.5126588705, 106.66371692, 75.3971022838,  17.95365148, &
                      6.28550089637, 3.01584797355, 1.21327705483, 94.5126588705,  106.6637169, &
                      75.3971022838, 17.9536514809, 6.28550089637, 3.01584797355,  1.213277055, &
                      94.5126588705, 106.66371692, 75.3971022838, 17.9536514809,  6.285500896, &
                      3.01584797355, 1.21327705483, 94.5126588705, 106.66371692,  75.39710228, &
                      17.9536514809, 6.28550089637, 3.01584797355, 1.21327705483,  94.51265887, &
                      106.66371692, 75.3971022838, 17.9536514809, 6.28550089637,  3.015847974, &
                      1.21327705483, 94.5126588705, 106.66371692, 75.3971022838,  17.95365148, &
                      6.28550089637, 3.01584797355, 1.21327705483, 94.5126588705,  106.6637169, &
                      75.3971022838, 17.9536514809, 6.28550089637, 3.01584797355,  1.213277055, &
                      94.5126588705, 106.66371692, 75.3971022838, 17.9536514809,  6.285500896, &
                      3.01584797355, 1.21327705483, 94.5126588705, 106.66371692,  75.39710228, &
                      17.9536514809, 6.28550089637, 3.01584797355, 1.21327705483,  94.51265887, &
                      106.66371692, 75.3971022838, 17.9536514809, 6.28550089637,  3.015847974, &
                      1.21327705483, 94.5126588705, 106.66371692, 75.3971022838,  17.95365148, &
                      6.28550089637, 3.01584797355, 1.21327705483, 94.5126588705,  106.6637169, &
                      75.3971022838, 17.9536514809, 6.28550089637, 3.01584797355,  1.213277055, &
                      94.5126588705, 106.66371692, 75.3971022838, 17.9536514809,  6.285500896, &
                      3.01584797355, 1.21327705483, 94.5126588705, 106.66371692,  75.39710228, &
                      17.9536514809, 6.28550089637, 3.01584797355, 1.21327705483,  94.51265887, &
                      106.66371692, 75.3971022838, 17.9536514809, 6.28550089637,  3.015847974, &
                      1.21327705483, 94.5126588705, 106.66371692, 75.3971022838,  17.95365148, &
                      6.28550089637, 3.01584797355, 1.21327705483, 94.5126588705,  106.6637169, &
                      75.3971022838, 17.9536514809, 6.28550089637, 3.01584797355,  1.213277055, &
                      94.5126588705, 106.66371692, 75.3971022838, 17.9536514809,  6.285500896, &
                      3.01584797355, 1.21327705483, 94.5126588705, 106.66371692,  75.39710228, &
                      17.9536514809, 6.28550089637, 3.01584797355, 1.21327705483,  94.51265887, &
                      106.66371692, 75.3971022838, 17.9536514809, 6.28550089637,  3.015847974, &
                      1.21327705483, 94.5126588705, 106.66371692, 75.3971022838,  17.95365148, &
                      6.28550089637, 3.01584797355, 1.21327705483, 94.5126588705,  106.6637169, &
                      75.3971022838, 17.9536514809, 6.28550089637, 3.01584797355,  1.213277055, &
                      94.5126588705, 106.66371692, 75.3971022838, 17.9536514809,  6.285500896, &
                      3.01584797355, 1.21327705483, 94.5126588705, 106.66371692,  75.39710228, &
                      17.9536514809, 6.28550089637, 3.01584797355, 1.21327705483,  94.51265887, &
                      106.66371692, 75.3971022838, 17.9536514809, 6.28550089637,  3.015847974, &
                      1.21327705483], shape(phi_test))

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-5))

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  t2 = testCond(all(abs(phi_test(:,:) / phi(0,:,:) - 1.0) < 1e-5))

  if (t1 == 0) then
    print *, 'solver: reflection test phi failed'
  else if (t2 == 0) then
    print *, 'solver: reflection test psi failed'
  else
    print *, 'all tests passed for solver reflect'
  end if

  call finalize_solver()
  call finalize_control

end subroutine reflect1

subroutine reflect2()
  use control
  use solver
  use angle, only : wt, p_leg

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, a, an, c
  double precision :: phi_test(2,10), psi_test(2, 4, 10)

  ! Define problem parameters
  call initialize_control('test/region_test_options', .true.)

  call initialize_solver()

  call solve()

  phi_test = reshape([3.24531157658, 2.63796538436, 3.19115570063, 2.58999578667,  3.078187077, &
                      2.48612684495, 2.89631103472, 2.29888415373, 2.62828488257,  1.901027709, &
                      2.32001662547, 1.4356296947, 2.07807385965, 1.17059736292,  1.931679428, &
                      1.05641889755, 1.84839685462, 1.00287398739, 1.81061886872, 0.9806778431], shape(phi_test))

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-5))

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) * psi(:,2*number_angles - a+1,c)
    end do
  end do
  t2 = testCond(all(abs(phi_test(:,:) / phi(0,:,:) - 1.0) < 1e-5))

  if (t1 == 0) then
    print *, 'solver2: reflection test phi failed'
  else if (t2 == 0) then
    print *, 'solver2: reflection test psi failed'
  else
    print *, 'all tests passed for solver reflect'
  end if

  call finalize_solver()
  call finalize_control()

end subroutine reflect2

! Eigenvalue test with vacuum conditions
subroutine eigenV1g()
  use control
  use solver
  use angle, only : wt

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, a, c
  double precision :: phi_test(1,10), psi_test(1,4,10), keff_test

  ! Define problem parameters
  call initialize_control('test/eigen_test_options', .true.)
  source_value = 0.0
  boundary_type = [0.0, 0.0]

  call initialize_solver()

  keff_test = 0.689359112

  phi_test = reshape([0.0591785175256, 0.110145339207, 0.149705182747, 0.177850799073,  0.192479273, &
                      0.192479272989, 0.177850799073, 0.149705182747, 0.110145339207, 0.05917851753],shape(phi_test))

  call solve()

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-6)

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 10) < 5e-4))

  t3 = testCond(abs(d_keff - keff_test) < 1e-6)

  if (t1 == 0) then
    print *, 'solver: eigen 1g vacuum solver phi failed'
  else if (t2 == 0) then
    print *, 'solver: eigen 1g vacuum solver psi failed'
  else if (t3 == 0) then
    print *, 'solver: eigen 1g vacuum solver keff failed'
  else
    print *, 'all tests passed for eigen 1g vacuum solver'
  end if

  call finalize_solver()
  call finalize_control()

end subroutine eigenV1g

! Eigenvalue test with vacuum conditions
subroutine eigenV2g()
  use control
  use solver
  use angle, only : wt

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, a, c
  double precision :: phi_test(2,10), psi_test(2,4,10), keff_test

  ! Define problem parameters
  call initialize_control('test/eigen_test_options', .true.)
  xs_name = 'test/2gXS.anlxs'
  source_value = 0.0
  boundary_type = [0.0, 0.0]

  call initialize_solver()

  keff_test = 0.809952323

  phi_test = reshape([0.0588274918942, 0.00985808742285, 0.109950141974, 0.0193478123294,  0.149799204, &
                      0.026176658006, 0.178131036674, 0.0312242163256, 0.192866360834, 0.03377131381, &
                      0.192866360834, 0.0337713138118, 0.178131036674, 0.0312242163256,  0.149799204, &
                      0.026176658006, 0.109950141974, 0.0193478123294, 0.0588274918942, 0.009858087423],shape(phi_test))

  call solve()

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-6)

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 20) < 5e-4))

  t3 = testCond(abs(d_keff - keff_test) < 1e-6)

  if (t1 == 0) then
    print *, 'solver: eigen 2g vacuum solver phi failed'
  else if (t2 == 0) then
    print *, 'solver: eigen 2g vacuum solver psi failed'
  else if (t3 == 0) then
    print *, 'solver: eigen 2g vacuum solver keff failed'
  else
    print *, 'all tests passed for eigen 2g vacuum solver'
  end if

  call finalize_solver()
  call finalize_control()

end subroutine eigenV2g

! Eigenvalue test with vacuum conditions
subroutine eigenV7g()
  use control
  use solver
  use angle, only : wt

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, a, c
  double precision :: phi_test(7,10), psi_test(7,4,10), keff_test

  ! Define problem parameters
  call initialize_control('test/eigen_test_options', .true.)
  xs_name = 'test/testXS.anlxs'
  source_value = 0.0
  boundary_type = [0.0, 0.0]

  call initialize_solver()

  keff_test = 0.434799909

  phi_test = reshape([0.405482638045, 0.779754084603, 0.0334300551874, 3.65103139747e-05,          0.0, &
                      0.0, 0.0, 0.544370812461, 1.11178775288, 0.05133797579, &
                      5.78689008784e-05, 0.0, 0.0, 0.0,  0.654146849, &
                      1.36528127489, 0.0647156539292, 7.33654247178e-05, 0.0,          0.0, &
                      0.0, 0.730006225009, 1.53741263546, 0.0737521345137, 8.37472143e-05, &
                      0.0, 0.0, 0.0, 0.768744495465,   1.62460022, &
                      0.0783274641844, 8.89907718817e-05, 0.0, 0.0,          0.0, &
                      0.768744495465, 1.62460022034, 0.0783274641844, 8.89907718817e-05,          0.0, &
                      0.0, 0.0, 0.730006225009, 1.53741263546, 0.07375213451, &
                      8.37472142959e-05, 0.0, 0.0, 0.0,  0.654146849, &
                      1.36528127489, 0.0647156539292, 7.33654247178e-05, 0.0,          0.0, &
                      0.0, 0.544370812461, 1.11178775288, 0.0513379757929, 5.786890088e-05, &
                      0.0, 0.0, 0.0, 0.405482638045, 0.7797540846, &
                      0.0334300551874, 3.65103139747e-05, 0.0, 0.0, 0.0],shape(phi_test))

  call solve()

  phi_test = phi_test / phi_test(1,1) * phi(0,1,1)

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-6)

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 70) < 7e-3))

  t3 = testCond(abs(d_keff - keff_test) < 1e-6)

  if (t1 == 0) then
    print *, 'solver: eigen 7g vacuum solver phi failed'
  else if (t2 == 0) then
    print *, 'solver: eigen 7g vacuum solver psi failed'
  else if (t3 == 0) then
    print *, 'solver: eigen 7g vacuum solver keff failed'
  else
    print *, 'all tests passed for eigen 7g vacuum solver'
  end if

  call finalize_solver()
  call finalize_control()

end subroutine eigenV7g

! Eigenvalue test with reflective conditions for 1 group
subroutine eigenR1g()
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

end subroutine eigenR1g

! Eigenvalue test with reflective conditions for 2 groups
subroutine eigenR2g()
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

end subroutine eigenR2g

! Eigenvalue test with reflective conditions
subroutine eigenR7g()
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

end subroutine eigenR7g

! Eigenvalue test with reflect conditions for a pin cell
subroutine eigenR1gPin()
  use control
  use solver
  use angle, only : wt, number_angles
  use mesh, only : number_cells

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, a, c, an
  double precision :: phi_test(1,16), psi_test(1,4,16), keff_test

  ! Define problem parameters
  call initialize_control('test/pin_test_options', .true.)

  call initialize_solver()

  keff_test = 0.682474204

  phi_test = reshape([0.133418376522, 0.133560750803, 0.133845902455, 0.134709753202, 0.1359152954, &
                      0.13679997221, 0.137380731077, 0.137668452106, 0.137668452106, 0.1373807311, &
                      0.13679997221, 0.135915295391, 0.134709753202, 0.133845902455, 0.1335607508, &
                      0.133418376521],shape(phi_test))

  call solve()

  phi_test = phi_test / phi_test(1,1) * phi(0,1,1)

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-6)

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 16) < 5e-4))

  t3 = testCond(abs(d_keff - keff_test) < 1e-6)

  if (t1 == 0) then
    print *, 'solver: eigen 1g reflect pin solver phi failed'
  else if (t2 == 0) then
    print *, 'solver: eigen 1g reflect pin solver psi failed'
  else if (t3 == 0) then
    print *, 'solver: eigen 1g reflect pin solver keff failed'
  else
    print *, 'all tests passed for eigen 1g reflect pin solver'
  end if

  call finalize_solver()
  call finalize_control()

end subroutine eigenR1gPin

! Eigenvalue test with reflect conditions for a pin cell
subroutine eigenR2gPin()
  use control
  use solver
  use angle, only : wt

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, a, c
  double precision :: phi_test(2,16), psi_test(2,4,16), keff_test

  ! Define problem parameters
  call initialize_control('test/pin_test_options', .true.)
  xs_name = 'test/2gXS.anlxs'

  call initialize_solver()

  keff_test = 0.841854685

  phi_test = reshape([0.133931831084, 0.0466324063141, 0.134075529413, 0.0455080808626, 0.1343633343, &
                      0.0432068414739, 0.13516513934, 0.0384434752118, 0.136157377422, 0.0332992956, &
                      0.136742846609, 0.0304645081033, 0.137069783633, 0.0289701995061, 0.1372163852, &
                      0.0283256746626, 0.137216385156, 0.0283256746626, 0.137069783633, 0.02897019951, &
                      0.136742846609, 0.0304645081033, 0.136157377422, 0.0332992956043, 0.1351651393, &
                      0.0384434752119, 0.134363334286, 0.0432068414741, 0.134075529413, 0.04550808086, &
                      0.133931831085, 0.0466324063142],shape(phi_test))

  call solve()

  phi_test = phi_test / phi_test(1,1) * phi(0,1,1)
  psi_test = psi_test / psi_test(1,1,1) * psi(1,1,1)

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-6)

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 32) < 5e-4))

  t3 = testCond(abs(d_keff - keff_test) < 1e-6)

  if (t1 == 0) then
    print *, 'solver: eigen 2g reflect pin solver phi failed'
  else if (t2 == 0) then
    print *, 'solver: eigen 2g reflect pin solver psi failed'
  else if (t3 == 0) then
    print *, 'solver: eigen 2g reflect pin solver keff failed'
  else
    print *, 'all tests passed for eigen 2g reflect pin solver'
  end if

  call finalize_solver()
  call finalize_control()

end subroutine eigenR2gPin

! Eigenvalue test with reflect conditions for a pin cell
subroutine eigenR7gPin()
  use control
  use solver
  use angle, only : wt

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, c, a
  double precision :: phi_test(7,16), psi_test(7,4,16), keff_test

  ! Define problem parameters
  call initialize_control('test/pin_test_options', .true.)
  xs_name = 'test/testXS.anlxs'
  material_map = [6, 1, 6]

  call initialize_solver()

  keff_test = 1.351545156

  phi_test = reshape([2.2650616717, 2.8369334989, 0.0787783060158, 7.04175236386e-05, 3.247808073e-11, &
                      2.42090423431e-13, 1.5855325733e-13, 2.26504648629, 2.83730954426, 0.07884130784, &
                      7.05024889238e-05, 3.23070059067e-11, 2.416262144e-13, 1.50689303232e-13,    2.2650161, &
                      2.83806173523, 0.0789674406093, 7.06726888056e-05, 3.19637591127e-11, 2.406809989e-13, &
                      1.33594887863e-13, 2.26484830727, 2.84044264348, 0.0792086051569, 7.098912578e-05, &
                      3.13235926512e-11, 2.39206404496e-13, 1.09840721921e-13, 2.26458581485,  2.843834874, &
                      0.0794963139764, 7.13605988661e-05, 3.05732041796e-11, 2.37613706246e-13, 8.964680919e-14, &
                      2.26439199657, 2.84637204739, 0.0797108850311, 7.16364816476e-05, 3.001907576e-11, &
                      2.36383833154e-13, 7.77296907704e-14, 2.26426423355, 2.84806017199, 0.07985335909, &
                      7.18191179175e-05, 2.96537913389e-11, 2.35550014439e-13, 7.10923697151e-14,  2.264200786, &
                      2.8489032388, 0.0799244247599, 7.1910052492e-05, 2.94724771287e-11, 2.351334627e-13, &
                      6.81268597471e-14, 2.26420078604, 2.8489032388, 0.0799244247599, 7.191005249e-05, &
                      2.9472717738e-11, 2.3514457477e-13, 6.81336094823e-14, 2.26426423355,  2.848060172, &
                      0.0798533590933, 7.18191179264e-05, 2.96545146531e-11, 2.35583516359e-13, 7.111336685e-14, &
                      2.26439199657, 2.84637204739, 0.0797108850311, 7.16364816625e-05, 3.002028626e-11, &
                      2.36440226704e-13, 7.77673697266e-14, 2.26458581485, 2.84383487412, 0.07949631398, &
                      7.1360598887e-05, 3.05749093972e-11, 2.37693842494e-13, 8.97059192338e-14,  2.264848307, &
                      2.84044264348, 0.0792086051571, 7.09891258041e-05, 3.13258032808e-11, 2.393115064e-13, &
                      1.09929579303e-13, 2.26501610046, 2.83806173523, 0.0789674406095, 7.067268884e-05, &
                      3.19663665349e-11, 2.40808670657e-13, 1.33719616187e-13, 2.26504648629,  2.837309544, &
                      0.0788413078367, 7.05024889589e-05, 3.23098965985e-11, 2.41773215344e-13, 1.508514105e-13, &
                      2.2650616717, 2.8369334989, 0.078778306016, 7.04175236773e-05, 3.248125734e-11, &
                      2.42256465053e-13, 1.58755730155e-13],shape(phi_test))

  call solve()

  phi_test = phi_test / phi_test(1,1) * phi(0,1,1)

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-6)

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 112) < 5e-4))

  t3 = testCond(abs(d_keff - keff_test) < 1e-6)

  if (t1 == 0) then
    print *, 'solver: eigen 7g reflect pin solver phi failed'
  else if (t2 == 0) then
    print *, 'solver: eigen 7g reflect pin solver psi failed'
  else if (t3 == 0) then
    print *, 'solver: eigen 7g reflect pin solver keff failed'
  else
    print *, 'all tests passed for eigen 7g reflect pin solver'
  end if

  call finalize_solver()
  call finalize_control()

end subroutine eigenR7gPin

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
