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
  call anisotropic1g()
  call anisotropic2g()
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

  phi_test = reshape([&
                  2.4357651635750400, 4.5836984126707350, 1.5626711116986940, &
                  1.3124537478583550, 1.1204636058787650, &
                  0.8672367395585173, 0.5956067699430927, 2.4776960002851410, &
                  4.7794291846806920, 1.7103921496696810, &
                  1.4548228501607490, 1.2432932006010790, 1.0039569575643390, &
                  0.7527600778861741, 2.5169314999452900, &
                  4.9758787760487800, 1.8492836220562230, 1.5846119891495940, &
                  1.3519417160622780, 1.1180587163812740, &
                  0.8104097740272025, 2.5932090306445380, 5.2331193914375660, &
                  1.9710498120829970, 1.6943075865403320, &
                  1.4416510847758330, 1.2003777674906530, 0.8132470007127552, &
                  2.7012481624697990, 5.5296765835424770, &
                  2.0704667249744150, 1.7806078773493410, 1.5114055990508270, &
                  1.2527349216609840, 0.8089175035144761, &
                  2.7964153140653490, 5.7866666198688900, 2.1546205611683410, &
                  1.8528645755599590, 1.5694429022506580, &
                  1.2957556804601720, 0.8133788681729145, 2.8794118652886920, &
                  6.0077440233244780, 2.2256132640367530, &
                  1.9133492705416030, 1.6177827839331540, 1.3313522323268000, &
                  0.8198846386251999, 2.9508291798209730, &
                  6.1958019815287070, 2.2850542628717220, 1.9637277813254120, &
                  1.6578887671558740, 1.3608063086420320, &
                  0.8264004671599318, 3.0111659064328310, 6.3531681551614780, &
                  2.3341740765040160, 2.0052207132762490, &
                  1.6908216938360000, 1.3849848215662650, 0.8322760571196950, &
                  3.0608371190159760, 6.4817076421448410, &
                  2.3739075579243520, 2.0387192710962770, 1.7173487952234950, &
                  1.4044785679738320, 0.8372837595232706, &
                  3.1001804612054250, 6.5828901628062710, 2.4049558149231540, &
                  2.0648684114310010, 1.7380208407270610, &
                  1.4196914263971640, 0.8413345741736370, 3.1294607767084180, &
                  6.6578387922026070, 2.4278320614003590, &
                  2.0841259935753220, 1.7532262559536610, 1.4308975533609440, &
                  0.8443902101775597, 3.1488736617755260, &
                  6.7073660646536390, 2.4428948770531990, 2.0968040762550530, &
                  1.7632284339552100, 1.4382777203593370, &
                  0.8464333756161143, 3.1585480782896000, 6.7319997983506910, &
                  2.4503712844015810, 2.1030966453875370, &
                  1.7681905218280590, 1.4419418140173990, 0.8474564013673678, &
                  3.1585480782895990, 6.7319997983506910, &
                  2.4503712844015810, 2.1030966453875370, 1.7681905218280590, &
                  1.4419418140173990, 0.8474564013673679, &
                  3.1488736617755260, 6.7073660646536390, 2.4428948770532000, &
                  2.0968040762550530, 1.7632284339552100, &
                  1.4382777203593370, 0.8464333756161145, 3.1294607767084180, &
                  6.6578387922026070, 2.4278320614003590, &
                  2.0841259935753220, 1.7532262559536620, 1.4308975533609440, &
                  0.8443902101775596, 3.1001804612054250, &
                  6.5828901628062730, 2.4049558149231540, 2.0648684114310010, &
                  1.7380208407270610, 1.4196914263971640, &
                  0.8413345741736371, 3.0608371190159760, 6.4817076421448400, &
                  2.3739075579243530, 2.0387192710962770, &
                  1.7173487952234960, 1.4044785679738320, 0.8372837595232707, &
                  3.0111659064328290, 6.3531681551614780, &
                  2.3341740765040180, 2.0052207132762500, 1.6908216938360000, &
                  1.3849848215662650, 0.8322760571196950, &
                  2.9508291798209740, 6.1958019815287060, 2.2850542628717220, &
                  1.9637277813254120, 1.6578887671558740, &
                  1.3608063086420310, 0.8264004671599318, 2.8794118652886920, &
                  6.0077440233244780, 2.2256132640367540, &
                  1.9133492705416030, 1.6177827839331540, 1.3313522323268000, &
                  0.8198846386251998, 2.7964153140653500, &
                  5.7866666198688890, 2.1546205611683400, 1.8528645755599600, &
                  1.5694429022506580, 1.2957556804601720, &
                  0.8133788681729145, 2.7012481624697990, 5.5296765835424770, &
                  2.0704667249744150, 1.7806078773493410, &
                  1.5114055990508270, 1.2527349216609840, 0.8089175035144759, &
                  2.5932090306445380, 5.2331193914375680, &
                  1.9710498120829960, 1.6943075865403320, 1.4416510847758330, &
                  1.2003777674906520, 0.8132470007127554, &
                  2.5169314999452900, 4.9758787760487800, 1.8492836220562230, &
                  1.5846119891495940, 1.3519417160622780, &
                  1.1180587163812740, 0.8104097740272026, 2.4776960002851420, &
                  4.7794291846806920, 1.7103921496696810, &
                  1.4548228501607490, 1.2432932006010790, 1.0039569575643390, &
                  0.7527600778861746, 2.4357651635750400, &
                  4.5836984126707330, 1.5626711116986940, 1.3124537478583550, &
                  1.1204636058787650, 0.8672367395585174, &
                  0.5956067699430929], shape(phi_test))

  call solve()

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                          * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  t2 = testCond(all(abs(phi_test(:,:) / phi(0,:,:) - 1.0) < 1e-12))

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

  phi_test = reshape([&
                  94.512658870513917, 106.66371692003509, 75.397102283827067, &
                  17.953651480950320, 6.2855008963666350, &
                  3.0158479735463930, 1.2132770548341580, 94.512658870513917, &
                  106.66371692003509, 75.397102283827067, &
                  17.953651480950320, 6.2855008963666350, 3.0158479735463930, &
                  1.2132770548341580, 94.512658870513917, &
                  106.66371692003509, 75.397102283827067, 17.953651480950320, &
                  6.2855008963666350, 3.0158479735463930, &
                  1.2132770548341580, 94.512658870513917, 106.66371692003509, &
                  75.397102283827067, 17.953651480950320, &
                  6.2855008963666350, 3.0158479735463920, 1.2132770548341580, &
                  94.512658870513917, 106.66371692003509, &
                  75.397102283827067, 17.953651480950320, 6.2855008963666350, &
                  3.0158479735463920, 1.2132770548341580, &
                  94.512658870513917, 106.66371692003509, 75.397102283827067, &
                  17.953651480950320, 6.2855008963666350, &
                  3.0158479735463920, 1.2132770548341580, 94.512658870513917, &
                  106.66371692003509, 75.397102283827067, &
                  17.953651480950320, 6.2855008963666350, 3.0158479735463920, &
                  1.2132770548341580, 94.512658870513917, &
                  106.66371692003509, 75.397102283827067, 17.953651480950320, &
                  6.2855008963666350, 3.0158479735463920, &
                  1.2132770548341580, 94.512658870513917, 106.66371692003509, &
                  75.397102283827067, 17.953651480950320, &
                  6.2855008963666350, 3.0158479735463920, 1.2132770548341580, &
                  94.512658870513917, 106.66371692003509, &
                  75.397102283827067, 17.953651480950320, 6.2855008963666350, &
                  3.0158479735463920, 1.2132770548341580, &
                  94.512658870513917, 106.66371692003509, 75.397102283827067, &
                  17.953651480950320, 6.2855008963666350, &
                  3.0158479735463920, 1.2132770548341580, 94.512658870513917, &
                  106.66371692003509, 75.397102283827067, &
                  17.953651480950320, 6.2855008963666350, 3.0158479735463920, &
                  1.2132770548341580, 94.512658870513917, &
                  106.66371692003509, 75.397102283827067, 17.953651480950320, &
                  6.2855008963666350, 3.0158479735463920, &
                  1.2132770548341580, 94.512658870513917, 106.66371692003509, &
                  75.397102283827067, 17.953651480950320, &
                  6.2855008963666350, 3.0158479735463920, 1.2132770548341580, &
                  94.512658870513917, 106.66371692003509, &
                  75.397102283827067, 17.953651480950320, 6.2855008963666350, &
                  3.0158479735463920, 1.2132770548341580, &
                  94.512658870513917, 106.66371692003509, 75.397102283827067, &
                  17.953651480950320, 6.2855008963666350, &
                  3.0158479735463920, 1.2132770548341580, 94.512658870513917, &
                  106.66371692003509, 75.397102283827067, &
                  17.953651480950320, 6.2855008963666350, 3.0158479735463920, &
                  1.2132770548341580, 94.512658870513917, &
                  106.66371692003509, 75.397102283827067, 17.953651480950320, &
                  6.2855008963666350, 3.0158479735463920, &
                  1.2132770548341580, 94.512658870513917, 106.66371692003509, &
                  75.397102283827067, 17.953651480950320, &
                  6.2855008963666350, 3.0158479735463920, 1.2132770548341580, &
                  94.512658870513917, 106.66371692003509, &
                  75.397102283827067, 17.953651480950320, 6.2855008963666350, &
                  3.0158479735463920, 1.2132770548341580, &
                  94.512658870513917, 106.66371692003509, 75.397102283827067, &
                  17.953651480950320, 6.2855008963666350, &
                  3.0158479735463920, 1.2132770548341580, 94.512658870513917, &
                  106.66371692003509, 75.397102283827067, &
                  17.953651480950320, 6.2855008963666350, 3.0158479735463920, &
                  1.2132770548341580, 94.512658870513917, &
                  106.66371692003509, 75.397102283827067, 17.953651480950320, &
                  6.2855008963666350, 3.0158479735463920, &
                  1.2132770548341580, 94.512658870513917, 106.66371692003509, &
                  75.397102283827067, 17.953651480950320, &
                  6.2855008963666350, 3.0158479735463920, 1.2132770548341580, &
                  94.512658870513917, 106.66371692003509, &
                  75.397102283827067, 17.953651480950320, 6.2855008963666350, &
                  3.0158479735463920, 1.2132770548341580, &
                  94.512658870513917, 106.66371692003509, 75.397102283827067, &
                  17.953651480950320, 6.2855008963666350, &
                  3.0158479735463920, 1.2132770548341580, 94.512658870513917, &
                  106.66371692003509, 75.397102283827067, &
                  17.953651480950320, 6.2855008963666350, 3.0158479735463920, &
                  1.2132770548341580, 94.512658870513917, &
                  106.66371692003509, 75.397102283827067, 17.953651480950320, &
                  6.2855008963666350, 3.0158479735463920, &
                  1.2132770548341580], shape(phi_test))

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                          * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  t2 = testCond(all(abs(phi_test(:,:) / phi(0,:,:) - 1.0) < 1e-12))

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

  phi_test = reshape([&
                  3.2453115765816380, 2.6379653843582970, 3.1911557006288490, &
                  2.5899957866672410, 3.0781870772267670, &
                  2.4861268449530300, 2.8963110347254490, 2.2988841537287830, &
                  2.6282848825690060, 1.9010277088673150, &
                  2.3200166254673200, 1.4356296946991250, 2.0780738596493520, &
                  1.1705973629208930, 1.9316794281903030, &
                  1.0564188975526510, 1.8483968546234210, 1.0028739873869340, &
                  1.8106188687176360, 0.9806778431098238], shape(phi_test))

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                          * psi(:,2*number_angles - a+1,c)
    end do
  end do
  t2 = testCond(all(abs(phi_test(:,:) / phi(0,:,:) - 1.0) < 1e-12))

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
  legendre_order = 0

  call initialize_solver()

  keff_test = 0.689359111542

  phi_test = reshape([&
                0.05917851752472814, 0.1101453392055481, 0.1497051827466689, &
                0.1778507990738045, 0.1924792729907672, &
                0.1924792729907672, 0.1778507990738046, 0.1497051827466690, &
                0.1101453392055482, 0.05917851752472817],shape(phi_test))

  call solve()

  phi = phi / phi(0,1,1) * phi_test(1,1)

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                          * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 10) < 1e-12))

  t3 = testCond(abs(d_keff - keff_test) < 1e-12)

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

  keff_test = 0.809952323298

  phi_test = reshape([&
              0.05882749189352335, 0.009858087422743867, 0.1099501419733834, &
              0.01934781232927005, 0.1497992040255296, &
              0.02617665800594787, 0.1781310366743199, 0.03122421632573503, &
              0.1928663608358422, 0.03377131381204261, &
              0.1928663608358422, 0.03377131381204262, 0.1781310366743200, &
              0.03122421632573504, 0.1497992040255296, &
              0.02617665800594788, 0.1099501419733835, 0.01934781232927007, &
              0.05882749189352339, 0.009858087422743878],shape(phi_test))

  call solve()

  phi = phi / phi(0,1,1) * phi_test(1,1)

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                          * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 20) < 1e-12))

  t3 = testCond(abs(d_keff - keff_test) < 1e-12)

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

  keff_test = 0.434799909025

  phi_test = reshape([&
              0.4054826380458894, 0.7797540846047309, 0.03343005518884316, &
              3.651031406190026e-05, 0.0000000000000000, &
              0.0000000000000000, 0.0000000000000000, 0.5443708124629247, &
              1.1117877528804340, 0.05133797579519819, &
              5.786890101941508e-05, 0.0000000000000000, 0.0000000000000000, &
              0.0000000000000000, 0.6541468489653601, &
              1.3652812748897960, 0.06471565393213054, 7.336542490037297e-05, &
              0.0000000000000000, 0.0000000000000000, &
              0.0000000000000000, 0.7300062250116002, 1.5374126354668270, &
              0.07375213451710246, 8.374721450748013e-05, &
              0.0000000000000000, 0.0000000000000000, 0.0000000000000000, &
              0.7687444954671941, 1.6246002203441580, &
              0.07832746418803245, 8.899077210821326e-05, 0.0000000000000000, &
              0.0000000000000000, 0.0000000000000000, &
              0.7687444954671943, 1.6246002203441580, 0.07832746418803245, &
              8.899077210821326e-05, 0.0000000000000000, &
              0.0000000000000000, 0.0000000000000000, 0.7300062250116002, &
              1.5374126354668280, 0.07375213451710246, &
              8.374721450748013e-05, 0.0000000000000000, 0.0000000000000000, &
              0.0000000000000000, 0.6541468489653601, &
              1.3652812748897960, 0.06471565393213052, 7.336542490037296e-05, &
              0.0000000000000000, 0.0000000000000000, &
              0.0000000000000000, 0.5443708124629247, 1.1117877528804350, &
              0.05133797579519818, 5.786890101941509e-05, &
              0.0000000000000000, 0.0000000000000000, 0.0000000000000000, &
              0.4054826380458894, 0.7797540846047309, &
              0.03343005518884316, 3.651031406190026e-05, 0.0000000000000000, &
              0.0000000000000000, 0.0000000000000000],shape(phi_test))

  call solve()

  phi_test = phi_test / phi_test(1,1) * phi(0,1,1)
  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                          * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 70) < 1e-12))

  t3 = testCond(abs(d_keff - keff_test) < 1e-12)

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
  legendre_order = 0

  call initialize_solver()

  call solve()

  keff_test = 0.714285714286

  phi_test = 1.0
  psi_test = 1.0

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))
  t2 = testCond(all(abs(psi(:,:,:) - psi_test) < 1e-12))
  t3 = testCond(abs(d_keff - keff_test) < 1e-12)

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

  keff_test = 0.840336134454

  phi_test = [0.1428571428571399, 0.0252100840336125]
  psi_test = [0.07142857142856951, 0.0126050420168061]

  phi = phi / phi(0,1,1) * phi_test(1)
  psi = psi / psi(1,1,1) * psi_test(1)

  do g = 1, 2
    phi(0,g,:) = phi(0,g,:) - phi_test(g)
    psi(g,:,:) = psi(g,:,:) - psi_test(g)
  end do


  t1 = testCond(all(abs(phi) < 1e-12))
  t2 = testCond(all(abs(psi) < 1e-12))
  t3 = testCond(abs(d_keff - keff_test) < 1e-12)

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

  keff_test = 2.42494653837

  phi_test = [3.9968764019364260, 5.3155446123379970, 0.2410710372839961, &
              0.0001149482887246657, 0.0000000000000000, &
              0.0000000000000000, 0.0000000000000000]
  psi_test = [1.9984382009682120, 2.6577723061689960, 0.1205355186419960, &
              5.747414436169997e-05, 0.0000000000000000, &
              0.0000000000000000, 0.0000000000000000]

  phi = phi / phi(0,1,1) * phi_test(1)
  psi = psi / psi(1,1,1) * psi_test(1)

  do g = 1, 7
    phi(0,g,:) = phi(0,g,:) - phi_test(g)
    psi(g,:,:) = psi(g,:,:) - psi_test(g)
  end do

  t1 = testCond(all(abs(phi(0,:,:)) < 1e-12))
  t2 = testCond(all(abs(psi) < 1e-12))
  t3 = testCond(abs(d_keff - keff_test) < 1e-12)

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
  legendre_order = 0

  call initialize_solver()

  keff_test = 0.682474203986

  phi_test = reshape([&
                  0.1334183765212053, 0.1335607508024285, 0.1338459024547992, &
                  0.1347097532018252, 0.1359152953908848, &
                  0.1367999722097782, 0.1373807310770533, 0.1376684521063088, &
                  0.1376684521063086, 0.1373807310770527, &
                  0.1367999722097771, 0.1359152953908832, 0.1347097532018231, &
                  0.1338459024547968, 0.1335607508024260, &
                  0.1334183765212027],shape(phi_test))

  call solve()

  phi_test = phi_test / phi_test(1,1) * phi(0,1,1)

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                          * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 16) < 1e-12))

  t3 = testCond(abs(d_keff - keff_test) < 1e-12)

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

  keff_test = 0.841854685248

  phi_test = reshape([&
                0.1339318310846705, 0.04663240631432017, 0.1340755294135996, &
                0.04550808086281581, 0.1343633342862139, &
                0.04320684147414546, 0.1351651393398062, 0.03844347521197749, &
                0.1361573774219628, 0.03329929560434541, &
                0.1367428466088811, 0.03046450810335382, 0.1370697836329809, &
                0.02897019950620232, 0.1372163851563189, &
                0.02832567466265055, 0.1372163851563192, 0.02832567466265063, &
                0.1370697836329816, 0.02897019950620256, &
                0.1367428466088823, 0.03046450810335424, 0.1361573774219644, &
                0.03329929560434609, 0.1351651393398082, &
                0.03844347521197848, 0.1343633342862161, 0.04320684147414675, &
                0.1340755294136020, 0.04550808086281728, &
                0.1339318310846729, 0.0466324063143218],shape(phi_test))

  call solve()

  phi_test = phi_test / phi_test(1,1) * phi(0,1,1)
  psi_test = psi_test / psi_test(1,1,1) * psi(1,1,1)

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                          * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 32) < 1e-12))

  t3 = testCond(abs(d_keff - keff_test) < 1e-12)

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

  keff_test = 1.3515451561148291

  phi_test = reshape([&
        2.2650616717248330, 2.8369334989286640, 0.07877830602847499, &
        7.04175246066569e-05, 3.25125257276947e-11, &
        2.432715876538015e-13, 1.594688968500785e-13, 2.2650464863137900, &
        2.8373095442874820, 0.07884130784914634, &
        7.050248988884302e-05, 3.234115428851006e-11, 2.427827585455429e-13, &
        1.515289029391496e-13, 2.2650161004805230, &
        2.8380617352518300, 0.07896744062200396, 7.067268976824239e-05, &
        3.19976020486985e-11, 2.418069550260201e-13, &
        1.343204034225335e-13, 2.2648483072967060, 2.8404426435028810, &
        0.0792086051696267, 7.098912673711065e-05, &
        3.13570088400893e-11, 2.402931204474427e-13, 1.104247242592327e-13, &
        2.2645858148744680, 2.8438348741494050, &
        0.07949631398906469, 7.136059982287322e-05, 3.06061130826834e-11, &
        2.386597640917436e-13, 9.011183599519791e-14, &
        2.2643919965970370, 2.8463720474176470, 0.07971088504374188, &
        7.163648260124387e-05, 3.005154823252787e-11, &
        2.373973994069494e-13, 7.812161732463536e-14, 2.2642642335724730, &
        2.8480601720159530, 0.07985335910593912, &
        7.181911886806704e-05, 2.968589519237929e-11, 2.365386404004055e-13, &
        7.144075059858521e-14, 2.2642007860664400, &
        2.8489032388283640, 0.07992442477259108, 7.191005343950646e-05, &
        2.950427766691452e-11, 2.361042454097472e-13, &
        6.845224668383328e-14, 2.2642007860664400, 2.8489032388283650, &
        0.07992442477259115, 7.191005343947548e-05, &
        2.950427829165693e-11, 2.36104305719265e-13, 6.845228405420781e-14, &
        2.2642642335724730, 2.8480601720159530, &
        0.07985335910593934, 7.181911886797399e-05, 2.968589707048321e-11, &
        2.365388222674651e-13, 7.144086685934094e-14, &
        2.2643919965970370, 2.8463720474176490, 0.07971088504374228, &
        7.163648260108838e-05, 3.005155137567659e-11, &
        2.373977056716846e-13, 7.812182597719825e-14, 2.2645858148744680, &
        2.8438348741494060, 0.07949631398906526, &
        7.136059982265474e-05, 3.060611751052739e-11, 2.386601995704075e-13, &
        9.011216336044022e-14, 2.2648483072967060, &
        2.8404426435028820, 0.07920860516962742, 7.098912673682837e-05, &
        3.135701458051651e-11, 2.402936920587044e-13, &
        1.104252163837134e-13, 2.2650161004805240, 2.8380617352518310, &
        0.07896744062200481, 7.067268976791016e-05, &
        3.199760881957515e-11, 2.41807649791448e-13, 1.34321094239084e-13, &
        2.2650464863137900, 2.8373095442874820, &
        0.07884130784914729, 7.050248988847486e-05, 3.23411617946501e-11, &
        2.427835585191803e-13, 1.515298009041951e-13, &
        2.2650616717248340, 2.8369334989286660, 0.07877830602847602, &
        7.041752460625216e-05, 3.25125339754215e-11, &
        2.432724907945733e-13, 1.59470018579366e-13],shape(phi_test))

  call solve()

  phi_test = phi_test / phi_test(1,1) * phi(0,1,1)

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                          * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 112) < 1e-12))

  t3 = testCond(abs(d_keff - keff_test) < 1e-12)

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

subroutine anisotropic1g()
  use control
  use solver
  use angle, only : wt

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, a, c, an
  double precision :: phi_test(1,16), psi_test(1,4,16), keff_test

  ! Define problem parameters
  call initialize_control('test/pin_test_options', .true.)
  legendre_order = 1

  call initialize_solver()

  keff_test = 0.6824742

  phi_test = reshape([&
              0.13341837652200000, 0.13356075080322422, 0.13384590245559674, &
              0.13470975320262829, 0.13591529539169578, &
              0.13679997221059509, 0.13738073107787424, 0.13766845210713188, &
              0.13766845210713188, 0.13738073107787430, &
              0.13679997221059520, 0.13591529539169586, 0.13470975320262843, &
              0.13384590245559685, 0.13356075080322433, &
              0.13341837652200009],shape(phi_test))

  call solve()

  phi_test = phi_test / phi_test(1,1) * phi(0,1,1)

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                          * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 16) < 1e-12))

  t3 = testCond(abs(d_keff - keff_test) < 1e-12)

  if (t1 == 0) then
    print *, 'solver: anisotropic 1g phi failed'
  else if (t2 == 0) then
    print *, 'solver: anisotropic 1g psi failed'
  else if (t3 == 0) then
    print *, 'solver: anisotropic 1g keff failed'
  else
    print *, 'all tests passed for anisotropic 1g solver'
  end if

  call finalize_solver()
  call finalize_control()

end subroutine anisotropic1g

subroutine anisotropic2g()
  use control
  use solver
  use angle, only : wt

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, a, c, an
  double precision :: phi_test(2,16), psi_test(2,4,16), keff_test

  ! Define problem parameters
  call initialize_control('test/pin_test_options', .true.)
  material_map = [5, 1, 5]
  angle_order = 2
  xs_name = 'test/2gXSaniso.anlxs'
  legendre_order = 7

  call initialize_solver()

  keff_test = 0.43942560
  phi_test = 0.0
  phi_test = reshape([&
                  30.873137430393538, 0.5498013920878216, 30.872051438564693, &
                  0.5499423487323769,&
                  30.869954350895192, 0.5502186237557053, 30.858188929487795, &
                  0.552892568086489,&
                  30.83962534645895, 0.55721332781946, 30.826024305427918, &
                  0.5602735275004431,&
                  30.817018210691153, 0.5622319371418444, 30.812423264396884, &
                  0.5631875110769776,&
                  30.812423264396884, 0.5631875110769776, 30.817018210691153, &
                  0.5622319371418444,&
                  30.826024305427918, 0.5602735275004431, 30.83962534645895, &
                  0.55721332781946,&
                  30.858188929487795, 0.552892568086489, 30.869954350895192, &
                  0.5502186237557053,&
                  30.872051438564693, 0.5499423487323769, 30.873137430393538, &
                  0.5498013920878216],shape(phi_test))

  call solve()

  phi_test = phi_test / phi_test(1,1) * phi(0,1,1)

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
      phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                          * psi(:, 2 * number_angles - a + 1,c)
    end do
  end do
  phi_test(:,:) = phi_test(:,:) / phi(0,:,:)

  t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 32) < 1e-12))

  t3 = testCond(abs(d_keff - keff_test) < 1e-12)

  if (t1 == 0) then
    print *, 'solver: anisotropic 2g phi failed'
  else if (t2 == 0) then
    print *, 'solver: anisotropic 2g psi failed'
  else if (t3 == 0) then
    print *, 'solver: anisotropic 2g keff failed'
  else
    print *, 'all tests passed for anisotropic 2g solver'
  end if

  call finalize_solver()
  call finalize_control()

end subroutine anisotropic2g

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
