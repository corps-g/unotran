program test_dgmsolver
  call test2g()
  call vacuum1()
  call reflect1()
  call vacuum2()
  call reflect2()
!  call eigenV2g()
!  call eigenV7g()
!  call eigenR2g()
!  call eigenR7g()
end program test_dgmsolver

! Test the 2 group dgm solution
subroutine test2g()
  use control
  use dgmsolver

  implicit none

  ! initialize types
  integer :: testCond, t1=1, t2=1
  double precision :: phi_test(0:0,2,1), psi_test(2,4,1)
  double precision :: phi_new(0:0,2,1), psi_new(2,4,1)

  call initialize_control('test/dgm_test_options1', .true.)

  phi_test = reshape([1.1420149990909008,0.37464706668551212], shape(phi_test))

  psi_test = reshape([0.81304488744042813,0.29884810509581583,&
                      1.31748796916478740,0.41507830480599001,&
                      1.31748796916478740,0.41507830480599001,&
                      0.81304488744042813,0.29884810509581583&
                      ], shape(psi_test))

  ! initialize the variables necessary to solve the problem
  call initialize_dgmsolver()

  ! set source
  source(:,:,:) = 1.0

  ! set phi and psi to the converged solution
  phi = phi_test
  psi = psi_test

  call dgmsweep(phi_new, psi_new, incoming)

  t1 = testCond(norm2(phi - phi_test) < 1e-12)
  t2 = testCond(norm2(psi - psi_test) < 1e-12)

  if (t1 == 0) then
    print *, 'DGM solver 2g: phi failed'
  else if (t2 == 0) then
    print *, 'DGM solver 2g: psi failed'
  else
    print *, 'all tests passed for DGM solver 2g'
  end if

  call finalize_dgmsolver()
  call finalize_control()

end subroutine test2g

! Test against detran with vacuum conditions
subroutine vacuum1()
  use control
  use dgmsolver

  implicit none

  ! initialize types
  integer :: l, c, a, g, counter, testCond, t1, t2
  double precision :: norm, error, phi_test(7,28)

  ! Define problem parameters
  call initialize_control('test/dgm_test_options2', .true.)
  call initialize_dgmsolver()

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

  phi(0,:,:) = phi_test

  call dgmsolve()

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-12)

  if (t1 == 0) then
    print *, 'dgmsolver: vacuum test 1 failed'
  else
    print *, 'all tests passed for dgmsolver vacuum 1'
  end if

  call finalize_dgmsolver()
  call finalize_control()

end subroutine vacuum1

! test against detran with reflective conditions
subroutine reflect1()
  use control
  use dgmsolver

  implicit none

  ! initialize types
  integer :: l, c, a, g, counter, testCond, t1, t2
  double precision :: norm, error, phi_test(7,28)

  ! Define problem parameters
  call initialize_control('test/dgm_test_options2', .true.)
  material_map = [1, 1, 1]
  boundary_type = [1.0, 1.0]
  allow_fission = .false.
  outer_tolerance = 1e-10
  lambda = 0.3
  max_inner_iters = 10
  max_outer_iters = 5000
  ignore_warnings = .true.


  call initialize_dgmsolver()


  source = 1.0

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

  phi(0,:,:) = phi_test

  call dgmsolve()

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))

  if (t1 == 0) then
    print *, 'dgmsolver: reflection test 1 failed'
  else
    print *, 'all tests passed for dgmsolver reflect 1'
  end if

  call finalize_dgmsolver()
  call finalize_control()

end subroutine reflect1

! Test against detran with vacuum conditions and 1 spatial cell
subroutine vacuum2()
  use control
  use dgmsolver

  implicit none

  ! initialize types
  integer :: fineMesh(1), materialMap(1), l, c, a, g, counter, testCond, t1, t2
  double precision :: coarseMesh(2), norm, error, phi_test(7,1), &
                      boundary(2), psi_test(7,4,1)

  call initialize_control('test/dgm_test_options3', .true.)
  call initialize_dgmsolver()

  phi_test = reshape([1.1076512516190389, 1.1095892550819531,&
                      1.0914913168898499, 1.0358809957845283,&
                      0.93405352272848619, 0.79552760081182894,&
                      0.48995843862242699],shape(phi_test))

  psi_test = reshape([ 0.62989551954274092,0.65696484059125337,&
                       0.68041606804080934,0.66450867626366705,&
                       0.60263096140806338,0.53438683380855967,&
                       0.38300537939872042,1.3624866129734592,&
                       1.3510195477616225,1.3107592451448140,&
                       1.2339713438183100,1.1108346317861364,&
                       0.93482033401344933,0.54700730201708170,&
                       1.3624866129734592,1.3510195477616225,&
                       1.3107592451448140,1.2339713438183100,&
                       1.1108346317861364,0.93482033401344933,&
                       0.54700730201708170,0.62989551954274092,&
                       0.65696484059125337,0.68041606804080934,&
                       0.66450867626366705,0.60263096140806338,&
                       0.53438683380855967,0.38300537939872042], &
                       shape(psi_test))

  phi(0,:,:) = phi_test
  psi(:,:,:) = psi_test
  source(:,:,:) = 1.0

  call dgmsolve()

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-12)

  if (t1 == 0) then
    print *, 'dgmsolver: vacuum test 2 failed'
  else
    print *, 'all tests passed for dgmsolver vacuum 2'
  end if

  call finalize_dgmsolver()
  call finalize_control()

end subroutine vacuum2

! test against detran with reflective conditions
subroutine reflect2()
  use control
  use dgmsolver

  implicit none

  ! initialize types
  integer :: fineMesh(1), materialMap(1), l, c, a, g, counter, testCond, t1, t2
  double precision :: coarseMesh(2), norm, error, phi_test(7,1), &
                      psi_test(7,4,1), boundary(2)

  call initialize_control('test/dgm_test_options3', .true.)
  boundary_type = [1.0, 1.0]
  call initialize_dgmsolver()

  source = 1.0

  phi_test = reshape([94.512658438949273,106.66371642824166,&
                      75.397102078564259,17.953651480951333,&
                      6.2855008963667123,3.0158479735464110,&
                      1.2132770548341645],shape(phi_test))

  psi_test = reshape([94.512658438949273,106.66371642824166,&
                      75.397102078564259,17.953651480951333,&
                      6.2855008963667123,3.0158479735464110,&
                      1.2132770548341645,94.512658454844384,&
                      106.66371644092482,75.397102082192546,&
                      17.953651480951343,6.2855008963667141,&
                      3.0158479735464105,1.2132770548341649,&
                      94.512658454844384,106.66371644092482,&
                      75.397102082192546,17.953651480951343,&
                      6.2855008963667141,3.0158479735464105,&
                      1.2132770548341649,94.512658438949273,&
                      106.66371642824166,75.397102078564259,&
                      17.953651480951333,6.2855008963667123,&
                      3.0158479735464110,1.2132770548341645], shape(psi_test))

  phi(0,:,:) = phi_test
  psi(:,:,:) = psi_test
  source(:,:,:) = 1.0

  call dgmsolve()

  t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))

  if (t1 == 0) then
    print *, 'dgmsolver: reflection test 2 failed'
  else
    print *, 'all tests passed for dgmsolver reflect 2'
  end if

  call finalize_dgmsolver()
  call finalize_control()

end subroutine reflect2

! Test the 2g eigenvalue problem
subroutine eigenV2g()

  use control
  use dgmsolver
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
  legendre_order = 0
  dgm_basis_name = '2gbasis'
  use_DGM = .true.
  use_recondensation = .false.
  outer_print = .false.
  inner_print = .false.
  lambda = 0.25
  max_inner_iters = 50
  max_outer_iters = 5000
  ignore_warnings = .true.

!  call output_control()

  call initialize_dgmsolver()

  keff_test = 0.8099523232983425

  phi_test = reshape([&
              0.05882749189352335, 0.009858087422743867, 0.1099501419733834, &
              0.01934781232927005, 0.1497992040255296, &
              0.02617665800594787, 0.1781310366743199, 0.03122421632573503, &
              0.1928663608358422, 0.03377131381204261, &
              0.1928663608358422, 0.03377131381204262, 0.1781310366743200, &
              0.03122421632573504, 0.1497992040255296, &
              0.02617665800594788, 0.1099501419733835, 0.01934781232927007, &
              0.05882749189352339, 0.009858087422743878],shape(phi_test))

  call dgmsolve()

  phi = phi / phi(0,1,1) * phi_test(1,1)

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-12)

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
    print *, 'dgmsolver: eigen 2g vacuum solver phi failed'
  else if (t2 == 0) then
    print *, 'dgmsolver: eigen 2g vacuum solver psi failed'
  else if (t3 == 0) then
    print *, 'dgmsolver: eigen 2g vacuum solver keff failed'
  else
    print *, 'all tests passed for eigen 2g vacuum dgmsolver'
  end if

  call finalize_dgmsolver()
  call finalize_control()

end subroutine eigenV2g

! Eigenvalue test with vacuum conditions
subroutine eigenV7g()
  use control
  use dgmsolver
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
  legendre_order = 0
  dgm_basis_name = 'basis'
  use_DGM = .true.
  use_recondensation = .false.
  energy_group_map = [4]
  outer_print = .true.
  inner_print = .false.
  lambda = 0.4
  max_inner_iters = 10
  max_outer_iters = 5000
  ignore_warnings = .true.

  call initialize_dgmsolver()

  keff_test = 0.4347999090254699

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

  call dgmsolve()

  print *, phi(0,:,:)

  phi_test = phi_test / phi_test(1, 1) * phi(0, 1, 1)

  print *
  print *, phi_test

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-12)

  phi_test = 0
  do c = 1, number_cells
    do a = 1, number_angles
      phi_test(:, c) = phi_test(:, c) + 0.5 * wt(a) * psi(:, a, c)
      phi_test(:, c) = phi_test(:, c) + 0.5 * wt(number_angles - a + 1) &
                                          * psi(:, 2 * number_angles - a + 1, c)
    end do
  end do
  t2 = testCond(all(abs(phi_test(:,:) - phi(0,:,:)) < 1e-12))

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

  call finalize_dgmsolver()
  call finalize_control()

end subroutine eigenV7g

! Eigenvalue test with reflective conditions for 2 groups
subroutine eigenR2g()
  use control
  use dgmsolver

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, g
  double precision :: phi_test(2), keff_test, psi_test(2)

  ! Define problem parameters
  call initialize_control('test/eigen_test_options', .true.)
  xs_name = 'test/2gXS.anlxs'
  legendre_order = 0
  dgm_basis_name = '2gbasis'
  use_DGM = .true.
  use_recondensation = .false.
  outer_print = .false.
  inner_print = .false.
  lambda = 0.1
  max_inner_iters = 10
  max_outer_iters = 5000
  ignore_warnings = .true.

  call initialize_dgmsolver()

  call dgmsolve()

  keff_test = 0.8403361344537817

  phi_test = [0.1428571428571399, 0.0252100840336125]
  psi_test = [0.07142857142856951, 0.0126050420168061]

  phi = phi / phi(0,1,1) * phi_test(1)
  psi = psi / psi(1,1,1) * psi_test(1)

  do g = 1, 2
    phi(0,g,:) = phi(0,g,:) - phi_test(g)
    psi(g,:,:) = psi(g,:,:) - psi_test(g)
  end do

  t1 = testCond(all(abs(phi) < 1e-6))
  t2 = testCond(all(abs(psi) < 1e-6))
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

  call finalize_dgmsolver()
  call finalize_control

end subroutine eigenR2g

! Eigenvalue test with reflective conditions
subroutine eigenR7g()
  use control
  use dgmsolver
  use dgm, only : energyMesh, basis

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, g, c, a, cg, i
  double precision :: phi_test(7), keff_test, psi_test(7)

  ! Define problem parameters
  call initialize_control('test/eigen_test_options', .true.)
  xs_name = 'test/testXS.anlxs'
  legendre_order = 0
  dgm_basis_name = 'basis'
  use_DGM = .true.
  use_recondensation = .false.
  energy_group_map = [4]
  outer_print = .false.
  inner_print = .false.
  lambda = 0.1
  max_inner_iters = 1000
  max_outer_iters = 5000
  ignore_warnings = .true.

  !call output_control()

  keff_test = 2.4249465383700772

  phi_test = [3.9968764019364260, 5.3155446123379970, 0.2410710372839961, &
              0.0001149482887246657, 0.0000000000000000, &
              0.0000000000000000, 0.0000000000000000]
  psi_test = [1.9984382009682120, 2.6577723061689960, 0.1205355186419960, &
              5.747414436169997e-05, 0.0000000000000000, &
              0.0000000000000000, 0.0000000000000000]

  call initialize_dgmsolver()

  do c=1, number_cells
    do a=1, number_angles * 2
      if (a == 1) phi(0,:,c) = phi_test
      psi(:,a,c) = phi_test
    end do
  end do

  incoming = 0
  do i = 0, expansion_order
    do a = 1, number_angles
      do g = 1, number_groups
        cg = energyMesh(g)
        ! Angular flux
        incoming(cg, a, i) = incoming(cg, a, i) +  basis(g, i) * psi(g, a, 1)
      end do
    end do
  end do

  d_keff = keff_test

  call dgmsolve()

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

  call finalize_dgmsolver()
  call finalize_control

end subroutine eigenR7g

! Eigenvalue test with reflect conditions for a pin cell
subroutine eigenR2gPin()
  use control
  use dgmsolver
  use angle, only : wt

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, a, c
  double precision :: phi_test(2,16), psi_test(2,4,16), keff_test

  ! Define problem parameters
  call initialize_control('test/pin_test_options', .true.)
  xs_name = 'test/2gXS.anlxs'

  call initialize_dgmsolver()

  keff_test = 0.8418546852484950

  phi_test = reshape([0.133931831084, 0.0466324063141, 0.134075529413, &
                      0.0455080808626, 0.1343633343, &
                      0.0432068414739, 0.13516513934, 0.0384434752118, &
                      0.136157377422, 0.0332992956, &
                      0.136742846609, 0.0304645081033, 0.137069783633, &
                      0.0289701995061, 0.1372163852, &
                      0.0283256746626, 0.137216385156, 0.0283256746626, &
                      0.137069783633, 0.02897019951, &
                      0.136742846609, 0.0304645081033, 0.136157377422, &
                      0.0332992956043, 0.1351651393, &
                      0.0384434752119, 0.134363334286, 0.0432068414741, &
                      0.134075529413, 0.04550808086, &
                      0.133931831085, 0.0466324063142],shape(phi_test))

  call dgmsolve()

  phi_test = phi_test / phi_test(1,1) * phi(0,1,1)
  psi_test = psi_test / psi_test(1,1,1) * psi(1,1,1)

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-12)

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

  call finalize_dgmsolver()
  call finalize_control()

end subroutine eigenR2gPin

! Eigenvalue test with reflect conditions for a pin cell
subroutine eigenR7gPin()
  use control
  use dgmsolver
  use angle, only : wt

  implicit none

  ! initialize types
  integer :: testCond, t1, t2, t3, c, a
  double precision :: phi_test(7,16), psi_test(7,4,16), keff_test

  ! Define problem parameters
  call initialize_control('test/pin_test_options', .true.)
  xs_name = 'test/testXS.anlxs'
  material_map = [6, 1, 6]

  call initialize_dgmsolver()

  keff_test = 1.3515451561148644

  phi_test = reshape([2.2650616717, 2.8369334989, 0.0787783060158, &
                      7.04175236386e-05, 3.247808073e-11, &
                      2.42090423431e-13, 1.5855325733e-13, 2.26504648629, &
                      2.83730954426, 0.07884130784, &
                      7.05024889238e-05, 3.23070059067e-11, 2.416262144e-13, &
                      1.50689303232e-13, 2.2650161, &
                      2.83806173523, 0.0789674406093, 7.06726888056e-05, &
                      3.19637591127e-11, 2.406809989e-13, &
                      1.33594887863e-13, 2.26484830727, 2.84044264348, &
                      0.0792086051569, 7.098912578e-05, &
                      3.13235926512e-11, 2.39206404496e-13, 1.09840721921e-13, &
                      2.26458581485, 2.843834874, &
                      0.0794963139764, 7.13605988661e-05, 3.05732041796e-11, &
                      2.37613706246e-13, 8.964680919e-14, &
                      2.26439199657, 2.84637204739, 0.0797108850311, &
                      7.16364816476e-05, 3.001907576e-11, &
                      2.36383833154e-13, 7.77296907704e-14, 2.26426423355, &
                      2.84806017199, 0.07985335909, &
                      7.18191179175e-05, 2.96537913389e-11, 2.35550014439e-13, &
                      7.10923697151e-14, 2.264200786, &
                      2.8489032388, 0.0799244247599, 7.1910052492e-05, &
                      2.94724771287e-11, 2.351334627e-13, &
                      6.81268597471e-14, 2.26420078604, 2.8489032388, &
                      0.0799244247599, 7.191005249e-05, &
                      2.9472717738e-11, 2.3514457477e-13, 6.81336094823e-14, &
                      2.26426423355, 2.848060172, &
                      0.0798533590933, 7.18191179264e-05, 2.96545146531e-11, &
                      2.35583516359e-13, 7.111336685e-14, &
                      2.26439199657, 2.84637204739, 0.0797108850311, &
                      7.16364816625e-05, 3.002028626e-11, &
                      2.36440226704e-13, 7.77673697266e-14, 2.26458581485, &
                      2.84383487412, 0.07949631398, &
                      7.1360598887e-05, 3.05749093972e-11, 2.37693842494e-13, &
                      8.97059192338e-14, 2.264848307, &
                      2.84044264348, 0.0792086051571, 7.09891258041e-05, &
                      3.13258032808e-11, 2.393115064e-13, &
                      1.09929579303e-13, 2.26501610046, 2.83806173523, &
                      0.0789674406095, 7.067268884e-05, &
                      3.19663665349e-11, 2.40808670657e-13, 1.33719616187e-13, &
                      2.26504648629, 2.837309544, &
                      0.0788413078367, 7.05024889589e-05, 3.23098965985e-11, &
                      2.41773215344e-13, 1.508514105e-13, &
                      2.2650616717, 2.8369334989, 0.078778306016, &
                      7.04175236773e-05, 3.248125734e-11, &
                      2.42256465053e-13, 1.58755730155e-13],shape(phi_test))

  call dgmsolve()

  phi_test = phi_test / phi_test(1,1) * phi(0,1,1)

  t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-12)

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

  call finalize_dgmsolver()
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
