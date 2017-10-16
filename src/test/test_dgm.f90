program test_dgm
  call test1()
  call test2()
  call test3()
end program test_dgm

! Test that the DGM moments are correctly computed for a flux of one
subroutine test1()
  use control
  use material, only : create_material, number_legendre
  use angle, only : initialize_angle, initialize_polynomials
  use mesh, only : create_mesh
  use state, only : initialize_state, source, d_source, d_nu_sig_f, d_delta, d_phi, &
                    d_chi, d_sig_s, d_sig_t, d_psi
  use dgmsolver, only : finalize_dgmsolver
  use dgm, only : number_coarse_groups, basis, energymesh, expansion_order, order, &
                  initialize_moments, initialize_basis, compute_flux_moments, compute_xs_moments, &
                  compute_source_moments

  implicit none

  ! initialize types
  integer :: testCond, l, i, c, cg, cgp,g,gp
  integer :: t1=1, t2=1, t3=1, t4=1, t5=1, t6=1, t7=1, t8=1, t9=1, t10=1, t11=1, t12=1, t13=1
  double precision :: norm, error, basis_test(7,4)
  double precision :: phi_m_test(0:7,2,1), psi_m_test(2,4,1),source_m_test(0:3,2,4,1)
  double precision :: sig_t_m_test(2,1), delta_m_test(0:3,2,4,1), sig_s_m_test(8,0:3,2,2,1)
  double precision :: nu_sig_f_m_test(2,1), chi_m_test(0:3,2,1)
  integer :: order_test(2), energyMesh_test(7), eo_test, ncg_test

  ! Define problem parameters
  call initialize_control('test/dgm_test_options3', .true.)
  xs_name = 'test.anlxs'
  allow_fission = .true.

  phi_m_test = 0
  psi_m_test = 0
  source_m_test = 0

  order_test = [3, 2]
  energyMesh_test = [1,1,1,1,2,2,2]
  eo_test = 3
  ncg_test = 2

  ! setup problem
  call create_mesh()
  call create_material()
  call initialize_angle()
  call initialize_polynomials(number_legendre)
  call initialize_state()
  source = 1.0

  ! test the moment construction
  call initialize_moments()

  t1 = testCond(all(order == order_test))
  t2 = testCond(all(energyMesh == energyMesh_test))
  t3 = testCond(expansion_order - eo_test == 0)
  t4 = testCond(number_coarse_groups-ncg_test == 0)

  ! Test reading the basis set from the file
  call initialize_basis()

  basis_test = reshape([ 0.5       ,  0.5       ,  0.5       ,  0.5       , 0.5773502691896258421,  0.5773502691896258421,  0.5773502691896258421,&
                        0.6708203932499369193, 0.2236067977499789639,  -0.2236067977499789639 ,  -0.6708203932499369193,  0.7071067811865474617,  0.        , -0.7071067811865474617,&
                         0.5       , -0.5       , -0.5       ,  0.5       ,  0.4082482904638630727, -0.8164965809277261455,  0.4082482904638630727,&
                        0.2236067977499789361,  -0.6708203932499369193, 0.6708203932499369193,  -0.2236067977499789361 ,  0.        ,  0.        ,  0.        ], &
                         shape(basis_test))

  t5 = testCond(all(abs(basis - basis_test) < 1e-12))

  call compute_source_moments()

  ! Test basis computation
  call compute_flux_moments()

  ! test scalar flux moments
  phi_m_test = reshape([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, &
                         1.7320508075688776, 1.7320508075688776, 1.7320508075688776, 1.7320508075688776, &
                         1.7320508075688776, 1.7320508075688776, 1.7320508075688776, 1.7320508075688776], &
                        shape(phi_m_test))

  t6 = testCond(all(abs(d_phi - phi_m_test) < 1e-12))

  ! test angular flux moments
  psi_m_test = reshape([2.0, 1.7320508075688776, &
                         2.0, 1.7320508075688776, &
                         2.0, 1.7320508075688776, &
                         2.0, 1.7320508075688776], &
                        shape(psi_m_test))
  
  t7 = testCond(all(abs(d_psi - psi_m_test) < 1e-12))

  ! test source moments
  source_m_test = reshape([2.0, 0.0, 0.0, 0.0, 1.7320508075688776, 0.0, 0.0, 0.0, &
                           2.0, 0.0, 0.0, 0.0, 1.7320508075688776, 0.0, 0.0, 0.0, &
                           2.0, 0.0, 0.0, 0.0, 1.7320508075688776, 0.0, 0.0, 0.0, &
                           2.0, 0.0, 0.0, 0.0, 1.7320508075688776, 0.0, 0.0, 0.0], &
                        shape(source_m_test))

  ! test total cross section moments
  sig_t_m_test = reshape([0.3760865, 1.0070863333333333], shape(sig_t_m_test))

  ! test angular cross section moments (delta)
  delta_m_test = reshape([ 0.00000000e+00,   -1.2284241926631045e-01,   1.8900000000000167e-04,  1.0668056713853770e-02,&
                           -3.2049378106392724e-17,  -4.8916473462696236e-01,   2.0934839066069319e-01,   0.00000000e+00,&
                           0.00000000e+00,   -1.2284241926631045e-01,   1.8900000000000167e-04,  1.0668056713853770e-02,&
                           -3.2049378106392724e-17,  -4.8916473462696236e-01,   2.0934839066069319e-01,   0.00000000e+00,&
                           0.00000000e+00,   -1.2284241926631045e-01,   1.8900000000000167e-04,  1.0668056713853770e-02,&
                           -3.2049378106392724e-17,  -4.8916473462696236e-01,   2.0934839066069319e-01,   0.00000000e+00,&
                           0.00000000e+00,   -1.2284241926631045e-01,   1.8900000000000167e-04,  1.0668056713853770e-02,&
                           -3.2049378106392724e-17,  -4.8916473462696236e-01,   2.0934839066069319e-01,   0.00000000e+00&
                           ], shape(delta_m_test))

  ! test scattering cross section moments
  sig_s_m_test = reshape([0.35342781806, 0.04743636186125, 0.0289331339485425, 0.02013445119055, &
                          0.01450924725765, 0.0093014956238, 0.005620442104, 0.0030043367622, &
                          -0.1087492753068697, 0.032934103959394, 0.0289449758029991, 0.0224441910315558, &
                          0.0167896813253091, 0.0112993763443412, 0.0069750093143139, 0.0039106927748581, &
                          -0.01426407321, 0.00452756650875, 0.0076277391555075, 0.00999316946045, &
                          0.00843282744865, 0.0066059506217, 0.0043978711625, 0.0026803333838,&
                          0.0140474443693441, -0.0065186049755285, -0.0052873421588672, -0.0000093035117265, &
                          0.0010863232979017, 0.0017467451521609, 0.0014185324207123, 0.0010077531638459, &
                          0.0015106138853239, -0.0004682252948101, -0.0000210589954063, -0.000000649273679, &
                          0.0000009154696808, 0.0000030289238497, -0.0000021934200679, 0.0000012245339402, &
                          -0.0020267012012036, 0.0006281901527882, 0.0000282536071598, 0.0000008710920493, &
                          -0.0000012282314626, -0.000004063727776, 0.0000029427818251, -0.0000016428846786, &
                          0.0015106138853239, -0.0004682252948101, -0.0000210589954063, -0.000000649273679, &
                          0.0000009154696808, 0.0000030289238497, -0.0000021934200679, 0.0000012245339402, &
                          -0.0006755670670679, 0.0002093967175961, 0.0000094178690533, 0.0000002903640164, &
                          -0.0000004094104875, -0.0000013545759253, 0.000000980927275, -0.0000005476282262, &
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
                          0.4236901533333333, 0.0053083109, 0.0016464802333333, 0.0012283232933333, &
                          0.0007583007633333, -0.0006374102666667, -0.000077602452, 0.0004478038633333, &
                          -0.0379884015739014, 0.0038739129355235, -0.0015822948479042, -0.0014793513753941, &
                          -0.0010802063604453, 0.0007826396112285, 0.0001042419486558, -0.0005035107714306, &
                          0.0326467618199471, -0.0018823973060567, 0.0009853706472698, 0.0007925234371622, &
                          0.0005038871725552, -0.0003595438342519, -0.0000560788795342, 0.0002638059860685, &
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape(sig_s_m_test))

  ! test fission cross section moments
  nu_sig_f_m_test = reshape([0.039245915, 1.504587272273979], shape(nu_sig_f_m_test))

  ! test chi spectrum moments
  chi_m_test = reshape([0.50000031545, 0.2595979010884317, -0.3848771845500001, -0.5216663031659152,&
                         0.0, 0.0, 0.0, 0.0], shape(chi_m_test))

  do i = 0, expansion_order
    call compute_xs_moments(i)
    source_m_test(i,:,:,:) = source_m_test(i,:,:,:) - d_source
    delta_m_test(i,:,:,:) = delta_m_test(i,:,:,:) - d_delta
    sig_s_m_test(:,i,:,:,:) = sig_s_m_test(:,i,:,:,:) - d_sig_s
    chi_m_test(i,:,:) = chi_m_test(i,:,:) - d_chi
  end do

  t8 = testCond(all(abs(source_m_test) < 1e-12))
  t9 = testCond(all(abs(d_sig_t - sig_t_m_test) < 1e-12))
  t10 = testCond(all(abs(delta_m_test) < 1e-12))
  t11 = testCond(all(abs(sig_s_m_test) < 1e-12))
  t12 = testCond(all(abs(d_nu_sig_f - nu_sig_f_m_test) < 1e-12))
  t13 = testCond(all(abs(chi_m_test) < 1e-12))


  if (t1 == 0) then
    print *, 'DGM1: order failed'
  else if (t2 == 0) then
    print *, 'DGM1: energy mesh failed'
  else if (t3 == 0) then
    print *, 'DGM1: expansion order failed'
  else if (t4 == 0) then
    print *, 'DGM1: number coarse groups failed'
  else if (t5 == 0) then
    print *, 'DGM1: basis failed'
  else if (t6 == 0) then
    print *, 'DGM1: phi moment failed'
  else if (t7 == 0) then
    print *, 'DGM1: psi moment failed'
  else if (t8 == 0) then
    print *, 'DGM1: source moment failed'
  else if (t9 == 0) then
    print *, 'DGM1: sig_t moment failed'
  else if (t10 == 0) then
    print *, 'DGM1: delta moment failed'
  else if (t11 == 0) then
    print *, 'DGM1: sig_s moment failed'
  else if (t12 == 0) then
    print *, 'DGM1: nu_sig_f moment failed'
  else if (t13 == 0) then
    print *, 'DGM1: chi moment failed'
  else
    print *, 'all tests passed for DGM1'
  end if

  call finalize_dgmsolver()
  call finalize_control()
end subroutine test1

! Test that the DO method is matched if using the delta basis and 1 fine per coarse energy group
subroutine test2()
  use control
  use material
  use angle, only : initialize_angle, initialize_polynomials
  use mesh, only : create_mesh
  use state, only : initialize_state, source, d_source, d_nu_sig_f, d_delta, d_phi, &
                    d_chi, d_sig_s, phi, psi, d_sig_t, d_psi
  use dgmsolver, only : finalize_dgmsolver
  use dgm, only : number_coarse_groups, basis, energymesh, expansion_order, order, &
                  initialize_moments, initialize_basis, compute_flux_moments, compute_xs_moments, &
                  compute_source_moments

  implicit none

  ! initialize types
  integer :: testCond, l, i, c, cg, cgp,g,gp
  integer :: t1=1, t2=1, t3=1, t4=1, t5=1, t6=1, t7=1, t8=1, t9=1, t10=1, t11=1, t12=1, t13=1
  double precision :: norm, error, basis_test(7,1)
  double precision :: phi_m_test(0:7,7,1), psi_m_test(7,1,1),source_m_test(7,4,1)
  double precision :: phi_test(0:7,7,1), psi_test(7,4,1)
  double precision :: sig_t_m_test(2,1), delta_m_test(7,4,1), sig_s_m_test(8,4,2,2,1)
  double precision :: nu_sig_f_m_test(7,1), chi_m_test(1,7,1)
  integer :: order_test(7), energyMesh_test(7), eo_test, ncg_test

  ! Define problem parameters
  call initialize_control('test/dgm_test_options4', .true.)

  phi_m_test = 0
  psi_m_test = 0
  source_m_test = 0

  order_test = [0,0,0,0,0,0,0]
  energyMesh_test = [1,2,3,4,5,6,7]
  eo_test = 0
  ncg_test = 7

  ! setup problem
  call create_mesh()
  call create_material()
  call initialize_angle()
  call initialize_polynomials(number_legendre)
  call initialize_state()

  phi_m_test = reshape([ 1.05033726e+00,  -1.38777878e-17,  -1.57267763e-01,   1.00000000e-08,&
                         1.43689742e-17,   1.00000000e-08,   9.88540222e-02,   6.93889390e-18,&
                         6.32597614e-02,  -8.67361738e-19,  -8.69286487e-03,   1.00000000e-08,&
                         6.81181057e-19,   2.16840434e-19,   5.46408649e-03,   1.00000000e-08,&
                         5.78458741e-03,   1.00000000e-08,  -7.15385990e-04,   1.00000000e-08,&
                         4.34869208e-20,   5.42101086e-20,   4.49671194e-04,  -5.42101086e-20,&
                         3.20772663e-05,  -4.23516474e-22,  -3.67904369e-06,  -4.23516474e-22,&
                         1.73044862e-22,   1.05879118e-22,   2.31254175e-06,   1.00000000e-08,&
                         1.55685757e-07,  -3.30872245e-24,  -1.78930684e-08,   8.27180613e-25,&
                         8.48610071e-25,   8.27180613e-25,   1.12470715e-08,   8.27180613e-25,&
                         7.86105345e-10,   1.00000000e-08,  -8.28643069e-11,   6.46234854e-27,&
                         2.51523310e-27,   3.23117427e-27,   5.20861358e-11,   6.46234854e-27,&
                         3.40490800e-13,  -6.31088724e-30,  -2.37767941e-14,  -1.57772181e-30,&
                        -1.77547007e-30,   1.57772181e-30,   1.49454134e-14,   3.15544362e-30], shape(phi_m_test))

  psi_m_test = reshape([0.66243843,  1.46237369,  0.73124892,  0.66469264,&
                       0.60404633,  0.59856738,  0.31047097,  1.52226789,&
                       3.07942675,  1.4095816 ,  1.23433893,  1.11360585,&
                       1.04656294,  0.44397067,  1.52226789,  3.07942675,&
                       1.4095816 ,  1.23433893,  1.11360585,  1.04656294,&
                       0.44397067,  0.66243843,  1.46237369,  0.73124892,&
                       0.66469264,  0.60404633,  0.59856738,  0.31047097], shape(psi_m_test))

  ! test the moment construction
  call initialize_moments()

  t1 = testCond(all(order == order_test))
  t2 = testCond(all(energyMesh == energyMesh_test))
  t3 = testCond(expansion_order-eo_test == 0)
  t4 = testCond(number_coarse_groups-ncg_test == 0)

  ! set phi and psi
  phi = phi_m_test(:,:,:)
  psi(:,1,:) = psi_m_test(:,1,:)
  psi(:,2,:) = psi_m_test(:,1,:)
  psi(:,3,:) = psi_m_test(:,1,:)
  psi(:,4,:) = psi_m_test(:,1,:)

  ! Test reading the basis set from the file
  call initialize_basis()
  call compute_source_moments()

  basis_test = reshape([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &
                         shape(basis_test))

  t5 = testCond(all(abs(basis - basis_test) < 1e-12))

  ! Test basis computation
  call compute_flux_moments()

  ! test scalar flux moments
  t6 = testCond(all(abs(d_phi - phi_m_test) < 1e-12))

  ! test angular flux moments
  t7 = testCond(all(abs(d_psi(:,1:1,:) - psi_m_test) < 1e-12))

  ! Get the cross section moments for order 0
  call compute_xs_moments(0)

  ! test source moments
  source_m_test = 1.0

  t8 = testCond(all(abs(d_source - source_m_test) < 1e-12))

  ! test total cross section moments
  t9 = testCond(all(abs(d_sig_t - sig_t(:,1:1)) < 1e-12))

  ! test angular cross section moments (delta)
  delta_m_test = 0.0

  t10 = testCond(all(abs(d_delta - delta_m_test) < 1e-12))

  ! test scattering cross section moments
  t11 = testCond(all(abs(d_sig_s(:,:,:,1) - sig_s(:,:,:,1)) < 1e-12))

  ! test fission cross section moments
  t12 = testCond(all(abs(d_nu_sig_f(:,1) - nu_sig_f(:,1) * phi(0,:,1) / d_phi(0,:,1)) < 1e-12))

  ! test chi moments
  t13 = testCond(all(abs(d_chi(:,1) - chi(:,1)) < 1e-12))

  if (t1 == 0) then
    print *, 'DGM2: order failed'
  else if (t2 == 0) then
    print *, 'DGM2: energy mesh failed'
  else if (t3 == 0) then
    print *, 'DGM2: expansion order failed'
  else if (t4 == 0) then
    print *, 'DGM2: number coarse groups failed'
  else if (t5 == 0) then
    print *, 'DGM2: basis failed'
  else if (t6 == 0) then
    print *, 'DGM2: phi moment failed'
  else if (t7 == 0) then
    print *, 'DGM2: psi moment failed'
  else if (t8 == 0) then
    print *, 'DGM2: source moment failed'
  else if (t9 == 0) then
    print *, 'DGM2: sig_t moment failed'
  else if (t10 == 0) then
    print *, 'DGM2: delta moment failed'
  else if (t11 == 0) then
    print *, 'DGM2: sig_s moment failed'
  else if (t12 == 0) then
    print *, 'DGM2: nu_sig_f moment failed'
  else if (t13 == 0) then
    print *, 'DGM2: chi moment failed'
  else
    print *, 'all tests passed for DGM2'
  end if

  call finalize_dgmsolver()
  call finalize_control()

end subroutine test2

! Test that providing truncation works as expected
subroutine test3()
  use control
  use material
  use angle, only : initialize_angle, initialize_polynomials
  use mesh, only : create_mesh
  use state, only : initialize_state, source
  use dgmsolver, only : finalize_dgmsolver
  use dgm, only : number_coarse_groups, basis, energymesh, expansion_order, order, &
                  initialize_moments, initialize_basis, compute_flux_moments, compute_xs_moments

  implicit none

  ! initialize types
  integer :: testCond
  integer :: t1=1, t2=1, t3=1, t4=1
  integer :: order_test(2), energyMesh_test(7), eo_test, ncg_test

  ! Define problem parameters
  call initialize_control('test/dgm_test_options5', .true.)

  order_test = [2,1]
  energyMesh_test = [1,1,1,1,2,2,2]
  eo_test = 2
  ncg_test = 2

  ! setup problem
  call create_mesh()
  call create_material()
  call initialize_angle()
  call initialize_polynomials(number_legendre)
  call initialize_state()

  ! test the moment construction
  call initialize_moments()

  t1 = testCond(all(order == order_test))
  t2 = testCond(all(energyMesh == energyMesh_test))
  t3 = testCond(expansion_order-eo_test == 0)
  t4 = testCond(number_coarse_groups-ncg_test == 0)

  if (t1 == 0) then
    print *, 'DGM3: order failed'
  else if (t2 == 0) then
    print *, 'DGM3: energy mesh failed'
  else if (t3 == 0) then
    print *, 'DGM3: expansion order failed'
  else if (t4 == 0) then
    print *, 'DGM3: number coarse groups failed'
  else
    print *, 'all tests passed for DGM3'
  end if

  call finalize_dgmsolver()

end subroutine test3

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
