program test_dgm

  use material, only : create_material
  use angle, only : initialize_angle, initialize_polynomials
  use mesh, only : create_mesh
  use state, only : initialize_state, source
  use dgm

  implicit none

  ! initialize types
  integer :: fineMesh(1), materialMap(1), testCond, em(1)
  integer :: t1=1, t2=1, t3=1, t4=1, t5=1, t6=1, t7=1, t8=1, t9=1, t10=1, t11=1
  double precision :: courseMesh(2), norm, error, basis_test(4,7)
  double precision :: phi_m_test(8,4,2,1), psi_m_test(2,4,1),source_m_test(4,2,4,1)
  double precision :: sig_t_m_test(2,1), delta_m_test(4,2,4,1), sig_s_m_test(8,4,2,2,1)
  integer :: order_test(2), energyMesh_test(7), eo_test, ncg_test
  ! Define problem parameters
  character(len=10) :: filename = 'test.anlxs'
  character(len=5) :: basisName = 'basis'

  phi_m_test = 0
  psi_m_test = 0
  source_m_test = 0
  fineMesh = [1]
  materialMap = [1]
  courseMesh = [0.0, 1.0]

  em = [4]
  order_test = [4, 3]
  energyMesh_test = [1,1,1,1,2,2,2]
  eo_test = 4
  ncg_test = 2

  ! setup problem
  call create_mesh(fineMesh, courseMesh, materialMap)
  call create_material(filename)
  call initialize_angle(2, 1)
  call initialize_polynomials(number_legendre)
  call initialize_state(.true., 'dd')
  source = 1.0

  ! test the moment construction
  call initialize_moments(em)

  t1 = testCond(all(order .eq. order_test))
  t2 = testCond(all(energyMesh .eq. energyMesh_test))
  t3 = testCond(expansion_order-eo_test .eq. 0)
  t4 = testCond(number_course_groups-ncg_test .eq. 0)

  ! Test reading the basis set from the file
  call initialize_basis(basisName)

  basis_test = reshape((/0.500000000,-0.670820390, 0.500000000,-0.22360680, &
                         0.500000000,-0.223606800,-0.500000000, 0.67082039, &
                         0.500000000, 0.223606800,-0.500000000,-0.67082039, &
                         0.500000000, 0.670820390, 0.500000000, 0.22360680, &
                         0.577350269, 0.707106781, 0.408248290, 0.00000000, &
                         0.577350269, 0.000000000,-0.816496581, 0.000000000, &
                         0.577350269,-0.707106781, 0.408248290, 0.000000000/), &
                         shape(basis_test))

  t5 = testCond(norm2(basis - basis_test) .lt. 1e-6)

  ! Test basis computation
  call compute_moments()

  ! test scalar flux moments
  phi_m_test = reshape((/2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, &
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
                         1.732050807, 1.732050807, 1.732050807, 1.732050807, &
                         1.732050807, 1.732050807, 1.732050807, 1.732050807, &
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0/), &
                        shape(phi_m_test))

  t6 = testCond(norm2(phi_moment - phi_m_test) .lt. 1e-6)

  ! test angular flux moments
  psi_m_test = reshape((/2.0, 1.732050807, &
                         2.0, 1.732050807, &
                         2.0, 1.732050807, &
                         2.0, 1.732050807/), &
                        shape(psi_m_test))
  
  t7 = testCond(norm2(psi_0_moment - psi_m_test) .lt. 1e-6)

  ! test source moments
  source_m_test = reshape((/2.0, 0.0, 0.0, 0.0, 1.732050807, 0.0, 0.0, 0.0, &
                         2.0, 0.0, 0.0, 0.0, 1.732050807, 0.0, 0.0, 0.0, &
                         2.0, 0.0, 0.0, 0.0, 1.732050807, 0.0, 0.0, 0.0, &
                         2.0, 0.0, 0.0, 0.0, 1.732050807, 0.0, 0.0, 0.0/), &
                        shape(source_m_test))

  t8 = testCond(norm2(source_moment - source_m_test) .lt. 1e-6)

  ! test total cross section moments
  sig_t_m_test = reshape((/0.376087, 1.0070863333/), shape(sig_t_m_test))

  t9 = testCond(norm2(sig_t_moment - sig_t_m_test) .lt. 1e-6)

  ! test angular cross section moments (delta)
  delta_m_test = reshape((/0.0, 0.1228424189, 0.000189, -0.0106680561, &
                           0.0, -0.4891647347, 0.2093483907, 0.0, &
                           0.0, 0.1228424189, 0.000189, -0.0106680561, &
                           0.0, -0.4891647347, 0.2093483907, 0.0, &
                           0.0, 0.1228424189, 0.000189, -0.0106680561, &
                           0.0, -0.4891647347, 0.2093483907, 0.0, &
                           0.0, 0.1228424189, 0.000189, -0.0106680561, &
                           0.0, -0.4891647347, 0.2093483907, 0.0/), shape(delta_m_test))

  t10 = testCond(norm2(delta_moment - delta_m_test) .lt. 1e-6)

  ! test scattering cross section moments
  sig_s_m_test = reshape((/0.3534278181, 0.0474363619,  0.0289331339,  0.0201344512,&
                          0.0145092473,  0.0093014956,  0.0056204421,  0.0030043368,&
                          0.108749275, -0.0329341039, -0.0289449757, -0.022444191,&
                          -0.0167896813, -0.0112993763, -0.0069750093, -0.0039106928,&
                          -0.0142640732, 0.0045275665,  0.0076277392,  0.0099931695,&
                          0.0084328274,  0.0066059506,  0.0043978712,  0.0026803334,&
                          -0.0140474438, 0.0065186048,  0.005287342, 9.30341132185895E-06,&
                          -0.0010863234, -0.0017467452, -0.0014185324, -0.0010077532,&
                          0.0, 0.0, 0.0, 0.0,&
                          0.0, 0.0, 0.0, 0.0,&
                          0.0, 0.0, 0.0, 0.0,&
                          0.0, 0.0, 0.0, 0.0,&
                          0.0, 0.0, 0.0, 0.0,&
                          0.0, 0.0, 0.0, 0.0,&
                          0.0, 0.0, 0.0, 0.0,&
                          0.0, 0.0, 0.0, 0.0,&
                          0.0015106139,  -0.0004682253, -0.000021059,  -6.49273679187172E-07,&
                          9.15469681138516E-07,  3.0289238507309E-06, -2.19342006865276E-06, 1.22453394059127E-06,&
                          0.0020267012,  -0.0006281901, -2.82536070321493E-05, -8.71092045378147E-07,&
                          1.22823145706903E-06,  4.06372775765521E-06,  -2.94278181177494E-06, 1.64288467119135E-06,&
                          0.0015106139,  -0.0004682253, -0.000021059,  -6.49273679187172E-07,&
                          9.15469681138516E-07,  3.0289238507309E-06, -2.19342006865276E-06, 1.22453394059127E-06,&
                          0.0006755671,  -0.0002093967, -9.41786915110972E-06, -2.9036401945454E-07,&
                          4.09410491792808E-07,  1.35457593941123E-06,  -9.80927285214446E-07, 5.47628231894008E-07,&
                          0.4236901533,  0.0053083109,  0.0016464802,  0.0012283233,&
                          0.0007583008,  -0.0006374103, -7.7602452E-05,  0.0004478039,&
                          -0.0379884016, 0.0038739129,  -0.0015822948, -0.0014793514,&
                          -0.0010802064, 0.0007826396,  0.0001042419,  -0.0005035108,&
                          0.0326467616,  -0.0018823973, 0.0009853706,  0.0007925234,&
                          0.0005038872,  -0.0003595438, -5.60788794899182E-05, 0.000263806,&
                          0.0, 0.0, 0.0, 0.0,&
                          0.0, 0.0, 0.0, 0.0/), shape(sig_s_m_test))
  print *, sig_s_moment

  print *

  print *, sig_s_moment - sig_s_m_test

  t11 = testCond(norm2(sig_s_moment - sig_s_m_test) .lt. 1e-6)

  if (t1 .eq. 0) then
    print *, 'DGM: order failed'
  else if (t2 .eq. 0) then
    print *, 'DGM: energy mesh failed'
  else if (t3 .eq. 0) then
    print *, 'DGM: expansion order failed'
  else if (t4 .eq. 0) then
    print *, 'DGM: number course groups failed'
  else if (t5 .eq. 0) then
    print *, 'DGM: basis failed'
  else if (t6 .eq. 0) then
    print *, 'DGM: phi moment failed'
  else if (t7 .eq. 0) then
    print *, 'DGM: psi moment failed'
  else if (t8 .eq. 0) then
    print *, 'DGM: source moment failed'
  else if (t9 .eq. 0) then
    print *, 'DGM: sig_t moment failed'
  else if (t10 .eq. 0) then
    print *, 'DGM: delta moment failed'
  else if (t11 .eq. 0) then
    print *, 'DGM: sig_s moment failed'
  else
    print *, 'all tests passed for DGM'
  end if

end program test_dgm

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
