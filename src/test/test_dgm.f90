program test_dgm

  use material, only : create_material
  use angle, only : initialize_angle, initialize_polynomials
  use mesh, only : create_mesh
  use state, only : initialize_state, source
  use dgm

  implicit none

  ! initialize types
  integer :: fineMesh(1), materialMap(1), testCond, em(1), l, i, c, cg, cgp,g,gp
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

  t5 = testCond(all((basis - basis_test) .lt. 1e-6))

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

  t6 = testCond(all((phi_moment - phi_m_test) .lt. 1e-6))

  ! test angular flux moments
  psi_m_test = reshape((/2.0, 1.732050807, &
                         2.0, 1.732050807, &
                         2.0, 1.732050807, &
                         2.0, 1.732050807/), &
                        shape(psi_m_test))
  
  t7 = testCond(all((psi_0_moment - psi_m_test) .lt. 1e-6))

  ! test source moments
  source_m_test = reshape((/2.0, 0.0, 0.0, 0.0, 1.732050807, 0.0, 0.0, 0.0, &
                         2.0, 0.0, 0.0, 0.0, 1.732050807, 0.0, 0.0, 0.0, &
                         2.0, 0.0, 0.0, 0.0, 1.732050807, 0.0, 0.0, 0.0, &
                         2.0, 0.0, 0.0, 0.0, 1.732050807, 0.0, 0.0, 0.0/), &
                        shape(source_m_test))

  t8 = testCond(all((source_moment - source_m_test) .lt. 1e-6))

  ! test total cross section moments
  sig_t_m_test = reshape((/0.376087, 1.0070863333/), shape(sig_t_m_test))

  t9 = testCond(all((sig_t_moment - sig_t_m_test) .lt. 1e-6))

  ! test angular cross section moments (delta)
  delta_m_test = reshape((/0.0, 0.1228424189, 0.000189, -0.0106680561, &
                           0.0, -0.4891647347, 0.2093483907, 0.0, &
                           0.0, 0.1228424189, 0.000189, -0.0106680561, &
                           0.0, -0.4891647347, 0.2093483907, 0.0, &
                           0.0, 0.1228424189, 0.000189, -0.0106680561, &
                           0.0, -0.4891647347, 0.2093483907, 0.0, &
                           0.0, 0.1228424189, 0.000189, -0.0106680561, &
                           0.0, -0.4891647347, 0.2093483907, 0.0/), shape(delta_m_test))

  t10 = testCond(all((delta_moment - delta_m_test) .lt. 1e-6))

  ! test scattering cross section moments
  sig_s_m_test = reshape((/ 3.53427818e-01,   4.74363619e-02,   2.89331339e-02,   2.01344512e-02,&
                            1.45092473e-02,   9.30149562e-03,   5.62044210e-03,   3.00433676e-03,&
                            1.08749275e-01,  -3.29341039e-02,  -2.89449757e-02,  -2.24441910e-02,&
                           -1.67896813e-02,  -1.12993763e-02,  -6.97500928e-03,  -3.91069276e-03,&
                           -1.42640732e-02,   4.52756651e-03,   7.62773916e-03,   9.99316946e-03,&
                            8.43282745e-03,   6.60595062e-03,   4.39787116e-03,   2.68033338e-03,&
                           -1.40474438e-02,   6.51860481e-03,   5.28734201e-03,   9.30341132e-06,&
                           -1.08632337e-03,  -1.74674520e-03,  -1.41853245e-03,  -1.00775318e-03,&
                            1.51061389e-03,  -4.68225295e-04,  -2.10589954e-05,  -6.49273679e-07,&
                            9.15469681e-07,   3.02892385e-06,  -2.19342007e-06,   1.22453394e-06,&
                            2.02670119e-03,  -6.28190150e-04,  -2.82536070e-05,  -8.71092045e-07,&
                            1.22823146e-06,   4.06372776e-06,  -2.94278181e-06,   1.64288467e-06,&
                            1.51061389e-03,  -4.68225295e-04,  -2.10589954e-05,  -6.49273679e-07,&
                            9.15469681e-07,   3.02892385e-06,  -2.19342007e-06,   1.22453394e-06,&
                            6.75567074e-04,  -2.09396720e-04,  -9.41786915e-06,  -2.90364019e-07,&
                            4.09410492e-07,   1.35457594e-06,  -9.80927285e-07,   5.47628232e-07,&
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                            4.23690153e-01,   5.30831090e-03,   1.64648023e-03,   1.22832329e-03,&
                            7.58300763e-04,  -6.37410267e-04,  -7.76024520e-05,   4.47803863e-04,&
                           -3.79884016e-02,   3.87391294e-03,  -1.58229485e-03,  -1.47935138e-03,&
                           -1.08020636e-03,   7.82639611e-04,   1.04241949e-04,  -5.03510771e-04,&
                            3.26467616e-02,  -1.88239731e-03,   9.85370646e-04,   7.92523436e-04,&
                            5.03887172e-04,  -3.59543834e-04,  -5.60788795e-05,   2.63805986e-04,&
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00 &
                           /), shape(sig_s_m_test))

  t11 = testCond(all((sig_s_moment - sig_s_m_test) .lt. 1e-7))

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
