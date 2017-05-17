program test_dgm
  call test1()
  call test2()
  call test3()
end program test_dgm

! Test that the DGM moments are correctly computed for a flux of one
subroutine test1()
  use material, only : create_material
  use angle, only : initialize_angle, initialize_polynomials
  use mesh, only : create_mesh
  use state, only : initialize_state, source
  use solver, only : finalize_solver
  use dgm

  implicit none

  ! initialize types
  integer :: fineMesh(1), materialMap(1), testCond, em(1), l, i, c, cg, cgp,g,gp
  integer :: t1=1, t2=1, t3=1, t4=1, t5=1, t6=1, t7=1, t8=1, t9=1, t10=1, t11=1, t12=1, t13=1
  double precision :: courseMesh(2), norm, error, basis_test(4,7), boundary(2)
  double precision :: phi_m_test(8,4,2,1), psi_m_test(2,4,1),source_m_test(4,2,4,1)
  double precision :: sig_t_m_test(2,1), delta_m_test(4,2,4,1), sig_s_m_test(8,4,2,2,1)
  double precision :: nu_sig_f_m_test(2,1), chi_m_test(4,2,1)
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
  call create_mesh(fineMesh, courseMesh, materialMap, boundary)
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
  delta_m_test = reshape((/ 0.00000000e+00,   2.45684838e-01,   3.78000000e-04,  -2.13361123e-02,&
                            0.00000000e+00,  -8.47258173e-01,   3.62602049e-01,   0.00000000e+00,&
                            0.00000000e+00,   2.45684838e-01,   3.78000000e-04,  -2.13361123e-02,&
                            0.00000000e+00,  -8.47258173e-01,   3.62602049e-01,   0.00000000e+00,&
                            0.00000000e+00,   2.45684838e-01,   3.78000000e-04,  -2.13361123e-02,&
                            0.00000000e+00,  -8.47258173e-01,   3.62602049e-01,   0.00000000e+00,&
                            0.00000000e+00,   2.45684838e-01,   3.78000000e-04,  -2.13361123e-02,&
                            0.00000000e+00,  -8.47258173e-01,   3.62602049e-01,   0.00000000e+00&
                            /), shape(delta_m_test))

  t10 = testCond(all((delta_moment - delta_m_test) .lt. 1e-6))

  ! test scattering cross section moments
  sig_s_m_test = reshape((/  7.06855636e-01,   9.48727237e-02,   5.78662679e-02,   4.02689024e-02,&
                             2.90184945e-02,   1.86029912e-02,   1.12408842e-02,   6.00867352e-03,&
                             2.17498550e-01,  -6.58682078e-02,  -5.78899515e-02,  -4.48883819e-02,&
                            -3.35793625e-02,  -2.25987526e-02,  -1.39500186e-02,  -7.82138551e-03,&
                            -2.85281464e-02,   9.05513302e-03,   1.52554783e-02,   1.99863389e-02,&
                             1.68656549e-02,   1.32119012e-02,   8.79574232e-03,   5.36066677e-03,&
                            -2.80948877e-02,   1.30372096e-02,   1.05746840e-02,   1.86068226e-05,&
                            -2.17264674e-03,  -3.49349039e-03,  -2.83706489e-03,  -2.01550636e-03,&
                             2.61646000e-03,  -8.10990000e-04,  -3.64752500e-05,  -1.12457500e-06,&
                             1.58564000e-06,   5.24625000e-06,  -3.79911500e-06,   2.12095500e-06,&
                             3.51034944e-03,  -1.08805726e-03,  -4.89366829e-05,  -1.50877568e-06,&
                             2.12735929e-06,   7.03858294e-06,  -5.09704761e-06,   2.84555972e-06,&
                             2.61646000e-03,  -8.10990000e-04,  -3.64752500e-05,  -1.12457500e-06,&
                             1.58564000e-06,   5.24625000e-06,  -3.79911500e-06,   2.12095500e-06,&
                             1.17011650e-03,  -3.62685757e-04,  -1.63122279e-05,  -5.02925234e-07,&
                             7.09119773e-07,   2.34619435e-06,  -1.69901590e-06,   9.48519921e-07,&
                             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                             7.33852872e-01,   9.19426418e-03,   2.85178742e-03,   2.12751835e-03,&
                             1.31341545e-03,  -1.10402697e-03,  -1.34411390e-04,   7.75619043e-04,&
                            -6.57978416e-02,   6.70981403e-03,  -2.74061507e-03,  -2.56231174e-03,&
                            -1.87097230e-03,   1.35557157e-03,   1.80552351e-04,  -8.72106238e-04,&
                             5.65458497e-02,  -3.26040778e-03,   1.70671202e-03,   1.37269086e-03,&
                             8.72758183e-04,  -6.22748188e-04,  -9.71314685e-05,   4.56925371e-04,&
                             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,&
                             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00 &
                            /), shape(sig_s_m_test))

  t11 = testCond(all((sig_s_moment - sig_s_m_test) .lt. 1e-7))

  ! test fission cross section moments
  nu_sig_f_m_test = reshape((/0.07849183, 2.6060216/), shape(nu_sig_f_m_test))

  t12 = testCond(all((nu_sig_f_moment - nu_sig_f_m_test) .lt. 1e-6))

  ! test chi spectrum moments
  chi_m_test = reshape((/0.4999999996, -0.2595977384, -0.3848769417, 0.5216659709,&
                         0.0, 0.0, 0.0, 0.0/), shape(chi_m_test))

  t13 = testCond(all((chi_moment - chi_m_test) .lt. 1e-6))

  if (t1 .eq. 0) then
    print *, 'DGM1: order failed'
  else if (t2 .eq. 0) then
    print *, 'DGM1: energy mesh failed'
  else if (t3 .eq. 0) then
    print *, 'DGM1: expansion order failed'
  else if (t4 .eq. 0) then
    print *, 'DGM1: number course groups failed'
  else if (t5 .eq. 0) then
    print *, 'DGM1: basis failed'
  else if (t6 .eq. 0) then
    print *, 'DGM1: phi moment failed'
  else if (t7 .eq. 0) then
    print *, 'DGM1: psi moment failed'
  else if (t8 .eq. 0) then
    print *, 'DGM1: source moment failed'
  else if (t9 .eq. 0) then
    print *, 'DGM1: sig_t moment failed'
  else if (t10 .eq. 0) then
    print *, 'DGM1: delta moment failed'
  else if (t11 .eq. 0) then
    print *, 'DGM1: sig_s moment failed'
  else if (t12 .eq. 0) then
    print *, 'DGM1: nu_sig_f moment failed'
  else if (t13 .eq. 0) then
    print *, 'DGM1: chi moment failed'
  else
    print *, 'all tests passed for DGM1'
  end if

  call finalize_solver()
end subroutine test1

! Test that the DO method is matched if using the delta basis and 1 fine per course energy group
subroutine test2()
  use material
  use angle, only : initialize_angle, initialize_polynomials
  use mesh, only : create_mesh
  use state, only : initialize_state, source
  use solver, only : finalize_solver
  use dgm

  implicit none

  ! initialize types
  integer :: fineMesh(1), materialMap(1), testCond, em(6), l, i, c, cg, cgp,g,gp
  integer :: t1=1, t2=1, t3=1, t4=1, t5=1, t6=1, t7=1, t8=1, t9=1, t10=1, t11=1, t12=1, t13=1
  double precision :: courseMesh(2), norm, error, basis_test(1,7), boundary(2)
  double precision :: phi_m_test(0:7,1,7,1), psi_m_test(7,1,1),source_m_test(4,2,4,1)
  double precision :: phi_test(0:7,7,1), psi_test(7,4,1)
  double precision :: sig_t_m_test(2,1), delta_m_test(4,2,4,1), sig_s_m_test(8,4,2,2,1)
  double precision :: nu_sig_f_m_test(7,1), chi_m_test(1,7,1)
  integer :: order_test(7), energyMesh_test(7), eo_test, ncg_test
  ! Define problem parameters
  character(len=10) :: filename = 'test.anlxs'
  character(len=10) :: basisName = 'deltaBasis'

  phi_m_test = 0
  psi_m_test = 0
  source_m_test = 0
  fineMesh = [1]
  materialMap = [1]
  courseMesh = [0.0, 1.0]
  boundary = [0.0, 0.0]

  em = [1,2,3,4,5,6]
  order_test = [1,1,1,1,1,1,1]
  energyMesh_test = [1,2,3,4,5,6,7]
  eo_test = 1
  ncg_test = 7

  ! setup problem
  call create_mesh(fineMesh, courseMesh, materialMap, boundary)
  call create_material(filename)
  call initialize_angle(2, 1)
  call initialize_polynomials(number_legendre)
  call initialize_state(.true., 'dd')
  source = 1.0
  phi_m_test = reshape((/1.05033726e+00,  -1.38777878e-17,  -1.57267763e-01,   0.00000000e+00,&
                         1.43689742e-17,   0.00000000e+00,   9.88540222e-02,   6.93889390e-18,&
                         6.32597614e-02,  -8.67361738e-19,  -8.69286487e-03,   0.00000000e+00,&
                         6.81181057e-19,   2.16840434e-19,   5.46408649e-03,   0.00000000e+00,&
                         5.78458741e-03,   0.00000000e+00,  -7.15385990e-04,   0.00000000e+00,&
                         4.34869208e-20,   5.42101086e-20,   4.49671194e-04,  -5.42101086e-20,&
                         3.20772663e-05,  -4.23516474e-22,  -3.67904369e-06,  -4.23516474e-22,&
                         1.73044862e-22,   1.05879118e-22,   2.31254175e-06,   0.00000000e+00,&
                         1.55685757e-07,  -3.30872245e-24,  -1.78930684e-08,   8.27180613e-25,&
                         8.48610071e-25,   8.27180613e-25,   1.12470715e-08,   8.27180613e-25,&
                         7.86105345e-10,   0.00000000e+00,  -8.28643069e-11,   6.46234854e-27,&
                         2.51523310e-27,   3.23117427e-27,   5.20861358e-11,   6.46234854e-27,&
                         3.40490800e-13,  -6.31088724e-30,  -2.37767941e-14,  -1.57772181e-30,&
                        -1.77547007e-30,   1.57772181e-30,   1.49454134e-14,   3.15544362e-30 &
                       /), shape(phi_m_test))

  psi_m_test = reshape((/0.66243843,  1.46237369,  0.73124892,  0.66469264,&
                       0.60404633,  0.59856738,  0.31047097,  1.52226789,&
                       3.07942675,  1.4095816 ,  1.23433893,  1.11360585,&
                       1.04656294,  0.44397067,  1.52226789,  3.07942675,&
                       1.4095816 ,  1.23433893,  1.11360585,  1.04656294,&
                       0.44397067,  0.66243843,  1.46237369,  0.73124892,&
                       0.66469264,  0.60404633,  0.59856738,  0.31047097&
                       /), shape(psi_m_test))

  ! test the moment construction
  call initialize_moments(em)

  t1 = testCond(all(order .eq. order_test))
  t2 = testCond(all(energyMesh .eq. energyMesh_test))
  t3 = testCond(expansion_order-eo_test .eq. 0)
  t4 = testCond(number_course_groups-ncg_test .eq. 0)

  ! set phi and psi
  phi = phi_m_test(:,1,:,:)
  psi = psi_m_test


  ! Test reading the basis set from the file
  call initialize_basis(basisName)

  basis_test = reshape((/1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0/), &
                         shape(basis_test))

  t5 = testCond(all((basis - basis_test) .lt. 1e-6))

  ! Test basis computation
  call compute_moments()

  ! test scalar flux moments
  t6 = testCond(all((phi_moment - phi_m_test) .lt. 1e-6))

  ! test angular flux moments
  t7 = testCond(all((psi_0_moment - psi_m_test) .lt. 1e-6))

  ! test source moments
  source_m_test = 1.0

  t8 = testCond(all((source_moment - source_m_test) .lt. 1e-6))

  ! test total cross section moments
  t9 = testCond(all((sig_t_moment - sig_t) .lt. 1e-6))

  ! test angular cross section moments (delta)
  delta_m_test = 0.0

  t10 = testCond(all((delta_moment - delta_m_test) .lt. 1e-6))

  ! test scattering cross section moments
  do g = 1, number_groups
    do gp = 1, number_groups
      do l = 0, number_legendre
        if (phi_moment(l,1,gp,1) == 0) then
          sig_s(l,gp,g,1) = 0.0
        else
          sig_s_moment(l,1,gp,g,1) = sig_s_moment(l,1,gp,g,1) / phi_moment(l,1,gp,1)
        end if
      end do
    end do
  end do

  t11 = testCond(all((sig_s_moment(:,1,:,:,:) - sig_s) .lt. 1e-7))

  ! test fission cross section moments
  t12 = testCond(all((nu_sig_f_moment(:,1) - nu_sig_f(:,1)*phi(0,:,1)) .lt. 1e-6))

  ! test chi moments
  t13 = testCond(all((chi_moment(1,:,1) - chi(:,1)) .lt. 1e-6))

  if (t1 .eq. 0) then
    print *, 'DGM2: order failed'
  else if (t2 .eq. 0) then
    print *, 'DGM2: energy mesh failed'
  else if (t3 .eq. 0) then
    print *, 'DGM2: expansion order failed'
  else if (t4 .eq. 0) then
    print *, 'DGM2: number course groups failed'
  else if (t5 .eq. 0) then
    print *, 'DGM2: basis failed'
  else if (t6 .eq. 0) then
    print *, 'DGM2: phi moment failed'
  else if (t7 .eq. 0) then
    print *, 'DGM2: psi moment failed'
  else if (t8 .eq. 0) then
    print *, 'DGM2: source moment failed'
  else if (t9 .eq. 0) then
    print *, 'DGM2: sig_t moment failed'
  else if (t10 .eq. 0) then
    print *, 'DGM2: delta moment failed'
  else if (t11 .eq. 0) then
    print *, 'DGM2: sig_s moment failed'
  else if (t12 .eq. 0) then
    print *, 'DGM2: nu_sig_f moment failed'
  else if (t13 .eq. 0) then
    print *, 'DGM2: chi moment failed'
  else
    print *, 'all tests passed for DGM2'
  end if

  call finalize_solver()

end subroutine test2

! Test that providing truncation works as expected
subroutine test3()
  use material
  use angle, only : initialize_angle, initialize_polynomials
  use mesh, only : create_mesh
  use state, only : initialize_state, source
  use solver, only : finalize_solver
  use dgm

  implicit none

  ! initialize types
  integer :: fineMesh(1), materialMap(1), testCond, em(1), l, i, c, cg, cgp,g,gp
  integer :: t1=1, t2=1, t3=1, t4=1, t5=1, t6=1, t7=1, t8=1, t9=1, t10=1, t11=1, t12=1, t13=1
  double precision :: courseMesh(2), norm, error, basis_test(1,7), boundary(2)
  double precision :: phi_m_test(0:7,1,7,1), psi_m_test(7,1,1),source_m_test(4,2,4,1)
  double precision :: phi_test(0:7,7,1), psi_test(7,4,1)
  double precision :: sig_t_m_test(2,1), delta_m_test(4,2,4,1), sig_s_m_test(8,4,2,2,1)
  double precision :: nu_sig_f_m_test(7,1), chi_m_test(1,7,1)
  integer :: order_test(2), energyMesh_test(7), eo_test, ncg_test
  ! Define problem parameters
  character(len=10) :: filename = 'test.anlxs'
  character(len=5) :: basisName = 'basis'

  fineMesh = [1]
  materialMap = [1]
  courseMesh = [0.0, 1.0]
  boundary = [0.0, 0.0]

  em = [4]
  order_test = [3,2]
  energyMesh_test = [1,1,1,1,2,2,2]
  eo_test = 3
  ncg_test = 2

  ! setup problem
  call create_mesh(fineMesh, courseMesh, materialMap, boundary)
  call create_material(filename)
  call initialize_angle(2, 1)
  call initialize_polynomials(number_legendre)
  call initialize_state(.true., 'dd')
  source = 1.0

  ! test the moment construction
  call initialize_moments(em, [3,2])

  t1 = testCond(all(order .eq. order_test))
  t2 = testCond(all(energyMesh .eq. energyMesh_test))
  t3 = testCond(expansion_order-eo_test .eq. 0)
  t4 = testCond(number_course_groups-ncg_test .eq. 0)

  if (t1 .eq. 0) then
    print *, 'DGM3: order failed'
  else if (t2 .eq. 0) then
    print *, 'DGM3: energy mesh failed'
  else if (t3 .eq. 0) then
    print *, 'DGM3: expansion order failed'
  else if (t4 .eq. 0) then
    print *, 'DGM3: number course groups failed'
  else
    print *, 'all tests passed for DGM3'
  end if

  call finalize_solver()

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
