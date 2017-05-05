program test_dgm

  use material, only : create_material
  use angle, only : initialize_angle, initialize_polynomials
  use mesh, only : create_mesh
  use state, only : initialize_state, source
  use dgm

  implicit none

  ! initialize types
  integer :: fineMesh(1), materialMap(1), testCond, em(1)
  integer :: t1=1, t2=1, t3=1, t4=1, t5=1, t6=1, t7=1, t8=1
  double precision :: courseMesh(2), norm, error, basis_test(4,7)
  double precision :: phi_m_test(8,4,2,1), psi_m_test(2,4,1),source_m_test(4,2,4,1)
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

  psi_m_test = reshape((/2.0, 1.732050807, &
                         2.0, 1.732050807, &
                         2.0, 1.732050807, &
                         2.0, 1.732050807/), &
                        shape(psi_m_test))
  
  t7 = testCond(norm2(psi_0_moment - psi_m_test) .lt. 1e-6)

  source_m_test = reshape((/2.0, 0.0, 0.0, 0.0, 1.732050807, 0.0, 0.0, 0.0, &
                         2.0, 0.0, 0.0, 0.0, 1.732050807, 0.0, 0.0, 0.0, &
                         2.0, 0.0, 0.0, 0.0, 1.732050807, 0.0, 0.0, 0.0, &
                         2.0, 0.0, 0.0, 0.0, 1.732050807, 0.0, 0.0, 0.0/), &
                        shape(source_m_test))

  t8 = testCond(norm2(source_moment - source_m_test) .lt. 1e-6)

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
