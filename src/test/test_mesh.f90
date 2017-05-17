program test_mesh

use mesh

implicit none

! initialize types
  integer :: fineMesh(3), materialMap(3), number_cells_test, mMap_test(8)
  double precision :: courseMesh(4), dx_test(8), boundary(2)
  integer :: t1=1, t2=1, t3=1, testCond
  
  ! Define problem parameters
  fineMesh = [2, 4, 2]
  materialMap = [1,2,3]
  courseMesh = [0.0, 1.0, 2.0, 3.0]
  boundary = [0.0, 0.0]
  
  ! Make the mesh
  call create_mesh(fineMesh, courseMesh, materialMap, boundary)
  
  number_cells_test = 8
  dx_test = [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5]
  mMap_test = [1, 1, 2, 2, 2, 2, 3, 3]

  ! Test number of cells
  t1 = testCond(number_cells_test .eq. number_cells)
  
  ! test dx
  t2 = testCond(norm2(dx-dx_test) .lt. 1e-7)
  
  ! test material map
  t3 = testCond(ALL(mMap.eq.mMap_test))
  
  ! Print appropriate output statements
  if (t1 .eq. 0) then
    print *, 'mesh: number_cells failed'
  else if (t2 .eq. 0) then
    print *, 'mesh: dx failed'
  else if (t3 .eq. 0) then
    print *, 'mesh: mMap failed'
  else
    print *, 'all tests passed for mesh'
  end if

end program test_mesh

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
