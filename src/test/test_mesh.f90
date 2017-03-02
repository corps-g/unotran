program test_mesh

use mesh

implicit none

! initialize types
  integer :: fineMesh(3), materialMap(3), number_cells_test, mMap_test(8)
  double precision :: courseMesh(4), dx_test(8)
  integer :: t1=1, t2=1, t3=1
  
  ! Define problem parameters
  fineMesh = [2, 4, 2]
  materialMap = [1,2,3]
  courseMesh = [0.0, 1.0, 2.0, 3.0]
  
  ! Make the mesh
  call create_mesh(fineMesh, courseMesh, materialMap)
  
  number_cells_test = 8
  dx_test = [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5]
  mMap_test = [1, 1, 2, 2, 2, 2, 3, 3]

  ! Test number of cells
  if (number_cells_test .eq. number_cells) then
    write(*,"(A)",advance="no") '.'
  else
    write(*,"(A)",advance="no") 'F'
    t1=0
  end if
  
  ! test dx
  if (norm2(dx-dx_test) .lt. 1e-7) then
    write(*,"(A)",advance="no") '.'
  else
    write(*,"(A)",advance="no") 'F'
    t2 = 0
  end if
  
  if (ALL(mMap.eq.mMap_test)) then
    write(*,"(A)") '.'
  else
    write(*,"(A)",advance="no") 'F'
    t3=0
  end if
  
  if (t1.eq.0) then
    print *, 'mesh: number_cells failed'
  else if (t2 .eq. 0) then
    print *, 'mesh: dx failed'
  else if (t3 .eq. 0) then
    print *, 'mesh: mMap failed'
  end if

end program test_mesh
