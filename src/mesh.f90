module mesh

integer :: nCells  ! Total number of cells in the mesh
double precision, allocatable, dimension(:) :: dx  ! Width of each cell
integer, allocatable, dimension(:) :: mMap  ! Material within each cell

contains

subroutine create(fineMesh, courseMesh, materialMap)
  integer, intent(in) :: fineMesh(:)
  integer, intent(in) :: materialMap(:)
  double precision, intent(in) :: courseMesh(:)
    
  double precision :: width  ! Total width of problem
  double precision :: ddx  ! Temperary variable
  integer :: n, c
  n = size(fineMesh)  ! Number of course mesh regions
  c = 1  ! counting variable
    
  nCells = sum(fineMesh)
  allocate(dx(nCells), mMap(nCells))
    
  do i = 1, n
    ddx = (courseMesh(i+1) - courseMesh(i)) / fineMesh(i)
    do j = 1, fineMesh(i)
      dx(c) = ddx
      mMap(c) = materialMap(i)
      c = c + 1
    end do
  end do
  width = courseMesh(n) - courseMesh(1)
end subroutine create

end module mesh
