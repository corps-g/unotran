module mesh
  implicit none

  integer :: number_cells  ! Total number of cells in the mesh
  double precision :: width  ! Total width of the problem
  double precision, allocatable, dimension(:) :: dx  ! Width of each cell
  integer, allocatable, dimension(:) :: mMap  ! Material within each cell

  contains

  ! Compute the cell size and material map for the problem
  subroutine create_mesh(fineMesh, courseMesh, materialMap)
    ! Inputs :
    !   fineMesh : vector of int for number of fine mesh divisions per cell
    !   courseMap : vector of float with bounds for course mesh regions
    !   materialMap : vector of int with material index for each course region

    integer, intent(in) :: fineMesh(:), materialMap(:)
    double precision, intent(in) :: courseMesh(:)

    double precision :: width  ! Total width of problem
    double precision :: ddx  ! Temporary variable
    integer :: n, c, i, j

    n = size(fineMesh)  ! Number of course mesh regions
    c = 1  ! counting variable
      
    number_cells = sum(fineMesh)
    allocate(dx(number_cells), mMap(number_cells))
      
    do i = 1, n  ! loop over course mesh cells
      ! get fine difference
      ddx = (courseMesh(i+1) - courseMesh(i)) / fineMesh(i)
      do j = 1, fineMesh(i)  ! loop over fine mesh cells
        dx(c) = ddx  ! store cell size
        mMap(c) = materialMap(i)  ! store material type
        c = c + 1
      end do
    end do

    ! Store the total width of the problem
    width = courseMesh(n) - courseMesh(1)
  end subroutine create_mesh

  subroutine finalize_mesh()
    if (allocated(dx)) then
      deallocate(dx)
    end if
    if (allocated(mMap)) then
      deallocate(mMap)
    end if
  end subroutine finalize_mesh

end module mesh
