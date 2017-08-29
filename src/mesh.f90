module mesh
  use control, only : fine_mesh, course_mesh, material_map, boundary_type

  implicit none

  integer :: number_cells  ! Total number of cells in the mesh
  double precision :: width  ! Total width of the problem
  double precision, allocatable, dimension(:) :: dx  ! Width of each cell
  integer, allocatable, dimension(:) :: mMap  ! Material within each cell

  contains

  ! Compute the cell size and material map for the problem
  subroutine create_mesh()
    double precision :: width  ! Total width of problem
    double precision :: ddx  ! Temporary variable
    integer :: n, c, i, j

    n = size(fine_mesh)  ! Number of course mesh regions
    c = 1  ! counting variable
      
    number_cells = sum(fine_mesh)
    allocate(dx(number_cells), mMap(number_cells))
      
    do i = 1, n  ! loop over course mesh cells
      ! get fine difference
      ddx = (course_mesh(i+1) - course_mesh(i)) / fine_mesh(i)
      do j = 1, fine_mesh(i)  ! loop over fine mesh cells
        dx(c) = ddx  ! store cell size
        mMap(c) = material_map(i)  ! store material type
        c = c + 1
      end do
    end do

    ! Store the total width of the problem
    width = course_mesh(n) - course_mesh(1)

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
