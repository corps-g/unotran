module mesh
  ! ############################################################################
  ! Create the cell indexing for the problem
  ! ############################################################################

  use control, only : fine_mesh, coarse_mesh, material_map, boundary_type

  implicit none

  integer :: &
      number_cells  ! Total number of cells in the mesh
  double precision :: &
      width         ! Total width of the problem
  double precision, allocatable, dimension(:) :: &
      dx            ! Width of each cell
  integer, allocatable, dimension(:) :: &
      mMap          ! Material within each cell

  contains

  ! Compute the cell size and material map for the problem
  subroutine create_mesh()
    ! ##########################################################################
    ! Compute the widths of each cell and the total number of cells
    ! ##########################################################################

    double precision :: &
        ddx  ! Temporary variable for cell width
    integer :: &
        n, & ! number of coarse mesh regions
        c, & ! counting index for total cells
        i, & ! coarse cell index
        j    ! fine cell index

    n = size(fine_mesh)  ! Number of coarse mesh regions
    c = 1  ! counting variable
      
    number_cells = sum(fine_mesh)
    allocate(dx(number_cells), mMap(number_cells))
      
    do i = 1, n  ! loop over coarse mesh cells
      ! get fine difference
      ddx = (coarse_mesh(i+1) - coarse_mesh(i)) / fine_mesh(i)
      do j = 1, fine_mesh(i)  ! loop over fine mesh cells
        dx(c) = ddx  ! store cell size
        mMap(c) = material_map(i)  ! store material type
        c = c + 1
      end do
    end do

    ! Store the total width of the problem
    width = coarse_mesh(n + 1) - coarse_mesh(1)

  end subroutine create_mesh

  subroutine finalize_mesh()
    ! ##########################################################################
    ! Deallocate any used arrays
    ! ##########################################################################

    if (allocated(dx)) then
      deallocate(dx)
    end if
    if (allocated(mMap)) then
      deallocate(mMap)
    end if
  end subroutine finalize_mesh

end module mesh
