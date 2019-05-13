module mesh
  ! ############################################################################
  ! Create the cell indexing for the problem
  ! ############################################################################

  implicit none

  double precision :: &
      width_x,   & ! Total width of the problem in x direction
      width_y      ! Total width of the problem in y direction
  double precision, allocatable, dimension(:) :: &
      dx,         & ! Width of each cell in EW direction
      dy            ! Width of each cell in NS direction
  integer, allocatable, dimension(:) :: &
      mMap          ! Material within each cell

  contains

  ! Compute the cell size and material map for the problem
  subroutine create_mesh()
    ! ##########################################################################
    ! Compute the widths of each cell and the total number of cells
    ! ##########################################################################

    ! Use Statements
    use control, only : number_cells, number_cells_x, number_cells_y, fine_mesh_x, &
                        fine_mesh_y, coarse_mesh_x, coarse_mesh_y, material_map, &
                        spatial_dimension, boundary_north, boundary_south

    ! Variable definitions
    double precision :: &
        ddx, & ! Temporary variable for x cell width
        ddy    ! Temporary variable for y cell width
    integer :: &
        nx,  & ! number of coarse mesh regions in x direction
        ny,  & ! number of coarse mesh regions in y direction
        c,   & ! counting index for total cells
        cx,  & ! counting index for x cells
        cy,  & ! counting index for y cells
        ix,  & ! coarse cell index for x cells
        iy,  & ! coarse cell index for y cells
        jx,  & ! fine cell index for x cells
        jy     ! fine cell index for y cells

    ! Check for 1D problem
    if (spatial_dimension == 1) then
      ! Set the y direction to a single spatial cell with reflective contitions
      boundary_north = 1.0
      boundary_south = 1.0

      if (allocated(fine_mesh_y)) deallocate(fine_mesh_y)
      if (allocated(coarse_mesh_y)) deallocate(coarse_mesh_y)

      allocate(fine_mesh_y(1), coarse_mesh_y(2))
      fine_mesh_y(1) = 1
      coarse_mesh_y = [0.0, 1.0]

    end if

    nx = size(fine_mesh_x)  ! Number of coarse mesh regions in x
    ny = size(fine_mesh_y)  ! Number of coarse mesh regions in x
      
    number_cells_x = sum(fine_mesh_x)
    number_cells_y = sum(fine_mesh_y)
    number_cells = number_cells_x * number_cells_y
    allocate(dx(number_cells_x), dy(number_cells_y), mMap(number_cells))

    ! Compute the mesh widths and read materials
    c = 1  ! Initialize counting variable for all cells
    cy = 1  ! Initialize counting variable for y cells

    do iy = 1, ny  ! Loop over the coarse y cells
      ddy = (coarse_mesh_y(iy+1) - coarse_mesh_y(iy)) / fine_mesh_y(iy)
      do jy = 1, fine_mesh_y(iy)  ! Loop over the fine y cells in region iy
        cx = 1  ! Initialize counting variable for x cells
        do ix = 1, nx  ! Loop over the coarse x cells
          ddx = (coarse_mesh_x(ix+1) - coarse_mesh_x(ix)) / fine_mesh_x(ix)
          do jx = 1, fine_mesh_x(ix)  ! Loop over the fine x cells in region ix
            mMap(c) = material_map((iy - 1) * nx + ix)
            dx(cx) = ddx  ! Store cell size in x direction
            cx = cx + 1  ! Increment x cells
            c = c + 1  ! Increment all cells
          end do  ! End jx loop
        end do  ! End ix loop
        dy(cy) = ddy  ! Store cell size in y direction
        cy = cy + 1  ! Increment y cells
      end do  ! End jy loop
    end do  ! End iy loop

    ! Store the total width of the problem
    width_x = coarse_mesh_x(nx + 1) - coarse_mesh_x(1)
    width_y = coarse_mesh_y(ny + 1) - coarse_mesh_y(1)

  end subroutine create_mesh

  subroutine finalize_mesh()
    ! ##########################################################################
    ! Deallocate any used arrays
    ! ##########################################################################

    if (allocated(dx)) then
      deallocate(dx)
    end if
    if (allocated(dy)) then
      deallocate(dy)
    end if
    if (allocated(mMap)) then
      deallocate(mMap)
    end if
  end subroutine finalize_mesh

end module mesh
