module dgm
  ! ############################################################################
  ! Compute the DGM Moments for fluxes and cross sections
  ! ############################################################################

  implicit none

  double precision, allocatable, dimension(:,:) :: &
      basis                ! Basis for expansion in energy
  double precision, allocatable, dimension(:,:,:) :: &
      chi_m,             & ! Chi spectrum moments
      phi_m_zero,        & ! Zeroth moment of scalar flux
      psi_m_zero           ! Zeroth moment of angular flux
  double precision, allocatable, dimension(:,:,:,:) :: &
      delta_m,           & ! Angular total XS moments
      source_m             ! Source moments
  double precision, allocatable, dimension(:,:,:,:,:) :: &
      sig_s_m              ! Scattering XS moments
  integer :: &
      expansion_order      ! Maximum expansion order
  integer, allocatable, dimension(:) :: &
      energyMesh,        & ! Array of number of fine groups per coarse group
      order,             & ! Expansion order for each coarse energy group
      basismap             ! Starting index for fine group for each coarse group

  contains

  subroutine initialize_moments()
    ! ##########################################################################
    ! Initialize the container for the cross section and flux moments
    ! ##########################################################################

    ! Use Statements
    use control, only : energy_group_map, truncation_map, number_angles, &
                        number_groups, number_coarse_groups, number_cells, &
                        number_legendre, number_fine_groups

    ! Variable definitions
    integer :: &
        g,  & ! outer fine group index
        gp, & ! inner coarse group index
        cg    ! coarse group index

    ! Get the number of coarse groups
    if (allocated(energy_group_map)) then
      number_coarse_groups = size(energy_group_map) + 1
    else
      number_coarse_groups = 1
    end if
    number_groups = number_coarse_groups

    ! Create the map of coarse groups and default to full expansion order
    allocate(energyMesh(number_fine_groups))
    allocate(order(number_coarse_groups))
    allocate(basismap(number_coarse_groups + 1))

    ! Load the group map
    basismap(1) = 0
    if (number_coarse_groups > 1) then
      basismap(2:number_coarse_groups) = energy_group_map
    end if
    basismap(number_coarse_groups + 1) = number_fine_groups

    do cg = 1, number_coarse_groups
      ! Create the energy map
      do g = basismap(cg), basismap(cg+1) - 1
        energyMesh(g + 1) = cg
      end do
      ! Get the DOF for full order
      order(cg) = basismap(cg+1) - basismap(cg) - 1
    end do

    ! Check if the optional argument for the truncation is present
    if (allocated(truncation_map)) then
      ! Update the order array with the truncated value if sensible
      do cg = 1, number_coarse_groups
        if ((truncation_map(cg) < order(cg)) .and. (truncation_map(cg) >= 0)) then
          order(cg) = truncation_map(cg)
        end if
      end do
    end if

    expansion_order = MAXVAL(order)

    ! Form the containers to hold the zeroth moments
    allocate(phi_m_zero(0:number_legendre, number_cells, number_coarse_groups))
    allocate(psi_m_zero(number_cells, 2 * number_angles, number_coarse_groups))

  end subroutine initialize_moments

  subroutine initialize_basis()
    ! ##########################################################################
    ! Load basis set from file
    ! ##########################################################################

    ! Use Statements
    use control, only : dgm_basis_name, number_fine_groups, number_coarse_groups

    ! Variable definitions
    double precision, allocatable, dimension(:) :: &
        array1 ! Temporary array
    integer :: &
        g,   & ! Fine group index
        cg,  & ! Coarse group index
        i      ! Cell index

    ! allocate the basis array
    allocate(basis(number_fine_groups, 0:expansion_order))
    allocate(array1(number_fine_groups))

    ! initialize the basis to zero
    basis = 0.0

    ! open the file and read into the basis container
    open(unit=5, file=dgm_basis_name)
    do g = 1, number_fine_groups
      cg = energyMesh(g)
      array1(:) = 0.0
      read(5,*) array1
      do i = 0, order(cg)
        basis(g, i) = array1(1 + basismap(cg) + i)
      end do
    end do

    ! clean up
    close(unit=5)

    deallocate(array1)

  end subroutine initialize_basis

  subroutine finalize_moments()
    ! ##########################################################################
    ! Deallocate the variable containers
    ! ##########################################################################

    if (allocated(basis)) then
      deallocate(basis)
    end if
    if (allocated(order)) then
      deallocate(order)
    end if
    if (allocated(energyMesh)) then
      deallocate(energyMesh)
    end if
    if (allocated(basismap)) then
      deallocate(basismap)
    end if
    if (allocated(chi_m)) then
      deallocate(chi_m)
    end if
    if (allocated(phi_m_zero)) then
      deallocate(phi_m_zero)
    end if
    if (allocated(psi_m_zero)) then
      deallocate(psi_m_zero)
    end if
    if (allocated(source_m)) then
      deallocate(source_m)
    end if
    if (allocated(delta_m)) then
      deallocate(delta_m)
    end if
    if (allocated(sig_s_m)) then
      deallocate(sig_s_m)
    end if
  end subroutine finalize_moments

end module dgm
