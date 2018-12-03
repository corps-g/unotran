module dgm
  ! ############################################################################
  ! Compute the DGM Moments for fluxes and cross sections
  ! ############################################################################

  implicit none

  double precision, allocatable, dimension(:,:) :: &
      basis,                & ! Basis for expansion in energy
      source_m                ! Source moments
  double precision, allocatable, dimension(:,:,:) :: &
      chi_m,                & ! Chi spectrum moments
      expanded_nu_sig_f       ! Expanded fission cross sections
  double precision, allocatable, dimension(:,:,:,:) :: &
      delta_m,              & ! Angular total XS moments
      phi_m,                & ! Scalar flux moments
      psi_m,                & ! Angular flux moments
      expanded_sig_t          ! Expanded total cross sections
  double precision, allocatable, dimension(:,:,:,:,:) :: &
      sig_s_m                 ! Scattering XS moments
  double precision, allocatable, dimension(:,:,:,:,:,:) :: &
      expanded_sig_s          ! Expanded scattering cross sections
  integer :: &
      expansion_order,      & ! Maximum expansion order
      dgm_order=0             ! Current order
  integer, allocatable, dimension(:) :: &
      order,                & ! Expansion order for each coarse energy group
      basismap                ! Starting index for fine group for each coarse group

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
        cg    ! coarse group index

    ! Get the number of coarse groups
    if (.not. allocated(energy_group_map)) then
      allocate(energy_group_map(number_fine_groups))
      energy_group_map = 1
    end if

    number_coarse_groups = maxval(energy_group_map)
    number_groups = number_coarse_groups

    ! Create the map of coarse groups and default to full expansion order
    allocate(order(number_coarse_groups))
    allocate(basismap(number_coarse_groups))

    order = 0
    ! Compute the order within each coarse group
    do g = 1, number_fine_groups
      order(energy_group_map(g)) = order(energy_group_map(g)) + 1
    end do

    ! The basis map is the index to start reading the basis from the file
    basisMap(1) = 1
    do cg = 2, number_coarse_groups
      basisMap(cg) = sum(order(1:(cg-1))) + 1
    end do

    ! Order begins at zero, so need to subtract one
    order(:) = order(:) - 1

    ! Check if the truncation array has the right number of entries
    if (allocated(truncation_map)) then
      if (size(truncation_map) /= merge(number_coarse_groups, 1, allocated(energy_group_map))) then
        print *, "INPUT ERROR : Incorrect number of entries in truncation array"
        print *, "The truncation map must contain ", merge(number_coarse_groups, 1, allocated(energy_group_map)), " entries"
        stop
      end if
    end if

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
    allocate(phi_m(0:expansion_order, 0:number_legendre, number_coarse_groups, number_cells))
    allocate(psi_m(0:expansion_order, number_coarse_groups, 2 * number_angles, number_cells))

  end subroutine initialize_moments

  subroutine initialize_basis()
    ! ##########################################################################
    ! Load basis set from file
    ! ##########################################################################

    ! Use Statements
    use control, only : dgm_basis_name, number_fine_groups, energy_group_map

    ! Variable definitions
    double precision, dimension(number_fine_groups) :: &
        array1 ! Temporary array
    integer :: &
        g,   & ! Fine group index
        gp,  & ! Fine group prime index
        cg,  & ! Coarse group index
        i      ! Cell index

    ! allocate the basis array
    allocate(basis(number_fine_groups, 0:expansion_order))

    ! initialize the basis to zero
    basis = 0.0

    ! open the file and read into the basis container
    open(unit=5, file=dgm_basis_name)
    do g = 1, number_fine_groups
      cg = energy_group_map(g)
      array1(:) = 0.0
      read(5,*) array1
      i = 0
      do gp = 1, number_fine_groups
        if (energy_group_map(gp) == cg) then
          basis(g, i) = array1(gp)
          i = i + 1
        end if
        if (order(cg) < i) then
          exit
        end if
      end do
    end do

    ! clean up
    close(unit=5)

  end subroutine initialize_basis

  subroutine compute_expanded_cross_sections()
    ! ##########################################################################
    ! Initialize and fill the expanded cross section containers
    ! ##########################################################################

    ! Use Statements
    use control, only : number_coarse_groups, scatter_legendre_order, number_fine_groups, &
                        energy_group_map
    use material, only : number_materials, sig_t, nu_sig_f, sig_s

    ! Variable definitions
    integer :: &
        m,   & ! Material index
        g,   & ! Group index
        gp,  & ! Group prime index
        cg,  & ! Coarse group index
        cgp, & ! Coarse group prime index
        l,   & ! Legendre index
        i,   & ! order index
        j      ! order prime index

    allocate(expanded_sig_t(0:expansion_order, number_coarse_groups, number_materials, 0:expansion_order))
    allocate(expanded_nu_sig_f(0:expansion_order, number_coarse_groups, number_materials))
    allocate(expanded_sig_s(0:expansion_order, 0:scatter_legendre_order, number_coarse_groups, &
                            number_coarse_groups, number_materials, 0:expansion_order))

    expanded_sig_t = 0.0
    expanded_nu_sig_f = 0.0
    expanded_sig_s = 0.0

    ! Fill the expanded total cross section
    do i = 0, expansion_order
      do m = 1, number_materials
        do g = 1, number_fine_groups
          cg = energy_group_map(g)
          do j = 0, expansion_order
            expanded_sig_t(j, cg, m, i) = expanded_sig_t(j, cg, m, i) &
                                        + basis(g, i) * sig_t(g, m) * basis(g, j)
          end do
        end do
      end do
    end do

    ! Fill the expanded fission cross section
    do m = 1, number_materials
      do g = 1, number_fine_groups
        cg = energy_group_map(g)
        do j = 0, expansion_order
          expanded_nu_sig_f(j, cg, m) = expanded_nu_sig_f(j, cg, m) &
                                      + basis(g, 0) * nu_sig_f(g, m) * basis(g, j)
        end do
      end do
    end do

    ! Fill the expanded scatter cross section
    do i = 0, expansion_order
      do m = 1, number_materials
        do g = 1, number_fine_groups
          cg = energy_group_map(g)
          do gp = 1, number_fine_groups
            cgp = energy_group_map(g)
            do l = 0, scatter_legendre_order
              do j = 0, expansion_order
                expanded_sig_s(j, l, cgp, cg, m, i) = expanded_sig_s(j, l, cgp, cg, m, i) &
                                                    + basis(g, i) * basis(gp, 0) * sig_s(l, gp, g, m) * basis(gp, j)
              end do
            end do
          end do
        end do
      end do
    end do


  end subroutine compute_expanded_cross_sections

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
    if (allocated(basismap)) then
      deallocate(basismap)
    end if
    if (allocated(chi_m)) then
      deallocate(chi_m)
    end if
    if (allocated(phi_m)) then
      deallocate(phi_m)
    end if
    if (allocated(psi_m)) then
      deallocate(psi_m)
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
    if (allocated(expanded_sig_t)) then
      deallocate(expanded_sig_t)
    end if
    if (allocated(expanded_nu_sig_f)) then
      deallocate(expanded_nu_sig_f)
    end if
    if (allocated(expanded_sig_s)) then
      deallocate(expanded_sig_s)
    end if
  end subroutine finalize_moments

end module dgm
