module dgm
  use control, only : energy_group_map, truncation_map, dgm_basis_name, dgm_expansion_order
  use material, only: number_groups, number_legendre, number_materials, sig_s, sig_t, nu_sig_f, chi
  use mesh, only: number_cells, mMap
  use angle, only: number_angles
  use state, only: phi, psi, source, reallocate_states, d_source, d_nu_sig_f, d_sig_s, &
                   d_chi, d_phi, d_delta, d_sig_t, d_psi

  implicit none

  double precision, allocatable :: basis(:,:), chi_m(:,:,:), source_m(:,:,:,:)
  integer :: expansion_order, number_coarse_groups
  integer, allocatable :: energyMesh(:), order(:), basismap(:)

  contains

  ! Initialize the container for the cross section and flux moments
  subroutine initialize_moments()
    integer :: g, gp, cg

    ! Get the number of coarse groups
    if (allocated(energy_group_map)) then
      number_coarse_groups = size(energy_group_map) + 1
    else
      number_coarse_groups = 1
    end if

    ! Create the map of coarse groups and default to full expansion order
    allocate(energyMesh(number_groups))
    allocate(order(number_coarse_groups))
    allocate(basismap(number_coarse_groups))
    order = -1
    g = 1

    do gp = 1, number_groups
      energyMesh(gp) = g
      order(g) = order(g) + 1
      if (g < number_coarse_groups) then
        if (gp == energy_group_map(g)) then
          g = g + 1
        end if
      end if
    end do
    basismap(:) = order(:)

    ! Check if the optional argument for the truncation is present
    if (allocated(truncation_map)) then
      ! Check if the truncation array has the right number of entries
      if (size(truncation_map) /= number_coarse_groups) then
        error stop "Incorrect number of entries in truncation array"
      end if
      ! Update the order array with the truncated value if sensible
      do cg = 1, number_coarse_groups
        if ((truncation_map(cg) < order(cg)) .and. (truncation_map(cg) >= 0)) then
          order(cg) = truncation_map(cg)
        end if
      end do
    end if

    expansion_order = MAXVAL(order)

    ! reallocate container arrays to number_coarse_groups
    call reallocate_states(number_coarse_groups)

  end subroutine initialize_moments

  ! Load basis set from file
  subroutine initialize_basis()
    double precision, allocatable, dimension(:) :: array1
    integer, allocatable :: cumsum(:)
    integer :: g, cg, i

    ! allocate the basis array
    allocate(basis(number_groups, 0:expansion_order))
    allocate(array1(number_groups))
    allocate(cumsum(number_coarse_groups))
    ! initialize the basis to zero
    basis = 0.0

    cumsum = basismap + 1

    do cg = 1, number_coarse_groups
      if (cg == 1) then
        cycle
      end if
      cumsum(cg) = cumsum(cg - 1) + cumsum(cg)
    end do
    cumsum = cumsum - basismap

    ! open the file and read into the basis container
    open(unit=5, file=dgm_basis_name)
    do g = 1, number_groups
      cg = energyMesh(g)
      array1(:) = 0.0
      read(5,*) array1
      do i = 0, order(cg)
        basis(g, i) = array1(cumsum(cg) + i)
      end do
    end do
    ! clean up
    close(unit=5)

    deallocate(array1, cumsum)

  end subroutine initialize_basis

  ! Deallocate the variable containers
  subroutine finalize_moments()
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
    if (allocated(source_m)) then
      deallocate(source_m)
    end if
  end subroutine finalize_moments

  ! Expand the flux moments using the basis functions
  subroutine compute_flux_moments()
    integer :: a, c, cg, cgp, g, gp, l, mat
    double precision :: num

    ! initialize all moments to zero
    d_phi = 0.0
    d_psi = 0.0

    ! Get moments for the fluxes
    do c = 1, number_cells
      ! get the material for the current cell
      mat = mMap(c)

      do a = 1, number_angles * 2
        do g = 1, number_groups
          cg = energyMesh(g)
          ! Scalar flux
          if (a == 1) then
            d_phi(:, cg, c) = d_phi(:, cg, c) + basis(g, 0) * phi(:, g, c)
          end if
          ! Angular flux
          d_psi(cg, a, c) = d_psi(cg, a, c) +  basis(g, 0) * psi(g, a, c)
        end do
      end do
    end do

  end subroutine compute_flux_moments

  ! Expand the cross section moments using the basis functions
  subroutine compute_xs_moments(order)
    integer :: a, c, cg, cgp, g, gp, l, i, mat
    integer, intent(in) :: order
    double precision :: num
    ! initialize all moments to zero
    d_sig_s = 0.0
    d_source(:, :, :) = source_m(:, :, :, order)
    d_delta = 0.0
    d_sig_t = 0.0
    d_nu_sig_f = 0.0
    d_chi(:, :) = chi_m(:, :, order)

    do c = 1, number_cells
      ! get the material for the current cell
      mat = mMap(c)
      do g = 1, number_groups
        cg = energyMesh(g)
        ! total cross section moment
        d_sig_t(cg, c) = d_sig_t(cg, c) + basis(g, 0) * sig_t(g, mat) * phi(0, g, c) / d_phi(0, cg, c)
        ! fission cross section moment
        d_nu_sig_f(cg, c) = d_nu_sig_f(cg, c) + nu_sig_f(g, mat) * phi(0, g, c) / d_phi(0, cg, c)
        ! Scattering cross section moment
        do gp = 1, number_groups
          cgp = energyMesh(gp)
          d_sig_s(:, cgp, cg, c) = d_sig_s(:, cgp, cg, c) &
                                   + basis(g, order) * sig_s(:, gp, g, mat) * phi(:, gp, c) / d_phi(:, cgp, c)
        end do
      end do

      do a = 1, number_angles * 2
        do g = 1, number_groups
          cg = energyMesh(g)
          if (d_psi(cg, a, c) == 0.0) then
            num = 0.0
          else
            num = basis(g, order) * (sig_t(g, mat) - d_sig_t(cg, c)) * psi(g, a, c) / d_psi(cg, a, c)
          end if
          ! angular total cross section moment (delta)
          d_delta(cg, a, c) = d_delta(cg, a, c) + num
        end do
      end do
    end do

  end subroutine compute_xs_moments

  ! Expand the source and chi using the basis functions
  subroutine compute_source_moments()
    integer :: order, c, a, g, cg, mat
    allocate(chi_m(number_coarse_groups, number_cells, 0:expansion_order))
    allocate(source_m(number_coarse_groups, number_angles * 2, number_cells, 0:expansion_order))

    chi_m = 0.0
    source_m = 0.0

    do order = 0, expansion_order
      do c = 1, number_cells
        mat = mMap(c)
        do a = 1, number_angles * 2
          do g = 1, number_groups
            cg = energyMesh(g)
            if (a == 1) then
              ! chi moment
              chi_m(cg, c, order) = chi_m(cg, c, order) + basis(g, order) * chi(g, mat)
            end if

            ! Source moment
            source_m(cg, a, c, order) = source_m(cg, a, c, order) + basis(g, order) * source(g, a, c)
          end do
        end do
      end do
    end do

  end subroutine compute_source_moments

end module dgm
