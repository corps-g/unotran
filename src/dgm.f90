module dgm
  use material, only: number_groups, number_legendre, number_materials, sig_s, sig_t, nu_sig_f, chi
  use mesh, only: number_cells, mMap
  use angle, only: number_angles
  use state, only: phi, psi, source

  implicit none

  double precision, allocatable :: sig_s_moment(:,:,:,:)
  double precision, allocatable :: phi_0_moment(:,:,:), source_moment(:,:,:), delta_moment(:,:,:)
  double precision, allocatable :: psi_0_moment(:,:,:), chi_moment(:,:)
  double precision, allocatable :: sig_t_moment(:,:), basis(:,:), nu_sig_f_moment(:,:)
  integer :: expansion_order, number_course_groups
  integer, allocatable :: energyMesh(:), order(:)

  contains

  ! Initialize the container for the cross section and flux moments
  subroutine initialize_moments(energyMap, truncation_order)
    integer, intent(in) :: energyMap(:)
    integer, intent(in), optional :: truncation_order(:)
    integer :: g, gp, cg

    phi = 1.0
    psi = 1.0
    ! Get the number of course groups
    number_course_groups = size(energyMap) + 1

    ! Create the map of course groups and default to full expansion order
    allocate(energyMesh(number_groups))
    allocate(order(number_course_groups))
    order = 0
    g = 1
    do gp = 1, number_groups
      energyMesh(gp) = g
      order(g) = order(g) + 1
      if (gp == energyMap(g)) then
        g = g + 1
      end if
    end do

    ! Check if the optional argument for the truncation is present
    if (present(truncation_order)) then
      ! Check if the truncation array has the right number of entries
      if (size(truncation_order) /= number_course_groups) then
        error stop "Incorrect number of entries in truncation array"
      end if
      ! Update the order array with the truncated value if sensible
      do cg = 1, number_course_groups
        if ((truncation_order(cg) < order(cg)) .and. (truncation_order(cg) >= 0)) then
          order(cg) = truncation_order(cg)
        end if
      end do
    end if

    expansion_order = MAXVAL(order)

    ! Allocate the moment containers
    allocate(sig_s_moment(0:number_legendre, number_course_groups, number_course_groups, number_cells))
    allocate(phi_0_moment(0:number_legendre, number_course_groups, number_cells))
    allocate(source_moment(number_course_groups, 2*number_angles, number_cells))
    allocate(psi_0_moment(number_course_groups, 2*number_angles, number_cells))
    allocate(delta_moment(number_course_groups, 2*number_angles, number_cells))
    allocate(sig_t_moment(number_course_groups, number_cells))
    allocate(nu_sig_f_moment(number_course_groups, number_cells))
    allocate(chi_moment(number_course_groups, number_cells))

  end subroutine initialize_moments

  ! Load basis set from file
  subroutine initialize_basis(fileName)
    character(len=*), intent(in) :: fileName
    double precision, allocatable, dimension(:) :: array1
    integer, allocatable :: cumsum(:)
    integer :: g, cg, i

    ! allocate the basis array
    allocate(basis(number_groups, 0:expansion_order))
    allocate(array1(number_groups))
    allocate(cumsum(number_course_groups))
    ! initialize the basis to zero
    basis = 0.0

    cumsum = order

    do cg = 1, number_course_groups
      if (cg == 1) then
        cycle
      end if
      cumsum(cg) = cumsum(cg - 1) + cumsum(cg)
    end do
    cumsum = cumsum - order

    ! open the file and read into the basis container
    open(unit=5, file=fileName)
    do g = 1, number_groups
      cg = energyMesh(g)
      array1(:) = 0.0
      read(5,*) array1
      do i = 1, order(cg)
        basis(g, i) = array1(cumsum(cg) + i)
      end do
    end do
    ! clean up
    close(unit=5)
    deallocate(array1, cumsum)

  end subroutine initialize_basis

  ! Deallocate the variable containers
  subroutine finalize_moments()
    if (allocated(sig_s_moment)) then
      deallocate(sig_s_moment)
    end if
    if (allocated(phi_0_moment)) then
      deallocate(phi_0_moment)
    end if
    if (allocated(source_moment)) then
      deallocate(source_moment)
    end if
    if (allocated(psi_0_moment)) then
      deallocate(psi_0_moment)
    end if
    if (allocated(delta_moment)) then
      deallocate(delta_moment)
    end if
    if (allocated(sig_t_moment)) then
      deallocate(sig_t_moment)
    end if
    if (allocated(nu_sig_f_moment)) then
      deallocate(nu_sig_f_moment)
    end if
    if (allocated(chi_moment)) then
      deallocate(chi_moment)
    end if
    if (allocated(basis)) then
      deallocate(basis)
    end if
    if (allocated(order)) then
      deallocate(order)
    end if
    if (allocated(energyMesh)) then
      deallocate(energyMesh)
    end if
  end subroutine finalize_moments

  ! Expand the flux moments using the basis functions
  subroutine compute_flux_moments()
    integer :: a, c, cg, cgp, g, gp, l, mat
    double precision :: num

    ! initialize all moments to zero
    phi_0_moment = 0.0
    psi_0_moment = 0.0

    ! Get moments for the fluxes
    do c = 1, number_cells
      ! get the material for the current cell
      mat = mMap(c)

      do a = 1, number_angles * 2
        do g = 1, number_groups
          cg = energyMesh(g)
          ! Scalar flux
          if (a == 1) then
            phi_0_moment(:, cg, c) = phi_0_moment(:, cg, c) + basis(g, 0) * phi(:, g, c)
          end if
          ! Angular flux
          psi_0_moment(cg, a, c) = psi_0_moment(cg, a, c) +  basis(g, 0) * psi(g, a, c)
        end do
      end do
    end do
  end subroutine compute_flux_moments

  ! Expand the cross section moments using the basis functions
  subroutine compute_xs_moments(order)
    integer :: a, c, cg, cgp, g, gp, l, i, mat
    integer, intent(in) :: order
    ! initialize all moments to zero
    sig_s_moment = 0.0
    source_moment = 0.0
    delta_moment = 0.0
    sig_t_moment = 0.0
    nu_sig_f_moment = 0.0
    chi_moment = 0.0

    do c = 1, number_cells
      ! get the material for the current cell
      mat = mMap(c)

      do a = 1, number_angles * 2
        do g = 1, number_groups
          cg = energyMesh(g)
          if (a == 1) then
            ! total cross section moment
            sig_t_moment(cg, c) = sig_t_moment(cg, c) + basis(g, 0) * sig_t(g, mat) * phi(0, g, c) / phi_0_moment(0, cg, c)
            ! fission cross section moment
            nu_sig_f_moment(cg, c) = nu_sig_f_moment(cg, c) + nu_sig_f(g, mat) * phi(0, g, c) / phi_0_moment(0, cg, c)
            ! chi moment
            chi_moment(cg, c) = chi_moment(cg, c) + basis(g, order) * chi(g, mat)
            ! Scattering cross section moment
            do gp = 1, number_groups
              cgp = energyMesh(gp)
              sig_s_moment(:, cgp, cg, c) = sig_s_moment(:, cgp, cg, c) + &
                                            basis(g, order) * sig_s(:, gp, g, mat) * phi(:, gp, c) / phi_0_moment(:, cg, c)
            end do
          end if
          ! angular total cross section moment (delta)
          delta_moment(cg, a, c) = delta_moment(cg, a, c) + &
                                   basis(g, order) * (sig_t(g, mat) - sig_t_moment(cg, c)) * psi(g, a, c) / psi_0_moment(cg, a, c)
          ! Source moment
          source_moment(cg, a, c) = source_moment(cg, a, c) + basis(g, order) * source(g, a, c)
        end do
      end do
    end do

  end subroutine compute_xs_moments

end module dgm
