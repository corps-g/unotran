module dgm
  use material, only: number_groups, number_legendre, number_materials, sig_s, sig_t, nu_sig_f, chi
  use mesh, only: number_cells, mMap
  use angle, only: number_angles
  use state, only: phi, psi, source

  implicit none

  double precision, allocatable, dimension(:,:,:,:,:) :: sig_s_moment
  double precision, allocatable, dimension(:,:,:,:) :: phi_moment, source_moment, delta_moment
  double precision, allocatable, dimension(:,:,:) :: psi_0_moment, chi_moment
  double precision, allocatable, dimension(:,:) :: sig_t_moment, basis, nu_sig_f_moment
  integer :: expansion_order, number_course_groups
  integer, allocatable :: energyMesh(:), order(:)

  contains

  ! Initialize the container for the cross section and flux moments
  subroutine initialize_moments(energyMap)
    integer, intent(in) :: energyMap(:)
    integer :: g, gp

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
      if (gp .eq. energyMap(g)) then
        g = g + 1
      end if
    end do

    expansion_order = MAXVAL(order)

    ! Allocate the moment containers
    allocate(sig_s_moment(0:number_legendre, expansion_order, number_course_groups, number_course_groups, number_cells))
    allocate(phi_moment(0:number_legendre, expansion_order, number_course_groups, number_cells))
    allocate(source_moment(expansion_order, number_course_groups, 2*number_angles, number_cells))
    allocate(psi_0_moment(number_course_groups, 2*number_angles, number_cells))
    allocate(delta_moment(expansion_order, number_course_groups, 2*number_angles, number_cells))
    allocate(sig_t_moment(number_course_groups, number_cells))
    allocate(nu_sig_f_moment(number_course_groups, number_cells))
    allocate(chi_moment(expansion_order, number_course_groups, number_cells))

  end subroutine initialize_moments

  ! Load basis set from file
  subroutine initialize_basis(fileName)
    character(len=*), intent(in) :: fileName
    double precision, allocatable, dimension(:) :: array1
    integer, allocatable :: cumsum(:)
    integer :: g, cg, i

    ! allocate the basis array
    allocate(basis(expansion_order, number_groups))
    allocate(array1(number_groups))
    allocate(cumsum(number_course_groups))
    ! initialize the basis to zero
    basis = 0.0

    cumsum = order

    do cg = 1, number_course_groups
      if (cg == 1) then
        cycle
      end if
      cumsum(cg) = cumsum(cg-1) + cumsum(cg)
    end do
    cumsum = cumsum - order

    ! open the file and read into the basis container
    open(unit=5, file=fileName)
    do g = 1, number_groups
      cg = energyMesh(g)
      array1(:) = 0.0
      read(5,*) array1
      do i = 1, order(cg)
        basis(i,g) = array1(cumsum(cg) + i)
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
    if (allocated(phi_moment)) then
      deallocate(phi_moment)
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

  ! Expand the moments using the basis functions
  subroutine compute_moments()
    integer :: a, c, cg, cgp, g, gp, l, i, mat
    double precision :: num

    ! initialize all moments to zero
    sig_s_moment = 0.0
    phi_moment = 0.0
    source_moment = 0.0
    psi_0_moment = 0.0
    delta_moment = 0.0
    sig_t_moment = 0.0
    nu_sig_f_moment = 0.0
    chi_moment = 0.0

    ! Get moments for the fluxes
    do c = 1, number_cells
      ! get the material for the current cell
      mat = mMap(c)

      ! Scalar flux
      do g = 1, number_groups
        cg = energyMesh(g)
        do i = 1, order(cg)
          do l = 0, number_legendre
            phi_moment(l, i, cg, c) = phi_moment(l, i, cg, c) + basis(i, g) * phi(l, g, c)
          end do
        end do
      end do

      ! Angular flux
      do a = 1, number_angles * 2
        do g = 1, number_groups
          cg = energyMesh(g)
          psi_0_moment(cg, a, c) = psi_0_moment(cg, a, c) +  basis(1, g) * psi(g, a, c)
        end do
      end do

      ! cross section moments
      do g = 1, number_groups
        cg = energyMesh(g)
        ! total cross section moment
        if (phi_moment(0, 1, cg, c) == 0) then
          num = 0.0
        else
          num = basis(1, g) * sig_t(g, mat) * phi(0, g, c) / phi_moment(0, 1, cg, c)
        end if
        sig_t_moment(cg, c) = sig_t_moment(cg, c) + num
        ! fission cross section moment
        nu_sig_f_moment(cg, c) = nu_sig_f_moment(cg, c) + nu_sig_f(g, mat) * phi(0, g, c)
        ! chi moment
        do i = 1, order(cg)
          chi_moment(i, cg, c) = chi_moment(i, cg, c) + basis(i,g) * chi(g, mat)
        end do
      end do

      do a = 1, number_angles * 2
        do g = 1, number_groups
          cg = energyMesh(g)
          do i = 1, order(cg)
            ! angular total cross section moment (delta)
            num = basis(i, g) * (sig_t(g, mat) - sig_t_moment(cg, c)) * psi(g, a, c)
            delta_moment(i, cg, a, c) = delta_moment(i, cg, a, c) + num
            ! Source moment
            source_moment(i, cg, a, c) = source_moment(i, cg, a, c) + basis(i, g) * source(g, a, c)
          end do
        end do
      end do

      do g = 1, number_groups
        cg = energyMesh(g)
        do gp = 1, number_groups
          cgp = energyMesh(gp)
          do i = 1, order(cg)
            do l = 0, number_legendre
              ! Scattering cross section moment
              num = basis(i,g) * sig_s(l, gp, g, mat) * phi(l, gp, c)
              sig_s_moment(l, i, cgp, cg, c) = sig_s_moment(l, i, cgp, cg, c) + num
            end do
          end do
        end do
      end do
    end do

  end subroutine compute_moments

end module dgm
