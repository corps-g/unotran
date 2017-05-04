module dgm
  use material, only: number_groups, number_legendre, number_materials, sig_s, sig_t
  use mesh, only: number_cells, mMap
  use angle, only: number_angles
  use state, only: phi, psi, source

  implicit none

  double precision, allocatable, dimension(:,:,:,:,:) :: sig_s_moment
  double precision, allocatable, dimension(:,:,:,:) :: phi_moment, source_moment, delta_moment
  double precision, allocatable, dimension(:,:,:) :: psi_0_moment
  double precision, allocatable, dimension(:,:) :: sig_t_moment, basis
  integer :: expansion_order, number_course_groups
  integer, allocatable :: energyMesh(:)

  contains

  ! Initialize the container for the cross section and flux moments
  subroutine initialize_moments(energyMap)
    integer, intent(in) :: energyMap(:)
    integer, dimension(:), allocatable :: order
    integer :: g, gp

    ! Get the number of course groups
    number_course_groups = size(energyMap) + 1

    ! Create the map of course groups and default to full expansion order
    allocate(energyMesh(number_groups))
    allocate(order(number_course_groups))
    order(:) = 0
    g = 1
    do gp = 1, number_groups
      energyMesh(gp) = g
      order(g) = order(g) + 1
      if (gp .eq. energyMap(g)) then
        g = g + 1
      end if
    end do

    expansion_order = MAXVAL(order)
    deallocate(order)

    ! Allocate the moment containers
    allocate(sig_s_moment(number_course_groups, number_course_groups, expansion_order, 0:number_legendre, number_cells))
    allocate(phi_moment(number_course_groups, expansion_order, 0:number_legendre, number_cells))
    allocate(source_moment(number_course_groups, expansion_order, number_angles, number_cells))
    allocate(psi_0_moment(number_course_groups, 2*number_angles, number_cells))
    allocate(delta_moment(number_course_groups, expansion_order, 2*number_angles, number_cells))
    allocate(sig_t_moment(number_course_groups, number_cells))
  end subroutine initialize_moments

  ! Load basis set from file
  subroutine initialize_basis(fileName)
    character(len=*), intent(in) :: fileName
    double precision, allocatable, dimension(:) :: array1
    integer :: g

    ! allocate the basis array
    allocate(basis(number_groups, expansion_order))
    allocate(array1(expansion_order))
    ! initialize the basis to zero
    basis(:,:) = 0.0

    ! open the file and read into the basis container
    open(unit=5, file=fileName)
    do g = 1, number_groups
      array1(:) = 0.0
      read(5,*) array1
      basis(g,:) = array1
    end do

    ! clean up
    deallocate(array1)

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
    if (allocated(basis)) then
      deallocate(basis)
    end if
    if (allocated(energyMesh)) then
      deallocate(energyMesh)
    end if
  end subroutine finalize_moments

  ! Expand the moments using the basis functions
  subroutine compute_moments()
    integer :: a, c, cg, cgp, g, gp, l, i, mat
    double precision :: num

    ! Get moments for the fluxes
    do c = 1, number_cells
      do i = 1, expansion_order
        do g = 1, number_groups
          cg = energyMesh(g)
          do l = 0, number_legendre
            ! Scalar flux
            phi_moment(l, cg, i, c) = phi_moment(l, cg, i, c) + basis(g, i) * phi(l, g, c)
          end do
          ! i represents the angle in this case
          ! Angular flux
          psi_0_moment(cg, i, c) = psi_0_moment(cg, i, c) +  basis(g, 0) * psi(g, i, c)
          psi_0_moment(cg, i+number_angles, c) = psi_0_moment(cg, i+number_angles, c) +  basis(g, 0) * psi(g, i+number_angles, c)
        end do
      end do
    end do

    ! Get moments for the cross sections
    do c = 1, number_cells
      mat = mMap(c)
      ! total cross section moment
      do g = 1, number_groups
        cg = energyMesh(g)
        sig_t_moment(cg, c) = sig_t_moment(cg, c) + basis(g,0) * sig_t(g, c) * phi(c, g, 0) / phi_moment(0, cg, 0, c)
      end do
      do a = 1, number_angles*2
        do i = 1, expansion_order
          do g = 1, number_groups
            cg = energyMesh(g)
            ! angular total cross section moment (delta)
            num = basis(g, i) * (sig_t(g, c) - sig_t_moment(cg, c)) * psi(g, a, c) / psi_0_moment(cg, a, c)
            delta_moment(cg, i, a, c) = delta_moment(cg, i, a, c) + num
            ! Source moment
            source_moment(cg, i, a, c) = source_moment(cg, i, a, c) + basis(g, i) * source(g, a, c)
          end do
        end do
      end do
      do i = 1, expansion_order
        do g = 1, number_groups
          cg = energyMesh(g)
          do gp = 1, number_groups
            cgp = energyMesh(gp)
            do l = 0, number_legendre
              ! Scattering cross section moment
              num = basis(g,i) * sig_s(l, gp, g, mat) * phi(l, gp, c) / phi_moment(l, cg, 0, c)
              sig_s_moment(l, cgp, cg, i, c) = sig_s_moment(l, cgp, cg, i, c) + num
            end do
          end do
        end do
      end do
    end do
  end subroutine compute_moments

end module dgm
