module dgm
  use material, only: number_groups, number_legendre, number_materials
  use mesh, only: number_cells
  use angle, only: number_angles

  implicit none

  double precision, allocatable, dimension(:,:,:,:,:) :: sig_s_moment
  double precision, allocatable, dimension(:,:,:,:) :: phi_moment, source_moment
  double precision, allocatable, dimension(:,:,:) :: psi_0_moment, delta_moment
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
    allocate(sig_s_moment(number_course_groups, number_course_groups, expansion_order, 0:number_legendre, number_materials))
    allocate(phi_moment(number_course_groups, expansion_order, 0:number_legendre, number_materials))
    allocate(source_moment(number_course_groups, expansion_order, number_angles, number_materials))
    allocate(psi_0_moment(number_course_groups, number_angles, number_materials))
    allocate(delta_moment(number_course_groups, expansion_order, number_materials))
    allocate(sig_t_moment(number_course_groups, number_materials))
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

!  do c = 1, number_cells
!    do i = 0, number_basis
!      do g = 1, number_groups
!        do l = 0, number_legendre
!          phi_moment(l,group(g),i,c) = phi_moment(l,group(g),i,c) + basis(g,i) * phi(l,g,c)
!        end do
!        do a = 1, number_angles
!          psi_moment(a,group(g),i,c) = psi_moment(a,group(g),i,c) + basis(g,i) * psi(g,a,c)
!        end do
!      end do
!    end do
!  end do

  end subroutine compute_moments

  ! Get the basis functions from a file
  subroutine read_basis()

  end subroutine read_basis

end module dgm
