module state
  use material, only: number_groups, number_legendre
  use mesh, only: number_cells
  use angle, only: number_angles
  implicit none
  double precision, allocatable, dimension(:,:,:) :: psi, source, phi
  logical :: store_psi
  character(len=2) :: equation
  
  contains
  
  ! Allocate the variable containers
  subroutine initialize_state(store, EQ)
    ! Inputs
    !   store : boolian for option to store angular flux
    !   EQ : Define the type of closure relation used

    logical, intent(in) :: store
    character(len=2), intent(in) :: EQ
    
    store_psi = store  ! Set the option to the given parameter
    equation = EQ  ! Set the option to the given parameter

    ! Only allocate psi if the option is to store psi    
    if (store_psi) then
      allocate(psi(number_groups,number_angles*2,number_cells))
      psi = 0.0
    end if
    
    ! Allocate the scalar flux and source containers
    allocate(phi(0:number_legendre,number_groups,number_cells))
    allocate(source(number_groups,number_angles*2,number_cells))
    
    ! Initialize containers to zero
    phi = 0.0
    source = 0.0
    
  end subroutine initialize_state
  
  ! Deallocate the variable containers
  subroutine finalize_state()
    if (allocated(phi)) then
      deallocate(phi)
    end if
    if (allocated(source)) then
      deallocate(source)
    end if
    if (allocated(psi)) then
      deallocate(psi)
    end if
  end subroutine finalize_state
  
end module state
