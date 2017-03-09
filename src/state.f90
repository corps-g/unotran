module state
  use material, only: number_groups, number_legendre
  use mesh, only: number_cells
  use angle, only: number_angles
  implicit none
  double precision, allocatable, dimension(:,:,:) :: psi, source, phi
  logical :: store_psi
  character(len=2) :: equation
  
  save
  
  contains
  
  subroutine initialize_state(store, EQ)
    logical, optional :: store
    character(len=2), optional :: EQ
    
    ! Check if the optional argument store is given
    if (present(store)) then
      store_psi = store  ! Set the option to the given parameter
    else
      store_psi = .false.  ! Default to not storing the angular flux
    end if
    
    ! Check if the optional argument EQ is given
    if (present(EQ)) then
      equation = EQ  ! Set the option to the given parameter
    else
      equation = 'DD'  ! Default to not storing the angular flux
    end if

    ! Only allocate psi if the option is to store psi    
    if (store_psi) then
      allocate(psi(number_groups,number_angles*2,number_cells))
      psi = 0.0
    end if
    
    allocate(phi(0:number_legendre,number_groups,number_cells))
    allocate(source(number_groups,number_angles*2,number_cells))
    
    phi = 0.0
    source = 0.0
    
  end subroutine initialize_state
  
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
