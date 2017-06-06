module state
  use control, only : store_psi, source_value, file_name
  use material, only : number_groups, number_legendre
  use mesh, only : number_cells
  use angle, only : number_angles

  implicit none

  double precision, allocatable :: psi(:,:,:), source(:,:,:), phi(:,:,:)
  
  contains
  
  ! Allocate the variable containers
  subroutine initialize_state()
    ! Allocate the scalar flux and source containers
    allocate(phi(0:number_legendre,number_groups,number_cells))
    allocate(source(number_groups,number_angles*2,number_cells))

    ! Only allocate psi if the option is to store psi    
    if (store_psi) then
      allocate(psi(number_groups,number_angles*2,number_cells))
      psi = 0.0
    end if

    ! Initialize containers to zero
    phi = 0.0

    ! Initialize source
    source = source_value
    
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
  
  subroutine output_state()
    character(:), allocatable :: fname

    fname = trim(file_name) // "_phi.bin"

    open(unit = 10, status='replace',file=fname,form='unformatted')  ! create a new file, or overwrite an existing on
    write(10) phi ! write the data in array x to the file
    close(10) ! close the file

    if (store_psi) then
      fname = trim(file_name) // "_psi.bin"

      open(unit = 10, status='replace',file=fname,form='unformatted')  ! create a new file, or overwrite an existing on
      write(10) psi ! write the data in array x to the file
      close(10) ! close the file
    end if

  end subroutine output_state

end module state
