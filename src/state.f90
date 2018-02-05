module state
  ! ############################################################################
  ! Define the container classes for fluxes and cross sections
  ! ############################################################################

  use control, only : store_psi, source_value, file_name, initial_phi, initial_psi, use_dgm, solver_type
  use material, only : number_groups, number_legendre, nu_sig_f
  use mesh, only : number_cells, mMap
  use angle, only : number_angles

  implicit none

  double precision, allocatable, dimension(:,:,:,:) :: &
      d_sig_s        ! Scattering cross section moments

  double precision, allocatable, dimension(:,:,:) :: &
      psi,        &  ! Angular flux
      source,     &  ! External source
      phi,        &  ! Scalar Flux
      d_source,   &  ! Extermal source moments
      d_delta,    &  ! Angular total cross section moments
      d_phi,      &  ! Scalar flux moments
      d_psi          ! Angular flux moments
                                                                                                    
  double precision, allocatable, dimension(:,:) :: &
      d_nu_sig_f, &  ! Fission cross section moments (times nu)
      d_chi,      &  ! Chi spectrum moments
      d_sig_t        ! Scalar total cross section moments
                                                     
  double precision, allocatable, dimension(:) :: &
      density        ! Fission density

  double precision :: &
      d_keff         ! k-eigenvalue
  
  contains
  
  ! Allocate the variable containers
  subroutine initialize_state()
    ! ##########################################################################
    ! Initialize each of the container variables to default/entered values
    ! ##########################################################################

    integer :: &
        ios = 0, & ! I/O error status
        c,       & ! Cell index
        g          ! Group index

    ! Allocate the scalar flux and source containers
    allocate(phi(0:number_legendre, number_groups, number_cells))
    ! Initialize phi
    ! Attempt to read file or use default if file does not exist
    open(unit = 10, status='old', file=initial_phi, form='unformatted', iostat=ios)
    if (ios > 0) then
      !print *, "initial phi file, ", initial_phi, " is missing, using default value"
      if (solver_type == 'fixed') then
        phi = 1.0  ! default value
      else if (solver_type == 'eigen') then
        phi = 0.0
        do c = 1, number_cells
          do g = 1, number_groups
            phi(0, g, c) = nu_sig_f(g, mMap(c))
          end do
        end do
      end if

    else
      read(10) phi ! read the data in array x to the file
    end if
    close(10) ! close the file

    allocate(source(number_groups, number_angles * 2, number_cells))
    ! Initialize source
    source = source_value

    ! Allocate space for the fission density
    allocate(density(number_cells))
    call update_density()

    ! Only allocate psi if the option is to store psi    
    if (store_psi) then
      allocate(psi(number_groups, number_angles * 2, number_cells))

      ! Initialize psi
      ! Attempt to read file or use default if file does not exist
      open(unit = 10, status='old', file=initial_psi, form='unformatted', iostat=ios)
      if (ios > 0) then
        !print *, "initial psi file, ", initial_psi, " is missing, using default value"
        psi = 1.0  ! default value
      else
        read(10) psi ! read the data in array x to the file
      end if
      close(10) ! close the file
    end if

    ! Initialize the eigenvalue to unity
    d_keff = 1.0

    ! Set external source to zero if eigen problem
    if (solver_type == 'eigen') then
      if (norm2(source) > 0.0) then
        print *, "WARNING : Eigen solver is setting external source to zero"
      end if
      source = 0.0
    end if

    call reallocate_states(number_groups)

  end subroutine initialize_state
  
  ! Deallocate the variable containers
  subroutine finalize_state()
    ! ##########################################################################
    ! Deallocate all allocated arrays
    ! ##########################################################################

    if (allocated(phi)) then
      deallocate(phi)
    end if
    if (allocated(source)) then
      deallocate(source)
    end if
    if (allocated(psi)) then
      deallocate(psi)
    end if
    if (allocated(d_source)) then
      deallocate(d_source)
    end if
    if (allocated(d_nu_sig_f)) then
      deallocate(d_nu_sig_f)
    end if
    if (allocated(d_delta)) then
      deallocate(d_delta)
    end if
    if (allocated(d_phi)) then
      deallocate(d_phi)
    end if
    if (allocated(d_chi)) then
      deallocate(d_chi)
    end if
    if (allocated(d_sig_s)) then
      deallocate(d_sig_s)
    end if
    if (allocated(density)) then
      deallocate(density)
    end if
  end subroutine finalize_state

  ! Resize the container arrays to number of energy groups, nG
  subroutine reallocate_states(nG)
    ! ##########################################################################
    ! Resize the moment containers to the coarse group structure when using DGM
    ! ##########################################################################

    integer, intent(in) :: &
        nG  ! Number of energy groups

    ! Deallocate arrays if needed
    if (allocated(d_source)) then
      deallocate(d_source)
    end if
    if (allocated(d_nu_sig_f)) then
      deallocate(d_nu_sig_f)
    end if
    if (allocated(d_delta)) then
      deallocate(d_delta)
    end if
    if (allocated(d_phi)) then
      deallocate(d_phi)
    end if
    if (allocated(d_psi)) then
      deallocate(d_psi)
    end if
    if (allocated(d_chi)) then
      deallocate(d_chi)
    end if
    if (allocated(d_sig_s)) then
      deallocate(d_sig_s)
    end if
    if (allocated(d_sig_t)) then
      deallocate(d_sig_t)
    end if

    ! Reallocate with the specified number of groups, nG
    allocate(d_source(nG, number_angles * 2, number_cells))
    allocate(d_nu_sig_f(nG, number_cells))
    allocate(d_sig_t(nG, number_cells))
    allocate(d_delta(nG, number_angles * 2, number_cells))
    allocate(d_phi(0:number_legendre, nG, number_cells))
    allocate(d_chi(nG, number_cells))
    allocate(d_sig_s(0:number_legendre, nG, nG, number_cells))
    if (store_psi) then
      allocate(d_psi(nG, number_angles * 2, number_cells))
    end if
  end subroutine
  
  subroutine update_density()
    ! ##########################################################################
    ! Compute the fission density from the scalar flux and fission cross section
    ! ##########################################################################

    use material, only : nu_sig_f
    use mesh, only : mMap

    integer :: &
        c,  & ! Cell index
        g     ! Group index

    density = 0
    do c = 1, number_cells
      do g = 1, number_groups
        density(c) = density(c) + phi(0, g, c) * nu_sig_f(g, mMap(c))
      end do
    end do

  end subroutine update_density

  subroutine output_state()
    ! ##########################################################################
    ! Save the scalar and angular flux objects to unformatted fortran file
    ! ##########################################################################

    character(:), allocatable :: &
        fname  ! Base file name

    fname = trim(file_name) // "_phi.bin"

    ! create a new file, or overwrite an existing one
    open(unit = 10, status='replace', file=fname, form='unformatted')
    write(10) phi ! write the data in array x to the file
    close(10) ! close the file

    if (store_psi) then
      fname = trim(file_name) // "_psi.bin"

      ! create a new file, or overwrite an existing one
      open(unit = 10, status='replace', file=fname, form='unformatted')
      write(10) psi ! write the data in array x to the file
      close(10) ! close the file
    end if

  end subroutine output_state

end module state
