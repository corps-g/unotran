module state
  ! ############################################################################
  ! Define the container classes for fluxes and cross sections
  ! ############################################################################

  use control, only : store_psi, source_value, file_name, initial_phi, &
                      initial_psi, use_dgm, solver_type, initial_keff, ignore_warnings
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
      d_sig_t,    &  ! Scalar total cross section moments
      d_incoming     ! Angular flux incident on the current cell
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
        a,       & ! Angle index
        g          ! Group index

    ! Allocate the scalar flux and source containers
    allocate(phi(0:number_legendre, number_cells, number_groups))
    ! Initialize phi
    ! Attempt to read file or use default if file does not exist
    open(unit = 10, status='old', file=initial_phi, form='unformatted', iostat=ios)
    if (ios > 0) then
      if (.not. ignore_warnings) then
        print *, "initial phi file, ", initial_phi, " is missing, using default value"
      end if
      if (solver_type == 'fixed') then
        phi = 1.0  ! default value
      else if (solver_type == 'eigen') then
        phi = 0.0
        do g = 1, number_groups
          do c = 1, number_cells
            phi(0, c, g) = nu_sig_f(g, mMap(c))
          end do
        end do
      end if
    else
      read(10) phi ! read the data in array x to the file
    end if
    close(10) ! close the file

    allocate(source(number_cells, 2 * number_angles, number_groups))
    ! Initialize source as isotropic
    source = 0.5 * source_value

    ! Only allocate psi if the option is to store psi    
    if (store_psi) then
      allocate(psi(number_cells, 2 * number_angles, number_groups))

      ! Initialize psi
      ! Attempt to read file or use default if file does not exist
      open(unit = 10, status='old', file=initial_psi, form='unformatted', iostat=ios)
      if (ios > 0) then
        if (.not. ignore_warnings) then
          print *, "initial psi file, ", initial_psi, " is missing, using default value"
        end if
        if (solver_type == 'fixed') then
          ! default to isotropic distribution
          psi = 0.5
        else
          ! default to isotropic distribution
          psi = 0.0
          do g = 1, number_groups
            do a = 1, 2 * number_angles
              do c = 1, number_cells
                psi(c, a, g) = phi(0, c, g) / 2
              end do
            end do
          end do
        end if
      else
        read(10) psi ! read the data in array x to the file
      end if
      close(10) ! close the file
    end if

    ! Initialize the angular flux incident on the boundary
    allocate(d_incoming(number_angles, number_groups))

    ! Initialize the eigenvalue to unity if fixed problem or default for eigen
    if (solver_type == 'fixed') then
      d_keff = 1.0
    else
      d_keff = initial_keff
    end if

    ! Set external source to zero if eigen problem
    if (solver_type == 'eigen') then
      if (norm2(source) > 0.0) then
        print *, "WARNING : Eigen solver is setting external source to zero"
      end if
      source = 0.0
    end if

    call normalize_flux(number_groups, phi, psi)

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
    if (allocated(d_psi)) then
      deallocate(d_psi)
    end if
    if (allocated(d_sig_t)) then
      deallocate(d_sig_t)
    end if
    if (allocated(d_incoming)) then
      deallocate(d_incoming)
    end if
  end subroutine finalize_state

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
    if (allocated(d_incoming)) then
      deallocate(d_incoming)
    end if

    ! Reallocate with the specified number of groups, nG
    allocate(d_source(number_cells, 2 * number_angles, nG))
    allocate(d_nu_sig_f(number_cells, nG))
    allocate(d_sig_t(number_cells, nG))
    allocate(d_delta(number_cells, 2 * number_angles, nG))
    allocate(d_phi(0:number_legendre, number_cells, nG))
    allocate(d_chi(number_cells, nG))
    allocate(d_sig_s(0:number_legendre, number_cells, nG, nG))
    if (store_psi) then
      allocate(d_psi(number_cells, 2 * number_angles, nG))
    end if
    allocate(d_incoming(number_angles, nG))
  end subroutine reallocate_states
  
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

  subroutine normalize_flux(nG, phi, psi)
    ! ##########################################################################
    ! Normalize the flux for the eigenvalue problem
    ! ##########################################################################

    integer, intent(in) :: &
        nG       ! Number of groups
    double precision, intent(inout), dimension(:,:,:) :: &
        phi, &   ! Scalar flux
        psi      ! Angular flux
    double precision :: &
        frac     ! Normalization fraction

    if (solver_type == 'eigen') then
      frac = sum(abs(phi(1,:,:))) / (number_cells * nG)

      ! normalize phi
      phi = phi / frac

      ! normalize psi
      psi = psi / frac
    end if

  end subroutine normalize_flux

end module state
