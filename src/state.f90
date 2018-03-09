module state
  ! ############################################################################
  ! Define the container classes for fluxes and cross sections
  ! ############################################################################

  implicit none

  double precision, allocatable, dimension(:,:,:,:) :: &
      d_sig_s        ! Scattering cross section moments
  double precision, allocatable, dimension(:,:,:) :: &
      psi,         & ! Angular flux
      source,      & ! External source
      phi,         & ! Scalar Flux
      d_source,    & ! Extermal source moments
      d_phi,       & ! Scalar flux moments
      d_psi          ! Angular flux moments
  double precision, allocatable, dimension(:,:) :: &
      d_nu_sig_f,  & ! Fission cross section moments (times nu)
      d_chi,       & ! Chi spectrum moments
      d_sig_t,     & ! Scalar total cross section moments
      d_incoming     ! Angular flux incident on the current cell
  double precision :: &
      d_keff,      & ! k-eigenvalue
      norm_frac      ! Fraction of normalization for eigenvalue problems
  
  contains
  
  subroutine initialize_state()
    ! ##########################################################################
    ! Initialize each of the container variables to default/entered values
    ! ##########################################################################

    ! Use Statements
    use control, only : number_fine_groups, number_coarse_groups, number_groups, &
                        use_DGM, ignore_warnings, initial_keff, initial_phi, &
                        initial_psi, number_angles, number_cells, number_legendre, &
                        solver_type, source_value, store_psi
    use mesh, only : mMap, create_mesh
    use material, only : nu_sig_f, create_material
    use angle, only : initialize_angle, initialize_polynomials
    use dgm, only : initialize_moments, initialize_basis

    ! Variable definitions
    integer :: &
        ios = 0, & ! I/O error status
        c,       & ! Cell index
        a,       & ! Angle index
        g          ! Group index

    ! Initialize the sub-modules
    ! initialize the mesh
    call create_mesh()
    ! read the material cross sections
    call create_material()
    ! initialize the angle quadrature
    call initialize_angle()
    ! get the basis vectors
    call initialize_polynomials()

    number_groups = number_fine_groups
    if (use_DGM) then
      ! Initialize DGM moments
      call initialize_moments()
      call initialize_basis()
      number_groups = number_coarse_groups
    end if

    ! Allocate the scalar flux and source containers
    allocate(phi(0:number_legendre, number_cells, number_fine_groups))
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

    allocate(source(number_cells, 2 * number_angles, number_fine_groups))
    ! Initialize source as isotropic
    source = 0.5 * source_value

    ! Only allocate psi if the option is to store psi    
    if (store_psi) then
      allocate(psi(number_cells, 2 * number_angles, number_fine_groups))

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
          do g = 1, number_fine_groups
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

    call normalize_flux(phi, psi)

    ! Allocate the moment containers
    allocate(d_source(number_cells, 2 * number_angles, number_groups))
    allocate(d_nu_sig_f(number_cells, number_groups))
    allocate(d_sig_t(number_cells, number_groups))
    allocate(d_phi(0:number_legendre, number_cells, number_groups))
    allocate(d_chi(number_cells, number_groups))
    allocate(d_sig_s(0:number_legendre, number_cells, number_groups, number_groups))
    if (store_psi) then
      allocate(d_psi(number_cells, 2 * number_angles, number_groups))
    end if

  end subroutine initialize_state
  
  subroutine finalize_state()
    ! ##########################################################################
    ! Deallocate all allocated arrays
    ! ##########################################################################

    ! Use Statements
    use control, only : use_DGM
    use angle, only : finalize_angle
    use mesh, only : finalize_mesh
    use material, only : finalize_material
    use dgm, only : finalize_moments

    call finalize_angle()
    call finalize_mesh()
    call finalize_material()
    if (use_DGM) then
      call finalize_moments()
    end if

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

  subroutine output_state()
    ! ##########################################################################
    ! Save the scalar and angular flux objects to unformatted fortran file
    ! ##########################################################################

    ! Use Statements
    use control, only : file_name, store_psi

    ! Variable definitions
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

  subroutine normalize_flux(phi, psi)
    ! ##########################################################################
    ! Normalize the flux for the eigenvalue problem
    ! ##########################################################################

    ! Use Statements
    use control, only : number_cells, solver_type, store_psi

    ! Variable definitions
    double precision, intent(inout), dimension(:,:,:) :: &
        phi,         & ! Scalar flux
        psi            ! Angular flux
    integer :: &
        number_groups  ! Number of energy groups

    if (solver_type == 'eigen') then
      number_groups = size(phi(1,1,:))

      norm_frac = sum(abs(phi(1,:,:))) / (number_cells * number_groups)

      ! normalize phi
      phi = phi / norm_frac
      if (store_psi) then
        ! normalize psi
        psi = psi / norm_frac
      end if
    end if

  end subroutine normalize_flux

end module state
