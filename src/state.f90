module state
  ! ############################################################################
  ! Define the container classes for fluxes and cross sections
  ! ############################################################################

  implicit none

  double precision, allocatable, dimension(:,:,:,:) :: &
      mg_sig_s               ! Scattering cross section moments
  double precision, allocatable, dimension(:,:,:) :: &
      psi,                 & ! Angular flux
      phi,                 & ! Scalar Flux
      mg_phi,              & ! Scalar flux mg container
      mg_psi,              & ! Angular flux mg container
      mg_incident_x,       & ! Angular flux incident on the current cell in x direction
      mg_incident_y,       & ! Angular flux incident on the current cell in y direction
      sigphi                 ! Scalar flux container
  double precision, allocatable, dimension(:,:) :: &
      mg_nu_sig_f,         & ! Fission cross section mg container (times nu)
      mg_chi,              & ! Chi spectrum mg container
      mg_source,           & ! External source mg container
      mg_sig_t               ! Scalar total cross section mg container
  double precision, allocatable, dimension(:) :: &
      mg_density             ! Fission density
  integer, allocatable, dimension(:) :: &
      mg_mMap                ! material map mg container
  double precision :: &
      mg_constant_source,  & ! Constant multigroup source
      keff,                & ! k-eigenvalue
      norm_frac,           & ! Fraction of normalization for eigenvalue problems
      sweep_count,         & ! Counter for the number of transport sweeps
      scaling,             & ! Scaling factor for source terms
      recon_convergence_rate ! Approximate rate of convergence for recon iters
  
  contains
  
  subroutine initialize_state()
    ! ##########################################################################
    ! Initialize each of the container variables to default/entered values
    ! ##########################################################################

    ! Use Statements
    use control, only : number_fine_groups, number_coarse_groups, number_groups, &
                        use_DGM, ignore_warnings, initial_keff, initial_phi, &
                        initial_psi, number_angles, number_cells, number_legendre, &
                        solver_type, source_value, store_psi, check_inputs, &
                        verify_control, homogenization_map, number_regions, &
                        scatter_leg_order, delta_leg_order, truncate_delta, &
                        number_cells_x, number_cells_y, spatial_dimension, scale_1D, &
                        scale_2D
    use mesh, only : mMap, create_mesh
    use material, only : nu_sig_f, create_material, number_materials
    use angle, only : initialize_angle, initialize_polynomials
    use dgm, only : initialize_moments, initialize_basis, compute_expanded_cross_sections

    ! Variable definitions
    integer :: &
        ios = 0, & ! I/O error status
        c,       & ! Cell index
        a          ! Angle index

    ! Initialize the sub-modules
    ! read the material cross sections
    call create_material()
    ! Determine the Legendre order of phi to store
    if (delta_leg_order == -1) then
      delta_leg_order = scatter_leg_order
    end if
    number_legendre = max(delta_leg_order, scatter_leg_order)
    ! Verify the inputs
    if (verify_control) then
      call check_inputs()
    end if
    ! initialize the mesh
    call create_mesh()
    ! initialize the angle quadrature
    call initialize_angle()
    ! get the basis vectors
    call initialize_polynomials()
    ! Initialize the constant source
    mg_constant_source = 0.5 * source_value
    ! Determine the correct size for the multigroup containers
    number_groups = number_fine_groups
    number_regions = number_materials
    if (use_DGM) then
      if (.not. truncate_delta) then
        store_psi = .true.
      end if
      ! Initialize DGM moments
      call initialize_moments()
      call initialize_basis()
      call compute_expanded_cross_sections()
      number_groups = number_coarse_groups
      if (allocated(homogenization_map)) then
        number_regions = maxval(homogenization_map)
      else
        allocate(homogenization_map(number_cells))
        number_regions = number_cells
        do c = 1, number_cells
          homogenization_map(c) = c
        end do
      end if
    end if

    if (spatial_dimension == 1) then
      scaling = 1 / scale_1D
    else if (spatial_dimension == 2) then
      scaling = 1 / scale_2D
    end if

    ! Set the sweep counter to zero
    sweep_count = 0

    ! Allocate the scalar flux and source containers
    allocate(phi(0:number_legendre, number_fine_groups, number_cells))
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
        do c = 1, number_cells
          phi(0, :, c) = nu_sig_f(:, mMap(c))
        end do
      end if
    else
      read(10) phi ! read the data in array x to the file
    end if
    close(10) ! close the file

    ! Only allocate psi if the option is to store psi
    if (store_psi) then
      allocate(psi(number_fine_groups, 2 * spatial_dimension * number_angles, number_cells))

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
          do c = 1, number_cells
            do a = 1, 2 * spatial_dimension * number_angles
              psi(:, a, c) = phi(0, :, c) * scaling
            end do
          end do
        end if
      else
        read(10) psi ! read the data in array x to the file
      end if
      close(10) ! close the file
    end if

    ! Initialize the angular flux incident on the boundary
    allocate(mg_incident_x(number_groups, number_angles, number_cells_y))
    allocate(mg_incident_y(number_groups, number_angles, number_cells_x))
    ! Assume vacuum conditions for incident flux
    mg_incident_x(:, :, :) = 0.0
    mg_incident_y(:, :, :) = 0.0


    ! Initialize the eigenvalue to unity if fixed problem or default for eigen
    if (solver_type == 'fixed') then
      keff = 1.0
    else
      keff = initial_keff
    end if

    ! Initialize the material map container
    allocate(mg_mMap(number_cells))

    ! Initialize the fission density
    allocate(mg_density(number_cells))

    ! Normalize the scalar and angular fluxes
    call normalize_flux(phi, psi)

    ! Size the mg containers to be fine/coarse group for non/DGM problems
    allocate(mg_source(number_groups, number_cells))
    allocate(mg_nu_sig_f(number_groups, number_regions))
    allocate(mg_sig_t(number_groups, number_regions))
    allocate(mg_phi(0:number_legendre, number_groups, number_cells))
    allocate(mg_chi(number_groups, number_regions))
    allocate(mg_sig_s(0:scatter_leg_order, number_groups, number_groups, number_regions))
    allocate(sigphi(0:scatter_leg_order, number_groups, number_cells))
    if (store_psi) then
      allocate(mg_psi(number_groups, 2 * spatial_dimension * number_angles, number_cells))
    end if

  end subroutine initialize_state
  
  subroutine finalize_state()
    ! ##########################################################################
    ! Deallocate all allocated arrays
    ! ##########################################################################

    ! Use Statements
    use angle, only : finalize_angle
    use mesh, only : finalize_mesh
    use material, only : finalize_material
    use dgm, only : finalize_moments

    ! Deallocate the variables in the submodules
    call finalize_angle()
    call finalize_mesh()
    call finalize_material()
    call finalize_moments()

    ! Deallocate the state variables
    if (allocated(phi)) then
      deallocate(phi)
    end if
    if (allocated(psi)) then
      deallocate(psi)
    end if
    if (allocated(mg_source)) then
      deallocate(mg_source)
    end if
    if (allocated(mg_nu_sig_f)) then
      deallocate(mg_nu_sig_f)
    end if
    if (allocated(mg_phi)) then
      deallocate(mg_phi)
    end if
    if (allocated(mg_chi)) then
      deallocate(mg_chi)
    end if
    if (allocated(mg_sig_s)) then
      deallocate(mg_sig_s)
    end if
    if (allocated(mg_psi)) then
      deallocate(mg_psi)
    end if
    if (allocated(mg_sig_t)) then
      deallocate(mg_sig_t)
    end if
    if (allocated(mg_incident_x)) then
      deallocate(mg_incident_x)
    end if
    if (allocated(mg_incident_y)) then
      deallocate(mg_incident_y)
    end if
    if (allocated(mg_density)) then
      deallocate(mg_density)
    end if
    if (allocated(mg_mMap)) then
      deallocate(mg_mMap)
    end if
    if (allocated(sigphi)) then
      deallocate(sigphi)
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

    deallocate(fname)

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
      ! This Legendre order is 1 indexed instead of zero indexed
      number_groups = size(phi(1,:,1))

      norm_frac = sum(abs(phi(1,:,:))) / (number_groups * number_cells)

      ! normalize phi
      phi = phi / norm_frac
      if (store_psi) then
        ! normalize psi
        psi = psi / norm_frac
      end if
    end if

  end subroutine normalize_flux

  subroutine update_fission_density(fine_flag)
    ! ##########################################################################
    ! Compute the fission density for each cell
    ! ##########################################################################

    ! Use Statements
    use control, only : number_cells, use_DGM
    use mesh, only : mMap
    use material, only : nu_sig_f
    use dgm, only : dgm_order, phi_m

    ! Variable definitions
    integer :: &
      c      ! Cell index
    logical, intent(in), optional :: &
      fine_flag                      ! Controls computing density using fine flux
    logical :: &
      fine_flag_default = .false., & ! Controls computing density using fine flux
      fine_flag_val                  ! Holds actual value of fine_flag

    if (present(fine_flag)) then
      fine_flag_val = fine_flag
    else
      fine_flag_val = fine_flag_default
    end if

    ! Reset the fission density
    mg_density = 0.0

    ! Sum the fission reaction rate over groups for each cell
    if (fine_flag_val) then
      ! Compute fission density using fine flux
      do c = 1, number_cells
        mg_density(c) = sum(nu_sig_f(:, mMap(c)) * phi(0,:,c))
      end do
    else if (use_DGM .and. dgm_order > 0) then
      ! Compute the fission density using the mg_flux for higher moments
      do c = 1, number_cells
        mg_density(c) = sum(mg_nu_sig_f(:, mg_mMap(c)) * phi_m(0, 0,:,c))
      end do
    else
      ! Compute the fission density using the mg_flux normally
      do c = 1, number_cells
        mg_density(c) = sum(mg_nu_sig_f(:, mg_mMap(c)) * mg_phi(0,:,c))
      end do
    end if

  end subroutine update_fission_density

  subroutine output_moments()
    ! ##########################################################################
    ! Print the output of the cross sections to the standard output
    ! ##########################################################################

    use control, only : store_psi

    ! Total XS
    print *, 'Sig_t = ', mg_sig_t

    ! Fission XS
    print *, 'vSig_f = ', mg_nu_sig_f

    ! Scatter XS
    print *, 'Sig_s = ', mg_sig_s

    ! Chi
    print *, 'chi = ', mg_chi

    ! Phi
    print *, 'phi = ', phi

    ! Psi
    if (store_psi) then
      print *, 'psi = ', psi
    end if

    ! incoming
    print *, 'incident_x = ', mg_incident_x
    print *, 'incident_y = ', mg_incident_y

    ! k
    print *, 'k = ', keff

  end subroutine output_moments

end module state
