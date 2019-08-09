module state
  ! ############################################################################
  ! Define the container classes for fluxes and cross sections
  ! ############################################################################

  use control, only : dp

  implicit none

  real(kind=dp), allocatable, dimension(:,:,:,:) :: &
      mg_incident_x,       & ! Angular flux incident on the current cell in x direction
      mg_incident_y,       & ! Angular flux incident on the current cell in y direction
      mg_sig_s               ! Scattering cross section moments
  real(kind=dp), allocatable, dimension(:,:,:) :: &
      psi,                 & ! Angular flux
      phi,                 & ! Scalar Flux
      mg_phi,              & ! Scalar flux mg container
      mg_psi,              & ! Angular flux mg container
      sigphi                 ! Scalar flux container
  real(kind=dp), allocatable, dimension(:,:) :: &
      mg_nu_sig_f,         & ! Fission cross section mg container (times nu)
      mg_chi,              & ! Chi spectrum mg container
      mg_source,           & ! External source mg container
      mg_sig_t               ! Scalar total cross section mg container
  real(kind=dp), allocatable, dimension(:) :: &
      mg_density             ! Fission density
  integer, allocatable, dimension(:) :: &
      mg_mMap                ! material map mg container
  real(kind=dp) :: &
      mg_constant_source,  & ! Constant multigroup source
      keff,                & ! k-eigenvalue
      norm_frac,           & ! Fraction of normalization for eigenvalue problems
      scaling,             & ! Scaling factor for source terms
      recon_convergence_rate ! Approximate rate of convergence for recon iters
  integer :: &
      exit_status,         & ! Allows setting exit signals
      sweep_count,         & ! Counter for the number of transport sweeps
      recon_count,         & ! Number of recon iterations
      eigen_count,         & ! Number of eigen iterations
      outer_count            ! Number of outer iterations
  
  contains
  
  subroutine initialize_state(skip_init_material)
    ! ##########################################################################
    ! Initialize each of the container variables to default/entered values
    ! ##########################################################################

    ! Use Statements
    use control, only : number_fine_groups, number_coarse_groups, number_groups, &
                        use_DGM, ignore_warnings, initial_keff, initial_phi, number_angles, &
                        initial_psi, number_angles_per_octant, number_cells, number_legendre, &
                        solver_type, source_value, store_psi, check_inputs, &
                        verify_control, homogenization_map, number_regions, &
                        scatter_leg_order, delta_leg_order, truncate_delta, &
                        number_cells_x, number_cells_y, spatial_dimension, number_moments
    use mesh, only : create_mesh
    use material, only : create_material, number_materials
    use angle, only : initialize_angle, initialize_polynomials, PI
    use dgm, only : initialize_moments, initialize_basis, compute_expanded_cross_sections

    ! Variable definitions
    integer :: &
        ios = 0, & ! I/O error status
        c,       & ! Cell index
        a          ! Angle index
    logical, intent(in), optional :: &
        skip_init_material                     ! Controls intializing the material
    logical :: &
        skip_init_material_default = .false., & ! Controls the default for intializing the material
        skip_init_material_val                 ! Holds actual value of init_material_flag

    if (present(skip_init_material)) then
      skip_init_material_val = skip_init_material
    else
      skip_init_material_val = skip_init_material_default
    end if

    ! Initialize the sub-modules
    ! read the material cross sections
    if (.not. skip_init_material_val) then
      call create_material()
    end if
    ! Determine the Legendre order of phi to store
    if (delta_leg_order == -1) then
      delta_leg_order = scatter_leg_order
    end if
    number_legendre = max(delta_leg_order, scatter_leg_order)
    if (spatial_dimension == 1) then
      number_moments = number_legendre
    else
      number_moments = (number_legendre + 1) ** 2 - 1
    end if
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
      if (.not. skip_init_material_val) then
        call compute_expanded_cross_sections()
      end if
      number_groups = number_coarse_groups
      if (allocated(homogenization_map)) then
        number_regions = maxval(homogenization_map)
      else
        allocate(homogenization_map(number_cells))
        number_regions = number_cells
        do c = 1, number_cells
          homogenization_map(c) = c
        end do  ! End c loop
      end if
    end if

    if (spatial_dimension == 1) then
      scaling = 0.5_8
    else
      scaling = 1.0_8 / (2.0_8 * PI)
    end if

    ! Initialize the constant source
    mg_constant_source = source_value * scaling

    ! Set the sweep counter to zero
    sweep_count = 0

    ! Allocate the scalar flux and source containers
    allocate(phi(0:number_moments, number_fine_groups, number_cells))
    ! Initialize phi
    ! Attempt to read file or use default if file does not exist
    open(unit = 10, status='old', file=initial_phi, form='unformatted', iostat=ios)
    if (ios > 0) then
      if (.not. ignore_warnings) then
        print *, "initial phi file, ", initial_phi, " is missing, using default value"
      end if
      if (solver_type == 'fixed') then
        phi = 1.0_8  ! default value
      else if (solver_type == 'eigen') then
        phi = 1.0_8
      end if
    else
      read(10) phi ! read the data in array x to the file
    end if
    close(10) ! close the file

    ! Only allocate psi if the option is to store psi
    if (store_psi) then
      allocate(psi(number_fine_groups, number_angles, number_cells))

      ! Initialize psi
      ! Attempt to read file or use default if file does not exist
      open(unit = 10, status='old', file=initial_psi, form='unformatted', iostat=ios)
      if (ios > 0) then
        if (.not. ignore_warnings) then
          print *, "initial psi file, ", initial_psi, " is missing, using default value"
        end if
        if (solver_type == 'fixed') then
          ! default to isotropic distribution
          psi = 0.5_8
        else
          ! default to isotropic distribution
          psi = 0.0_8
          do c = 1, number_cells
            do a = 1, number_angles
              psi(:, a, c) = phi(0, :, c) * scaling
            end do  ! End a loop
          end do  ! End c loop
        end if
      else
        read(10) psi ! read the data in array x to the file
      end if
      close(10) ! close the file
    end if

    ! Initialize the angular flux incident on the boundary
    if (spatial_dimension == 1) then
      allocate(mg_incident_x(number_groups, number_angles_per_octant, number_cells_y, 1))
      allocate(mg_incident_y(number_groups, number_angles_per_octant, number_cells_x, 1))
    else if (spatial_dimension == 2) then
      allocate(mg_incident_x(number_groups, number_angles_per_octant, number_cells_y, 4))
      allocate(mg_incident_y(number_groups, number_angles_per_octant, number_cells_x, 4))
    end if

    ! Assume vacuum conditions for incident flux
    mg_incident_x(:, :, :, :) = 0.0_8
    mg_incident_y(:, :, :, :) = 0.0_8

    ! Initialize the eigenvalue to unity if fixed problem or default for eigen
    if (solver_type == 'fixed') then
      keff = 1.0_8
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
    allocate(mg_phi(0:number_moments, number_groups, number_cells))
    allocate(mg_chi(number_groups, number_regions))
    allocate(mg_sig_s(0:scatter_leg_order, number_groups, number_groups, number_regions))
    if (spatial_dimension == 1) then
      allocate(sigphi(0:scatter_leg_order, number_groups, number_cells))
    else
      allocate(sigphi(0:(scatter_leg_order + 1) ** 2 - 1, number_groups, number_cells))
    end if
    if (store_psi) then
      allocate(mg_psi(number_groups, number_angles, number_cells))
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
    real(kind=dp), intent(inout), dimension(:,:,:) :: &
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
    mg_density = 0.0_8

    ! Sum the fission reaction rate over groups for each cell
    if (fine_flag_val) then
      ! Compute fission density using fine flux
      do c = 1, number_cells
        mg_density(c) = sum(nu_sig_f(:, mMap(c)) * phi(0,:,c))
      end do  ! End c loop
    else if (use_DGM .and. dgm_order > 0) then
      ! Compute the fission density using the mg_flux for higher moments
      do c = 1, number_cells
        mg_density(c) = sum(mg_nu_sig_f(:, mg_mMap(c)) * phi_m(0, 0,:,c))
      end do  ! End c loop
    else
      ! Compute the fission density using the mg_flux normally
      do c = 1, number_cells
        mg_density(c) = sum(mg_nu_sig_f(:, mg_mMap(c)) * mg_phi(0,:,c))
      end do  ! End c loop
    end if

  end subroutine update_fission_density

  subroutine output_moments()
    ! ##########################################################################
    ! Print the output of the cross sections to the standard output
    ! ##########################################################################

    use control, only : store_psi, use_DGM
    use dgm, only : delta_m

    ! Total XS
    print *, 'Sig_t = ', mg_sig_t

    ! Fission XS
    print *, 'vSig_f = ', mg_nu_sig_f

    ! Scatter XS
    print *, 'Sig_s = ', mg_sig_s

    ! Chi
    print *, 'chi = ', mg_chi

    ! Phi
    print *, 'phi = ', mg_phi

    ! Psi
    if (store_psi) then
      print *, 'psi = ', mg_psi
    end if

    ! Delta
    if (use_DGM) then
      print *, 'delta = ', delta_m
    end if

    ! incoming
    print *, 'incident_x = ', mg_incident_x
    print *, 'incident_y = ', mg_incident_y

    ! k
    print *, 'k = ', keff

  end subroutine output_moments

end module state
