module solver
  ! ############################################################################
  ! Solve the transport equation using discrete ordinates
  ! ############################################################################

  use control, only : lamb, outer_print, outer_tolerance, store_psi, solver_type
  use material, only : create_material, number_legendre, number_groups, finalize_material, &
                       sig_t, sig_s, nu_sig_f, chi
  use angle, only : initialize_angle, p_leg, number_angles, initialize_polynomials, finalize_angle
  use mesh, only : create_mesh, number_cells, finalize_mesh, mMap
  use state, only : initialize_state, phi, psi, source, finalize_state, output_state, update_density, &
                    d_source, d_nu_sig_f, d_delta, d_phi, d_chi, d_sig_s, d_sig_t, d_psi, d_keff, density
  use sweeper, only : sweep

  implicit none
  
  logical :: &
      printOption,  & ! Boolian flag to print to standard output or not
      use_fission     ! Flag to allow fission in the problem
  double precision, allocatable :: &
      incoming(:,:)  ! Angular flux incident on the current cell

  contains
  
  ! Initialize all of the variables and solvers
  subroutine initialize_solver()
    ! ##########################################################################
    ! Initialize the solver including mesh, quadrature, flux containers, etc.
    ! ##########################################################################

    integer :: &
        c,  & ! Cell index
        l     ! Legendre moment index

    ! initialize the mesh
    call create_mesh()
    ! read the material cross sections
    call create_material()
    ! initialize the angle quadrature
    call initialize_angle()
    ! get the basis vectors
    call initialize_polynomials(number_legendre)
    ! allocate the solutions variables
    call initialize_state()

    ! Fill container arrays
    do c = 1, number_cells
      d_nu_sig_f(:, c) = nu_sig_f(:, mMap(c))
      d_chi(:, c) = chi(:, mMap(c))
      do l = 0, number_legendre
        d_sig_s(l, :, :, c) = sig_s(l, :, :, mMap(c))
      end do
      d_sig_t(:, c) = sig_t(:, mMap(c))
    end do
    d_source(:, :, :) = source(:, :, :)
    d_delta(:, :, :) = 0.0
    d_phi(:, :, :) = phi(:, :, :)
    if (store_psi) then
      d_psi(:, :, :) = psi(:, :, :)
    end if

    ! todo: move to state
    allocate(incoming(number_groups, number_angles))
    incoming = 0.0

  end subroutine initialize_solver

  ! Interate equations until convergance
  subroutine solve()
    ! ##########################################################################
    ! Initialize the solver including mesh, quadrature, flux containers, etc.
    ! ##########################################################################

    double precision :: &
        error,               & ! inter-iteration error
    integer :: &
        counter,             & ! iteration counter
        a                      ! angle index

    ! Error of current iteration
    error = 1.0

    ! interation number
    counter = 1

    call normalize_flux()

    ! Interate to convergance
    do while (error > outer_tolerance)
      ! save phi from previous iteration
      d_phi = phi

      ! Sweep through the mesh
      call sweep(number_groups, phi, psi, incoming)

      if (solver_type == 'eigen') then
        ! Compute new eigenvalue if eigen problem
        d_keff = d_keff * sum(abs(phi(0,:,:))) / sum(abs(d_phi(0,:,:)))
      end if

      ! error is the difference in phi between successive iterations
      error = sum(abs(phi - d_phi))

      ! output the current error and iteration number
      if (outer_print) then
        if (solver_type == 'eigen') then
          print *, 'Error = ', error, 'Iteration = ', counter, 'Eigenvalue = ', d_keff
        else if (solver_type == 'fixed') then
          print *, 'Error = ', error, 'Iteration = ', counter
        end if
      end if

      if (solver_type == 'eigen') then
       call normalize_flux()
      end if

      ! increment the iteration
      counter = counter + 1
    end do

  end subroutine solve

  subroutine normalize_flux()
    ! ##########################################################################
    ! Normalize the scalar flux, used during eigenvalue solves
    ! ##########################################################################

    double precision :: &
        frac  ! Normalization fraction

    frac = sum(abs(phi(0,:,:))) / (number_cells * number_groups)

    ! normalize scalar flux
    phi = phi / frac

    ! normalize angular flux
    if (store_psi) then
        psi = psi / frac
    end if

  end subroutine normalize_flux

  subroutine output()
    ! ##########################################################################
    ! Output the fluxes to file
    ! ##########################################################################

    call output_state()

  end subroutine output

  subroutine finalize_solver()
    ! ##########################################################################
    ! Deallocate all used variables
    ! ##########################################################################

    call finalize_angle()
    call finalize_mesh()
    call finalize_material()
    call finalize_state()
    if (allocated(incoming)) then
      deallocate(incoming)
    end if
  end subroutine

end module solver
