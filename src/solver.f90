module solver
  ! ############################################################################
  ! Solve the transport equation using discrete ordinates
  ! ############################################################################

  use control, only : lamb, outer_print, outer_tolerance, store_psi, solver_type
  use material, only : create_material, number_legendre, number_groups, finalize_material, &
                       sig_t, sig_s, nu_sig_f, chi
  use angle, only : initialize_angle, p_leg, number_angles, initialize_polynomials, finalize_angle
  use mesh, only : create_mesh, number_cells, finalize_mesh, mMap
  use state, only : initialize_state, phi, psi, source, finalize_state, output_state, &
                    normalize_flux, update_density, &
                    d_source, d_nu_sig_f, d_delta, d_phi, d_chi, d_sig_s, d_sig_t, d_psi, d_keff, d_incoming
  use sweeper, only : sweep

  implicit none
  
  logical :: &
      printOption,  & ! Boolian flag to print to standard output or not
      use_fission     ! Flag to allow fission in the problem

  contains
  
  ! Initialize all of the variables and solvers
  subroutine initialize_solver()
    ! ##########################################################################
    ! Initialize the solver including mesh, quadrature, flux containers, etc.
    ! ##########################################################################

    integer :: &
        c,  & ! Cell index
        a,  & ! Angle index
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
      ! Default the incoming flux to be equal to the outgoing if present
      d_incoming = psi(:,(number_angles+1):,1)
    else
      ! Assume isotropic scalar flux for incident flux
      do a=1, number_angles
        d_incoming(:,a) = phi(0,:,1) / (number_angles * 2)
      end do
    end if

  end subroutine initialize_solver

  ! Interate equations until convergance
  subroutine solve()
    ! ##########################################################################
    ! Solve the neutron transport equation using discrete ordinates
    ! ##########################################################################

    double precision :: &
        error                  ! inter-iteration error
    integer :: &
        counter                ! iteration counter

    ! Error of current iteration
    error = 1.0

    ! interation number
    counter = 1

    ! Interate to convergance
    do while (error > outer_tolerance)
      ! save phi from previous iteration
      d_phi = phi

      ! Sweep through the mesh
      call sweep(number_groups, phi, psi)

      ! Compute the eigenvalue
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

      call normalize_flux(number_groups, phi, psi)

      ! increment the iteration
      counter = counter + 1
    end do

  end subroutine solve

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
  end subroutine

end module solver
