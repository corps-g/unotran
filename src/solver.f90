module solver
  ! ############################################################################
  ! Solve the transport equation using discrete ordinates
  ! ############################################################################

  implicit none
  
  contains
  
  ! Initialize all of the variables and solvers
  subroutine initialize_solver()
    ! ##########################################################################
    ! Initialize the solver including mesh, quadrature, flux containers, etc.
    ! ##########################################################################

    ! Use Statements
    use state, only : initialize_state, d_nu_sig_f, d_chi, d_sig_s, d_sig_t, &
                      d_source, source, d_delta, d_phi, phi, d_psi, psi, &
                      d_incoming
    use material, only : create_material, nu_sig_f, chi, sig_s, sig_t
    use angle, only : initialize_angle, number_angles, initialize_polynomials
    use mesh, only : create_mesh, number_cells, mMap
    use material, only : number_legendre
    use control, only : store_psi

    ! Variable definitions
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
      d_nu_sig_f(c, :) = nu_sig_f(:, mMap(c))
      d_chi(c, :) = chi(:, mMap(c))
      do l = 0, number_legendre
        d_sig_s(l, c, :, :) = sig_s(l, :, :, mMap(c))
      end do
      d_sig_t(c, :) = sig_t(:, mMap(c))
    end do
    d_source(:, :, :) = source(:, :, :)
    d_delta(:, :, :) = 0.0
    d_phi(:, :, :) = phi(:, :, :)
    if (store_psi) then
      d_psi(:, :, :) = psi(:, :, :)
      ! Default the incoming flux to be equal to the outgoing if present
      d_incoming = psi(1, (number_angles + 1):, :)
    else
      ! Assume isotropic scalar flux for incident flux
      do a = 1, number_angles
        d_incoming(a, :) = phi(0, 1, :) / 2
      end do
    end if

  end subroutine initialize_solver

  subroutine solve()
    ! ##########################################################################
    ! Solve the neutron transport equation using discrete ordinates
    ! ##########################################################################

    ! Use Statements
    use mg_solver, only : mg_solve
    use state, only : d_source, d_phi, d_psi, d_incoming
    use material, only : number_groups

    double precision :: &
        error                  ! inter-iteration error
    integer :: &
        counter                ! iteration counter

    call mg_solve(number_groups, d_source, d_phi, d_psi, d_incoming)

    print *, d_phi

!    ! Interate to convergance
!    do while (error > outer_tolerance)
!      ! save phi from previous iteration
!      d_phi = phi
!
!      ! Sweep through the mesh
!      call sweep(number_groups, phi, psi)
!
!      ! Compute the eigenvalue
!      if (solver_type == 'eigen') then
!        ! Compute new eigenvalue if eigen problem
!        d_keff = d_keff * sum(abs(phi(0,:,:))) / sum(abs(d_phi(0,:,:)))
!      end if
!
!      ! error is the difference in phi between successive iterations
!      error = sum(abs(phi - d_phi))
!
!      ! output the current error and iteration number
!      if (outer_print) then
!        if (solver_type == 'eigen') then
!          print *, 'Error = ', error, 'Iteration = ', counter, 'Eigenvalue = ', d_keff
!        else if (solver_type == 'fixed') then
!          print *, 'Error = ', error, 'Iteration = ', counter
!        end if
!      end if
!
!      call normalize_flux(number_groups, phi, psi)
!
!      ! increment the iteration
!      counter = counter + 1
!
!      ! Break out of loop if exceeding maximum outer iterations
!      if (counter > max_outer_iters) then
!        if (.not. ignore_warnings) then
!          print *, 'warning: exceeded maximum outer iterations'
!        end if
!        exit
!      end if
!
!    end do

  end subroutine solve

  subroutine output()
    ! ##########################################################################
    ! Output the fluxes to file
    ! ##########################################################################

    ! Use Statements
    use state, only : phi, psi, output_state

    print *, phi
    print *, psi
    call output_state()

  end subroutine output

  subroutine finalize_solver()
    ! ##########################################################################
    ! Deallocate all used variables
    ! ##########################################################################

    ! Use Statements
    use angle, only : finalize_angle
    use mesh, only : finalize_mesh
    use material, only : finalize_material
    use state, only : finalize_state

    call finalize_angle()
    call finalize_mesh()
    call finalize_material()
    call finalize_state()
  end subroutine

end module solver
