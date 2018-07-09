module solver
  ! ############################################################################
  ! Solve the transport equation using discrete ordinates
  ! ############################################################################

  implicit none
  
  contains
  
  subroutine initialize_solver()
    ! ##########################################################################
    ! Initialize the solver including mesh, quadrature, flux containers, etc.
    ! ##########################################################################

    ! Use Statements
    use state, only : initialize_state, d_nu_sig_f, d_chi, d_sig_s, d_sig_t, &
                      d_source, source, d_phi, phi, d_psi, psi, &
                      d_incoming
    use material, only : nu_sig_f, chi, sig_s, sig_t
    use mesh, only : mMap
    use control, only : store_psi, number_cells, number_angles, number_legendre

    ! Variable definitions
    integer :: &
        c,  & ! Cell index
        a,  & ! Angle index
        l     ! Legendre moment index

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

  subroutine solve(higher_dgm_arg)
    ! ##########################################################################
    ! Solve the neutron transport equation using discrete ordinates
    ! ##########################################################################

    ! Use Statements
    use mg_solver, only : mg_solve
    use state, only : d_source, d_phi, d_psi, d_incoming, d_keff, normalize_flux, &
                      phi, psi, update_fission_density
    use control, only : solver_type, eigen_print, ignore_warnings, max_eigen_iters, &
                        eigen_tolerance, number_cells, number_groups, number_legendre, &
                        use_DGM, number_angles

    ! Variable definitions
    logical, intent(in), optional :: &
        higher_dgm_arg  ! Set the solver to fixed source if on Higher DGM moments
    logical, parameter :: &
        higher_dgm_default = .false.  ! Default value of higher_dgm_arg
    logical :: &
        higher_dgm_flag ! Local variable to signal an eigen loop bypass
    double precision :: &
        eigen_error     ! Error between successive iterations
    integer :: &
        eigen_count     ! Iteration counter
    double precision, dimension(0:number_legendre, number_cells, number_groups) :: &
        old_phi         ! Scalar flux from previous iteration
    double precision, dimension(number_cells, 2 * number_angles, number_groups) :: &
        f_source        ! Source containing fission term

    if (present(higher_dgm_arg)) then
      higher_dgm_flag = higher_dgm_arg
    else
      higher_dgm_flag = higher_dgm_default
    end if

    ! Run eigen loop only if eigen problem
    if (solver_type == 'fixed' .or. higher_dgm_flag) then
      call mg_solve(d_source, d_phi, d_psi, d_incoming, higher_dgm_flag)
    else if (solver_type == 'eigen') then
      do eigen_count = 1, max_eigen_iters

        ! Save the old value of the scalar flux
        old_phi = d_phi

        ! Update the fission source
        call compute_source(d_phi, f_source)

        ! Solve the multigroup problem
        call mg_solve(f_source, d_phi, d_psi, d_incoming, higher_dgm_flag)

        ! Compute new eigenvalue if eigen problem
        d_keff = norm2(d_phi(0,:,:)) / norm2(old_phi(0,:,:))

        ! Normalize the fluxes
        call normalize_flux(d_phi, d_psi)

        ! Update the error
        eigen_error = sum(abs(d_phi - old_phi))

        ! Print output
        if (eigen_print > 0) then
          write(*, 1001) eigen_count, eigen_error, d_keff
          1001 format ( "eigen: ", i4, " Error: ", es12.5E2, " eigenvalue: ", f12.9)
          if (eigen_print > 1) then
            print *, d_phi
          end if
        end if

        ! Check if tolerance is reached
        if (eigen_error < eigen_tolerance) then
          exit
        end if

      end do

      if (eigen_count == max_eigen_iters) then
        if (.not. ignore_warnings) then
          ! Warning if more iterations are required
          write(*, 1002) eigen_count
          1002 format ('eigen iteration did not converge in ', i4, ' iterations')
        end if
      end if
    end if

    if (.not. use_dgm) then
      phi = d_phi
      psi = d_psi

      ! Compute the fission density
      call update_fission_density()
    end if

  end subroutine solve

  subroutine compute_source(phi, source)
    ! ##########################################################################
    ! Add the fission and external sources
    ! ##########################################################################

    ! Use Statements
    use state, only : d_chi, d_nu_sig_f, d_keff, d_source
    use control, only : source_value, number_cells, number_groups, use_DGM

    ! Variable definitions
    double precision, intent(in), dimension(0:,:,:) :: &
      phi     ! Scalar flux
    double precision, intent(inout), dimension(:,:,:) :: &
      source  ! Container to hold the computed source
    integer :: &
      c,    & ! Cell index
      g       ! Energy index


    do g = 1, number_groups
      do c = 1, number_cells
        source(c,:,g) = d_source(c,:,g) + 0.5 * d_chi(c, g) * dot_product(d_nu_sig_f(c,:), phi(0,c,:))
      end do
    end do

  end subroutine compute_source

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
    use state, only : finalize_state

    call finalize_state()
  end subroutine

end module solver
