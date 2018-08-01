module solver
  ! ############################################################################
  ! Solve the transport equation using discrete ordinates
  ! ############################################################################

  implicit none
  
  contains
  
  subroutine initialize_solver()
    ! ##########################################################################
    ! Initialize the solver including mesh, quadrature, flux containers, etc.
    ! The mg containers in state are set to fine group size
    ! ##########################################################################

    ! Use Statements
    use state, only : initialize_state, mg_nu_sig_f, mg_chi, mg_sig_s, mg_sig_t, &
                      mg_phi, phi, mg_psi, psi, &
                      mg_incoming, mg_mMap
    use material, only : nu_sig_f, chi, sig_s, sig_t
    use mesh, only : mMap
    use control, only : store_psi, number_cells, number_angles, number_legendre, &
                        number_regions

    ! Variable definitions
    integer :: &
        c,  & ! Cell index
        a,  & ! Angle index
        r,  & ! Region index
        l     ! Legendre moment index

    ! allocate the solutions variables
    call initialize_state()

    ! Fill multigroup arrays with the fine group data
    mg_mMap(:) = mMap(:)
    do r = 1, number_regions
      mg_nu_sig_f(r,:) = nu_sig_f(:,r)
      mg_chi(r,:) = chi(:,r)
      do l = 0, number_legendre
        mg_sig_s(l,r,:,:) = sig_s(l,:,:,r)
      end do
      mg_sig_t(r,:) = sig_t(:,r)
    end do
    mg_phi(:, :, :) = phi(:, :, :)
    if (store_psi) then
      mg_psi(:, :, :) = psi(:, :, :)
      ! Default the incoming flux to be equal to the outgoing if present
      mg_incoming = psi(1, (number_angles + 1):, :)
    else
      ! Assume isotropic scalar flux for incident flux
      do a = 1, number_angles
        mg_incoming(a, :) = phi(0, 1, :) / 2
      end do
    end if

  end subroutine initialize_solver

  subroutine solve()
    ! ##########################################################################
    ! Solve the neutron transport equation using discrete ordinates
    ! ##########################################################################

    ! Use Statements
    use mg_solver, only : mg_solve
    use state, only : mg_phi, mg_psi, mg_incoming, keff, normalize_flux, &
                      phi, psi, update_fission_density
    use control, only : solver_type, eigen_print, ignore_warnings, max_eigen_iters, &
                        eigen_tolerance, number_cells, number_groups, number_legendre, &
                        use_DGM, number_angles
    use dgm, only : dgm_order

    ! Variable definitions
    double precision :: &
        eigen_error     ! Error between successive iterations
    integer :: &
        eigen_count     ! Iteration counter
    double precision, dimension(0:number_legendre, number_cells, number_groups) :: &
        old_phi         ! Scalar flux from previous iteration

    ! Run eigen loop only if eigen problem
    if (solver_type == 'fixed' .or. dgm_order > 0) then
      call mg_solve(mg_phi, mg_psi, mg_incoming)
    else if (solver_type == 'eigen') then
      do eigen_count = 1, max_eigen_iters

        ! Save the old value of the scalar flux
        old_phi = mg_phi

        ! Solve the multigroup problem
        call mg_solve(mg_phi, mg_psi, mg_incoming)

        ! Compute new eigenvalue if eigen problem
        keff = keff * sum(abs(mg_phi(0,:,:))) / sum(abs(old_phi(0,:,:)))

        ! Normalize the fluxes
        call normalize_flux(mg_phi, mg_psi)

        ! Update the error
        eigen_error = maxval(abs(mg_phi - old_phi))

        ! Print output
        if (eigen_print > 0) then
          write(*, 1001) eigen_count, eigen_error, keff
          1001 format ( "eigen: ", i4, " Error: ", es12.5E2, " eigenvalue: ", f12.9)
          if (eigen_print > 1) then
            print *, mg_phi
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
      phi = mg_phi
      psi = mg_psi

      ! Compute the fission density
      call update_fission_density()
    end if

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
    use state, only : finalize_state

    call finalize_state()
  end subroutine

end module solver
