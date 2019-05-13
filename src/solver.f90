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
                      mg_phi, phi, mg_psi, psi, mg_mMap, &
                      update_fission_density
    use material, only : nu_sig_f, chi, sig_s, sig_t
    use mesh, only : mMap
    use control, only : store_psi, number_regions,scatter_leg_order

    ! Variable definitions
    integer :: &
        r     ! Region index

    ! allocate the solutions variables
    call initialize_state()

    ! Fill multigroup arrays with the fine group data
    mg_mMap(:) = mMap(:)
    do r = 1, number_regions
      mg_nu_sig_f(:,r) = nu_sig_f(:,r)
      mg_chi(:,r) = chi(:,r)
      mg_sig_s(:,:,:,r) = sig_s(:scatter_leg_order,:,:,r)
      mg_sig_t(:,r) = sig_t(:,r)
    end do  ! End r loop
    mg_phi(:, :, :) = phi(:, :, :)
    if (store_psi) then
      mg_psi(:, :, :) = psi(:, :, :)
    end if

    ! Update the fission density
    call update_fission_density()

  end subroutine initialize_solver

  subroutine solve()
    ! ##########################################################################
    ! Solve the neutron transport equation using discrete ordinates
    ! ##########################################################################

    ! Use Statements
    use mg_solver, only : mg_solve
    use state, only : mg_phi, mg_psi, keff, normalize_flux, &
                      phi, psi, update_fission_density
    use control, only : solver_type, eigen_print, ignore_warnings, max_eigen_iters, &
                        eigen_tolerance, number_cells, number_groups, &
                        use_DGM, min_eigen_iters, store_psi, eigen_converged, &
                        outer_converged, number_moments
    use dgm, only : dgm_order
    use omp_lib, only : omp_get_wtime

    ! Variable definitions
    double precision :: &
        eigen_error,  & ! Error between successive iterations
        start,        & ! Start time of the sweep function
        ave_sweep_time  ! Average time in seconds per sweep
    integer :: &
        eigen_count     ! Iteration counter
    double precision, dimension(0:number_moments, number_groups, number_cells) :: &
        old_phi         ! Scalar flux from previous iteration

    ave_sweep_time = 0.0

    ! Initialize the eigen convergence flag to False
    eigen_converged = .false.

    ! Run eigen loop only if eigen problem
    if (solver_type == 'fixed' .or. dgm_order > 0) then
      call mg_solve()
    else if (solver_type == 'eigen') then
      do eigen_count = 1, max_eigen_iters

        start = omp_get_wtime()

        ! Update the fission density
        call update_fission_density()

        ! Save the old value of the scalar flux
        old_phi = mg_phi

        ! Solve the multigroup problem
        call mg_solve()

        ! Compute new eigenvalue if eigen problem
        keff = keff * sum(abs(mg_phi(0,:,:))) / sum(abs(old_phi(0,:,:)))

        ! Normalize the fluxes
        call normalize_flux(mg_phi, mg_psi)

        ! Update the error
        eigen_error = maxval(abs(mg_phi - old_phi))

        ave_sweep_time = ((eigen_count - 1) * ave_sweep_time + (omp_get_wtime() - start)) / eigen_count

        ! Print output
        if (eigen_print > 0) then
          write(*, 1001) eigen_count, eigen_error, keff, ave_sweep_time, outer_converged
          1001 format ( "  eigen: ", i4, " Error: ", es12.5E2, " Eigenvalue: ", f14.10, " AveSweepTime: ", f5.2, " s", &
                        " OuterConverged: ", L1)
          if (eigen_print > 1) then
            print *, mg_phi
          end if
        end if

        ! Check if tolerance is reached
        if (eigen_error < eigen_tolerance .and. eigen_count >= min_eigen_iters) then
          ! Set the eigen convergence flag to True
          eigen_converged = .true.
          ! If the inner iterations are converged, exit
          if (outer_converged) then
            exit
          end if
        else
          ! Set the eigen convergence flag to False
          eigen_converged = .false.
        end if

      end do  ! End eigen_count loop

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
      if (store_psi) then
        psi = mg_psi
      end if

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
