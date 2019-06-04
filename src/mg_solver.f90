module mg_solver

  use control, only : dp

  implicit none
  
  contains
  
  subroutine mg_solve()
    ! ##########################################################################
    ! Solve the multigroup group equation
    ! ##########################################################################

    ! Use Statements
    use control, only : ignore_warnings, max_outer_iters, outer_print, outer_tolerance, &
                        min_outer_iters, number_cells, number_groups, spatial_dimension, &
                        outer_converged, eigen_converged, max_eigen_iters
    use sweeper_1D, only : apply_transport_operator_1D
    use sweeper_2D, only : apply_transport_operator_2D
    use state, only : mg_phi, outer_count
    use omp_lib, only : omp_get_wtime
    use dgm, only : dgm_order

    ! Variable definitions
    real(kind=dp) :: &
        outer_error    ! Residual error between iterations
    real(kind=dp), dimension(number_groups, number_cells) :: &
        old_phi
    real(kind=dp) :: &
        start,       & ! Start time of the sweep function
        ave_sweep_time ! Average time in seconds per sweep
    integer :: &
        outer_iters    ! Number of outer iterations to do

    ave_sweep_time = 0.0_8

    ! Initialize the outer convergence flag to False
    outer_converged = .false.

    if (dgm_order > 0) then
      outer_iters = max(max_eigen_iters, max_outer_iters)
    else
      outer_iters = max_outer_iters * merge(100, 1, eigen_converged)
    end if

    ! Begin loop to converge on the in-scattering source
    do outer_count = 1, outer_iters

      start = omp_get_wtime()

      ! Save the old flux
      old_phi = mg_phi(0,:,:)

      ! Update the scalar flux
      if (spatial_dimension == 1) then
        call apply_transport_operator_1D(mg_phi)
      else if (spatial_dimension == 2) then
        call apply_transport_operator_2D(mg_phi)
      else
        print *, "Dimension ", spatial_dimension, " has not been implemented"
        stop
      end if

      ! Update the error
      outer_error = maxval(abs(mg_phi(0,:,:) - old_phi(:,:)))

      ! Check for NaN during convergence
      if (outer_error /= outer_error) then
        print *, "NaN detected...exiting"
        stop
      end if

      ave_sweep_time = ((outer_count - 1) * ave_sweep_time + (omp_get_wtime() - start)) / outer_count

      ! Print output
      if (outer_print > 0) then
        write(*, 1001) outer_count, outer_error, ave_sweep_time
        1001 format ( "    outer: ", i4, " Error: ", es12.5E2, " ave sweep time: ", f5.2, " s")
        if (outer_print > 1) then
          print *, mg_phi
        end if
      end if

      ! Check if tolerance is reached
      if ((outer_error < outer_tolerance .and. outer_count >= min_outer_iters)) then
        ! Set the converged flag to True
        outer_converged = .true.
        exit
      end if
      
    end do  ! End outer_count loop

    if (outer_count == max_outer_iters) then
      if (.not. ignore_warnings) then
        ! Warning if more iterations are required
        write(*, 1002) outer_count
        1002 format ('outer iteration did not converge in ', i4, ' iterations')
      end if
    end if

  end subroutine mg_solve

end module mg_solver
