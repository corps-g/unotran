module mg_solver

  implicit none
  
  contains
  
  subroutine mg_solve()
    ! ##########################################################################
    ! Solve the multigroup group equation
    ! ##########################################################################

    ! Use Statements
    use control, only : ignore_warnings, max_outer_iters, outer_print, outer_tolerance, &
                        min_outer_iters, number_cells, number_groups
    use sweeper, only : apply_transport_operator
    use state, only : mg_phi
    use omp_lib, only : omp_get_wtime

    ! Variable definitions
    integer :: &
        outer_count    ! Counter for the outer loop
    double precision :: &
        outer_error    ! Residual error between iterations
    double precision, dimension(number_groups, number_cells) :: &
        old_phi
    double precision :: &
        start,       & ! Start time of the sweep function
        ave_sweep_time ! Average time in seconds per sweep

    ave_sweep_time = 0.0

    ! Begin loop to converge on the in-scattering source
    do outer_count = 1, max_outer_iters

      start = omp_get_wtime()

      ! Save the old flux
      old_phi = mg_phi(0,:,:)

      ! Update the scalar flux
      call apply_transport_operator(mg_phi)

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
        exit
      end if
      
      
    end do

    if (outer_count == max_outer_iters) then
      if (.not. ignore_warnings) then
        ! Warning if more iterations are required
        write(*, 1002) outer_count
        1002 format ('outer iteration did not converge in ', i4, ' iterations')
      end if
    end if

  end subroutine mg_solve

end module mg_solver
