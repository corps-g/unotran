module mg_solver

  implicit none
  
  contains
  
  subroutine mg_solve()
    ! ##########################################################################
    ! Solve the within group equation
    ! ##########################################################################

    ! Use Statements
    use control, only : ignore_warnings, max_outer_iters, outer_print, outer_tolerance, &
                        min_outer_iters
    use sources, only : compute_source
    use sweeper, only : apply_transport_operator
    use state, only : mg_phi, mg_source

    ! Variable definitions
    integer :: &
        outer_count    ! Counter for the outer loop
    double precision :: &
        outer_error    ! Residual error between iterations

    ! Begin loop to converge on the in-scattering source
    do outer_count = 1, max_outer_iters

      ! Update the forcing function
      call compute_source()

      ! Update the scalar flux
      call apply_transport_operator(mg_phi)

      ! Update the error
      outer_error = maxval(abs(mg_phi(0,:,:) - mg_source(:,:)))

      ! Check for NaN during convergence
      if (outer_error /= outer_error) then
        print *, "NaN detected...exiting"
        stop
      end if

      ! Print output
      if (outer_print > 0) then
        write(*, 1001) outer_count, outer_error
        1001 format ( "    outer: ", i4, " Error: ", es12.5E2)
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
