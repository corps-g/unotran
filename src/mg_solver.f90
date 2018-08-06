module mg_solver

  implicit none
  
  contains
  
  subroutine mg_solve()
    ! ##########################################################################
    ! Solve the within group equation
    ! ##########################################################################

    ! Use Statements
    use control, only : ignore_warnings, max_outer_iters, outer_print, outer_tolerance, &
                        number_groups, number_cells, number_legendre, min_outer_iters
    use wg_solver, only : wg_solve
    use state, only : mg_phi, mg_psi, mg_incoming
    use dgm, only : dgm_order

    ! Variable definitions
    double precision, dimension(0:number_legendre, number_cells, number_groups) :: &
        phi_new              ! Save the previous scalar flux
    integer :: &
        outer_count,       & ! Counter for the outer loop
        g                    ! Group index
    double precision :: &
        outer_error          ! Residual error between iterations

    ! Save the scalar flux from previous iteration
    phi_new = mg_phi

    ! Begin loop to converge on the in-scattering source
    do outer_count = 1, max_outer_iters
      ! Loop over the energy groups
      do g = 1, number_groups
        ! Solve the within group problem
        call wg_solve(g, phi_new(:,:,g), mg_psi(:,:,g), mg_incoming(:,g))
      end do

      ! Update the error
      outer_error = maxval(abs(phi_new - mg_phi))

      ! Update the scalar flux
      mg_phi = phi_new

      ! Check for NaN during convergence
      if (outer_error /= outer_error) then
        print *, "NaN detected...exiting"
        stop
      end if

      ! Print output
      if (outer_print > 0) then
        write(*, 1001) outer_count, outer_error
        1001 format ( "  outer: ", i4, " Error: ", es12.5E2)
        if (outer_print > 1) then
          print *, mg_phi
        end if
      end if

      ! Check if tolerance is reached
      if ((outer_error < outer_tolerance .and. outer_count >= min_outer_iters) .or. dgm_order > 0) then
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
