module mg_solver

  implicit none
  
  contains
  
  subroutine mg_solve(phi, psi, incident)
    ! ##########################################################################
    ! Solve the within group equation
    ! ##########################################################################

    ! Use Statements
    use control, only : ignore_warnings, max_outer_iters, outer_print, outer_tolerance, &
                        number_groups, number_cells, number_angles, number_legendre
    use wg_solver, only : wg_solve
    use sources, only : compute_in_source
    use dgm, only : dgm_order

    ! Variable definitions
    double precision, intent(inout), dimension(0:,:,:) :: &
        phi                  ! Scalar flux
    double precision, intent(inout), dimension(:,:,:) :: &
        psi                  ! Angular flux
    double precision, intent(inout), dimension(:,:) :: &
        incident             ! Angular flux incident on the cell
    double precision, allocatable, dimension(:,:,:) :: &
        phi_old              ! Save the previous scalar flux
    integer :: &
        outer_count,       & ! Counter for the outer loop
        l,                 & ! Legendre index
        g                    ! Group index
    double precision :: &
        outer_error          ! Residual error between iterations

    ! Save the scalar flux from previous iteration
    allocate(phi_old(0:number_legendre, number_cells, number_groups))
    phi_old = 0.0

    ! Begin loop to converge on the in-scattering source
    do outer_count = 1, max_outer_iters
      ! save old value of scalar flux
      phi_old = phi

      ! Loop over the energy groups
      do g = 1, number_groups
        ! Compute the into group sources for group g
        call compute_in_source(g)

        ! Solve the within group problem
        call wg_solve(g, phi(:,:,g), psi(:,:,g), incident(:,g))

      end do

      ! Update the error
      outer_error = maxval(abs(phi_old - phi))

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
          print *, phi
        end if
      end if

      ! Check if tolerance is reached
      if (outer_error < outer_tolerance .or. dgm_order > 0) then
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

    ! Deallocate memory
    deallocate(phi_old)

  end subroutine mg_solve

end module mg_solver
