module wg_solver

  implicit none
  
  contains
  
  subroutine wg_solve(g, phi_g, psi_g, incident)
    ! ##########################################################################
    ! Solve the within group equation
    ! ##########################################################################

    ! Use Statements
    use control, only : ignore_warnings, max_inner_iters, inner_print, &
                        inner_tolerance, number_cells, number_legendre, number_angles
    use sweeper, only : sweep

    ! Variable definitions
    integer, intent(in) :: &
        g           ! Group index
    double precision, intent(inout), dimension(0:,:) :: &
        phi_g       ! Scalar flux
    double precision, intent(inout), dimension(:,:) :: &
        psi_g       ! Angular flux
    double precision, intent(inout), dimension(:) :: &
        incident    ! Angular flux incident on the cell in group g
    double precision, allocatable, dimension(:,:) :: &
        phi_g_old   ! Save the previous scalar flux
    integer :: &
        inner_count ! Counter for the inner loop
    double precision :: &
        inner_error ! Residual error between iterations

    ! Initialize container to hold the old scalar flux for error calculations
    allocate(phi_g_old(0:number_legendre, number_cells))
    phi_g_old = 0.0

    ! Begin loop to converge on the within-group scattering source
    do inner_count = 1, max_inner_iters

      ! Sweep through the mesh
      call sweep(g, phi_g, psi_g, incident)

      ! Update the error
      inner_error = maxval(abs(phi_g_old - phi_g))

      ! save old value of scalar flux
      phi_g_old = phi_g

      ! Print output
      if (inner_print > 0) then
        write(*, 1001) inner_count, inner_error
        1001 format ( "    Inner: ", i4, " Error: ", es12.5E2)
        if (inner_print > 1) then
          print *, phi_g
        end if
      end if

      ! Check if tolerance is reached
      if (inner_error < inner_tolerance) then
        exit
      end if
    end do

    if (inner_count == max_inner_iters) then
      if (.not. ignore_warnings) then
        ! Warning if more iterations are required
        write(*, 1002) inner_count
        1002 format ('inner iteration did not converge in ', i4, ' iterations')
      end if
    end if

    ! Deallocate memory
    deallocate(phi_g_old)

  end subroutine wg_solve

end module wg_solver
