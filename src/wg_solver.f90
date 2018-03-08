module wg_solver

  implicit none
  
  contains
  
  subroutine wg_solve(g, source, phi_g, psi_g, incident)
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
    double precision, intent(in), dimension(:,:) :: &
        source      ! Fission, In-Scattering, External source in group g
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
    double precision, dimension(number_cells, 2 * number_angles) :: &
        total_S     ! Sum of all sources

    ! Initialize container to hold the old scalar flux for error calculations
    allocate(phi_g_old(0:number_legendre, number_cells))
    phi_g_old = 0.0

    ! Begin loop to converge on the within-group scattering source
    do inner_count = 1, max_inner_iters
      ! Reset the source to only be in-scattering, fission, and external
      total_S = source

      ! Add the within-group scattering to the source
      call compute_within_scattering(g, phi_g, total_S)

      ! Sweep through the mesh
      call sweep(g, total_S, phi_g, psi_g, incident)

      ! Update the error
      inner_error = maxval(abs(phi_g_old - phi_g))

      ! save old value of scalar flux
      phi_g_old = phi_g

      ! Print output
      if (inner_print) then
        write(*, 1001) inner_count, inner_error
        1001 format ( "    Inner: ", i4, " Error: ", es12.5E2)
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

  subroutine compute_within_scattering(g, phi_g, source)
    ! ##########################################################################
    ! Compute the within-group scattering and add to the source term
    ! ##########################################################################

    ! Use Statements
    use angle, only : p_leg
    use state, only : d_sig_s
    use control, only : number_cells, number_angles, number_legendre

    ! Variable definitions
    integer, intent(in) :: &
        g       ! Group index
    double precision, intent(in), dimension(0:,:) :: &
        phi_g   ! Scalar flux for group g
    double precision, intent(inout), dimension(:,:) :: &
        source  ! Fission, Scattering, and External sources for group g
    integer :: &
        a,    & ! Angle index
        c,    & ! Cell index
        l       ! Legendre index

    ! Include the within-group scattering into source
    do a = 1, 2 * number_angles
      do c = 1, number_cells
        do l = 0, number_legendre
          source(c, a) = source(c, a) &
                         + 0.5 / (2 * l + 1) * p_leg(l, a) * d_sig_s(l, c, g, g) * phi_g(l, c)
        end do
      end do
    end do


  end subroutine compute_within_scattering

end module wg_solver
