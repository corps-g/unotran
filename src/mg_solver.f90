module mg_solver

  implicit none
  
  contains
  
  subroutine mg_solve(number_energy_groups, source, phi, psi, incident)
    ! ##########################################################################
    ! Solve the within group equation
    ! ##########################################################################

    ! Use Statements
    use control, only : ignore_warnings, max_outer_iters, outer_print, outer_tolerance
    use angle, only : number_angles
    use mesh, only : number_cells
    use material, only : number_legendre
    use wg_solver, only : wg_solve

    ! Variable definitions
    integer, intent(in) :: &
        number_energy_groups         ! Group index
    double precision, intent(in), dimension(:,:,:) :: &
        source  ! External source
    double precision, intent(inout), dimension(:,:,:) :: &
        phi,    & ! Scalar flux
        psi       ! Angular flux
    double precision, intent(inout), dimension(:,:) :: &
        incident  ! Angular flux incident on the cell
    double precision, allocatable, dimension(:,:,:) :: &
        phi_old   ! Save the previous scalar flux
    double precision, dimension(number_cells, 2 * number_angles, number_energy_groups) :: &
        total_S    ! Sum of all sources
    integer :: &
        outer_count, & ! Counter for the outer loop
        g ! Group index
    double precision :: &
        outer_error  ! Residual error between iterations

    ! Save the scalar flux from previous iteration
    allocate(phi_old(0:number_legendre, number_cells, number_energy_groups))
    phi_old = phi

    total_S = source

    ! Begin loop to converge on the in-scattering source
    do outer_count = 1, max_outer_iters

      ! Loop over the energy groups
      do g = 1, number_energy_groups
        call compute_source(g, number_energy_groups, total_S(:,:,g))

        ! Solve the within group problem
        call wg_solve(g, total_S(:,:,g), phi(:,:,g), psi(:,:,g), incident(:,g))

      end do

      ! Update the error
      outer_error = maxval(abs(phi_old - phi))

      ! save old value of scalar flux
      phi_old = phi

      ! Print output
      if (outer_print) then
        write(*, 1001) outer_count, outer_error
        1001 format ( "  outer: ", i3, " Error: ", f12.9)
      end if

      ! Check if tolerance is reached
      if (outer_error < outer_tolerance) then
        exit
      end if
    end do

    if (outer_count == max_outer_iters) then
      if (.not. ignore_warnings) then
        ! Warning if more iterations are required
        write(*, 1002) outer_count
        1002 format ('inner iteration did not converge in ', i3, ' iterations')
      end if
    end if

  end subroutine mg_solve

  subroutine compute_source(g, nG, source)
    ! ##########################################################################
    ! Get the source including external, fission, and in-scatter
    ! ##########################################################################

    ! Use Statements
    use material, only : number_legendre
    use mesh, only : number_cells
    use angle, only : number_angles, p_leg
    use state, only : d_phi, d_nu_sig_f, d_sig_s, d_keff, d_chi

    ! Variable definitions
    integer, intent(in) :: &
        g,    & ! Group index
        nG      ! Number of energy groups
    double precision, intent(inout), dimension(:,:) :: &
        source  ! Container to hold the computed source
    integer :: &
        c,    & ! Cell index
        gp,   & ! Alternate group index
        a,    & ! Angle index
        l       ! Legendre index

    ! Add the fission source
    do c = 1, number_cells
      source(c,:) = source(c,:) + 0.5 * d_chi(c, g) * dot_product(d_nu_sig_f(c,:), d_phi(0,c,:)) / d_keff
    end do

    ! Add the in-scattering source for each Legendre moment
    do gp = 1, nG
      do a = 1, 2 * number_angles
        do c = 1, number_cells
          do l = 0, number_legendre
            source(c,a) = source(c,a) + 0.5 / (2 * l + 1) * p_leg(l, a) * d_sig_s(l, c, gp, g) * d_phi(l,c,gp)
          end do
        end do
      end do
    end do

  end subroutine compute_source

end module mg_solver
