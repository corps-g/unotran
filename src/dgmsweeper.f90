module dgmsweeper
  ! ##########################################################################
  ! Provide the sweeping through moments for the DGM solver
  ! ##########################################################################

  use control
  use material, only : number_groups, number_legendre
  use mesh, only : dx, number_cells, mMap
  use angle, only : number_angles, p_leg, wt
  use sweeper, only : sweep
  use state, only : d_source, d_nu_sig_f, d_chi, d_sig_s, d_phi, d_delta, &
                    d_sig_t, d_psi, d_keff, d_incoming, normalize_flux
  use dgm, only : number_coarse_groups, expansion_order, &
                  energymesh, basis, compute_xs_moments, compute_flux_moments, &
                  compute_incoming_flux, cumsum, order

  implicit none
  
  contains

  subroutine dgmsweep(phi_new, psi_new)
    ! ##########################################################################
    ! Sweep through the moments for DGM solver
    ! ##########################################################################

    double precision, intent(inout), dimension(:,:,:) :: &
        phi_new, & ! Scalar flux for current iteration
        psi_new    ! Angular flux for current iteration
    double precision, allocatable, dimension(:,:,:) :: &
        phi_m,   & ! Scalar flux moment
        psi_m      ! Angular flux moment
    integer :: &
        i          ! Expansion order index

    ! Initialize the flux containers
    allocate(phi_m(0:number_legendre, number_coarse_groups, number_cells))
    allocate(psi_m(number_coarse_groups, number_angles * 2, number_cells))
    phi_new = 0.0
    psi_new = 0.0
    phi_m = 0.0
    psi_m = 0.0

    ! Get the initial 0th order the flux moments
    call compute_flux_moments()

    ! Sweep through the moments
    do i = 0, expansion_order
      ! Set Incoming to the proper order
      call compute_incoming_flux(order=i)

      ! Compute the cross section moments
      call compute_xs_moments(order=i)

      ! Converge the ith order flux moments
      call inner_solve(i, phi_m, psi_m)

      ! Unfold ith order flux
      call unfold_flux_moments(i, psi_m, phi_new, psi_new)
    end do

    ! Normalize the unfolded fluxes (if eigenvalue problem)
    call normalize_flux(number_groups, phi_new, psi_new)

    deallocate(phi_m, psi_m)

  end subroutine dgmsweep

  subroutine inner_solve(i, phi_m, psi_m)
    ! ##########################################################################
    ! Sweep through the cells, groups, and angles for discrete ordinates
    ! ##########################################################################

    integer, intent(in) :: &
        i              ! Expansion order index
    double precision, intent(inout), dimension(0:,:,:) :: &
        phi_m          ! Scalar flux moments
    double precision, intent(inout), dimension(:,:,:) :: &
        psi_m          ! Angular flux moments
    double precision :: &
        inner_error, & ! Error between successive iterations for moment i
        frac           ! Fraction for computing the eigenvalue
    integer :: &
        counter        ! Iteration counter

    ! Initialize iteration variables
    inner_error = 1.0
    counter = 1

    ! Interate to convergance tolerance
    do while (inner_error > inner_tolerance)
      ! Use discrete ordinates to sweep over the moment equation
      call sweep(number_coarse_groups, phi_m, psi_m)
      ! Update the 0th order moments if working on converging zeroth moment
      if (i == 0) then
        if (solver_type == 'eigen') then
          frac = sum(abs(phi_m(0,:,:))) / sum(abs(d_phi(0,:,:)))
          d_keff = d_keff * frac
          phi_m = phi_m / frac
          psi_m = psi_m / frac
        end if

        ! error is the difference in the norm of phi for successive iterations
        inner_error = sum(abs(d_phi - phi_m))

        ! output the current error and iteration number
        if (inner_print) then
          print *, '    ', 'eps = ', inner_error, ' counter = ', counter, &
                   ' order = ', i, phi_m(0,:,1)
        end if

        !call normalize_flux(number_coarse_groups, d_phi, d_psi)

        ! increment the iteration
        counter = counter + 1

        ! Break out of loop if exceeding maximum inner iterations
        if (counter > max_inner_iters) then
          if (.not. ignore_warnings) then
            print *, 'warning: exceeded maximum inner iterations'
          end if
          exit
        end if
      else
        ! Higher orders converge in a single pass
        exit
      end if

    end do

  end subroutine inner_solve

  ! Unfold the flux moments
  subroutine unfold_flux_moments(ord, psi_moment, phi_new, psi_new)
    ! ##########################################################################
    ! Compute the fluxes from the moments
    ! ##########################################################################

    integer, intent(in) :: &
        ord           ! Expansion order
    double precision, intent(in), dimension(:,:,:) :: &
        psi_moment    ! Angular flux moments
    double precision, intent(inout), dimension(:,:,:) :: &
        psi_new,    & ! Scalar flux for current iteration
        phi_new       ! Angular flux for current iteration
    double precision, allocatable, dimension(:) :: &
        M                    ! Legendre polynomial integration vector
    integer :: &
        limit,      & ! Limiting expansion order for basis
        o,          & ! Octant index
        a,          & ! Angle index
        c,          & ! Cell index
        cg,         & ! Coarse group index
        g,          & ! Fine group index
        gp,         & ! Alternate fine group index
        an,         & ! Global angle index
        cmin,       & ! Lower cell number
        cmax,       & ! Upper cell number
        cstep,      & ! Cell stepping direction
        amin,       & ! Lower angle number
        amax,       & ! Upper angle number
        astep         ! Angle stepping direction
    logical :: &
        octant        ! Positive/Negative octant flag
    double precision :: &
        val           ! Variable to hold a double value

    allocate(M(0:number_legendre))

    ! Recover the angular flux from moments
    do c = 1, number_cells
      do a = 1, number_angles * 2
        ! legendre polynomial integration vector
        an = merge(a, 2 * number_angles - a + 1, a <= number_angles)
        M = wt(an) * p_leg(:,a)
        do cg = 1, number_coarse_groups
          g = ord + cumsum(cg)
          ! Get the limiting order with respect to the coarse group
          if (cg == number_coarse_groups) then
            limit = number_groups + 1
          else
            limit = cumsum(cg + 1)
          end if
          ! Check if there is a basis for this order
          if (g < limit) then
            ! Loop over the fine groups within the coarse group
            do gp = cumsum(cg), order(cg) + cumsum(cg)
              val = basis(ord + cumsum(cg), gp - cumsum(cg)) * psi_moment(cg, a, c)
              psi_new(gp, a, c) = psi_new(gp, a, c) + val
              phi_new(:, gp, c) = phi_new(:, gp, c) + M(:) * val
            end do
          end if
        end do
      end do
    end do

    deallocate(M)

  end subroutine unfold_flux_moments
  
end module dgmsweeper
