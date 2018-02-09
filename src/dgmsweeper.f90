module dgmsweeper
  ! ##########################################################################
  ! Provide the sweeping through moments for the DGM solver
  ! ##########################################################################

  use control
  use material, only : number_groups, number_legendre
  use mesh, only : dx, number_cells, mMap
  use angle, only : number_angles
  use sweeper, only : sweep
  use state, only : d_source, d_nu_sig_f, d_chi, d_sig_s, d_phi, d_delta, &
                    d_sig_t, d_psi, d_keff, d_incoming
  use dgm, only : number_coarse_groups, expansion_order, &
                  energymesh, basis, compute_xs_moments, compute_flux_moments, &
                  compute_incoming_flux

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
      call unfold_flux_moments(i, phi_m, psi_m, &
                               phi_new, psi_new)
    end do

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
        frac           ! Normalization fraction
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
        ! error is the difference in the norm of phi for successive iterations
        inner_error = sum(abs(d_phi - phi_m))

        ! output the current error and iteration number
        if (inner_print) then
          print *, '    ', 'eps = ', inner_error, ' counter = ', counter, &
                   ' order = ', i, phi_m(0,:,1)
        end if

        ! increment the iteration
        counter = counter + 1

        if (solver_type == 'eigen') then
          frac = sum(abs(phi_m(0,:,:))) / sum(abs(d_phi(0,:,:)))
          d_keff = d_keff * frac
          !phi_m = phi_m * frac
        end if

        !d_phi = (1.0 - lamb) * d_phi + lamb * phi_m
        !d_psi = (1.0 - lamb) * d_psi + lamb * psi_m
        d_phi = phi_m
        d_psi = psi_m

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

      ! Normalize the unfolded fluxes (if eigenvalue problem)
      !call normalize_flux(d_phi, d_psi, incoming)

    end do

  end subroutine inner_solve

  ! Unfold the flux moments
  subroutine unfold_flux_moments(order, phi_moment, psi_moment, &
                                 phi_new, psi_new)
    ! ##########################################################################
    ! Compute the fluxes from the moments
    ! ##########################################################################

    integer, intent(in) :: &
        order         ! Expansion order
    double precision, intent(in), dimension(:,:,:) :: &
        phi_moment, & ! Scalar flux moments
        psi_moment    ! Angular flux moments
    double precision, intent(out), dimension(:,:,:) :: &
        psi_new,    & ! Scalar flux for current iteration
        phi_new       ! Angular flux for current iteration
    integer :: &
        a,          & ! Angle index
        c,          & ! Cell index
        cg,         & ! Coarse group index
        g,          & ! Fine group index
        mat           ! Material index

    do c = 1, number_cells
      do a = 1, number_angles * 2
        do g = 1, number_groups
          cg = energyMesh(g)
          ! Scalar flux
          if (a == 1) then
            phi_new(:, g, c) = phi_new(:, g, c) &
                               + basis(g, order) * phi_moment(:, cg, c)
          end if
          ! Angular flux
          psi_new(g, a, c) = psi_new(g, a, c) &
                             + basis(g, order) * psi_moment(cg, a, c)
        end do
      end do
    end do

    ! Normalize the unfolded fluxes (if eigenvalue problem)
    call normalize_flux(phi_new, psi_new)

  end subroutine unfold_flux_moments
  
  subroutine normalize_flux(phi, psi)
    ! ##########################################################################
    ! Normalize the flux for the eigenvalue problem
    ! ##########################################################################

    double precision, intent(inout), dimension(:,:,:) :: &
        phi, &   ! Scalar flux
        psi      ! Angular flux
    double precision :: &
        frac     ! Normalization fraction

    if (solver_type == 'eigen') then
      frac = sum(abs(phi(1,:,:))) / (number_cells * number_groups)

      ! normalize phi
      phi = phi / frac

      ! normalize psi
      psi = psi / frac
    end if

  end subroutine normalize_flux


end module dgmsweeper
