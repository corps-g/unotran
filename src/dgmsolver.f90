module dgmsolver
  ! ############################################################################
  ! Solve discrete ordinates using the Discrete Generalized Multigroup Method
  ! ############################################################################

  implicit none
  
  contains
  
  subroutine initialize_dgmsolver()
    ! ##########################################################################
    ! Initialize all of the variables and solvers
    ! ##########################################################################

    ! Use Statements
    use state, only : initialize_state

    ! allocate the solutions variables
    call initialize_state()
    call compute_source_moments()

  end subroutine initialize_dgmsolver

  subroutine dgmsolve(bypass_arg)
    ! ##########################################################################
    ! Interate DGM equations to convergance
    ! ##########################################################################

    ! Use Statements
    use control, only : max_recon_iters, recon_print, recon_tolerance, store_psi, &
                        ignore_warnings, lamb, number_cells, number_fine_groups, &
                        number_legendre, number_angles, solver_type
    use state, only : d_keff, phi, psi, d_phi, d_psi, normalize_flux, norm_frac, &
                      update_fission_density, output_moments
    use dgm, only : expansion_order, phi_m_zero, psi_m_zero
    use solver, only : solve

    ! Variable definitions
    logical, intent(in), optional :: &
        bypass_arg        ! Allow the dgmsolver to do one recon loop selectively
    logical, parameter :: &
        bypass_default = .false.  ! Default value of bypass_arg
    logical :: &
        bypass_flag       ! Local variable to signal an eigen loop bypass
    integer :: &
        recon_count,    & ! Iteration counter
        i                 ! Expansion index
    double precision :: &
        recon_error       ! Error between successive iterations
    double precision, dimension(0:number_legendre, number_cells, number_fine_groups) :: &
        old_phi        ! Scalar flux from previous iteration
    double precision, dimension(number_cells, 2 * number_angles, number_fine_groups) :: &
        old_psi        ! Angular flux from previous iteration

    if (present(bypass_arg)) then
      bypass_flag = bypass_arg
    else
      bypass_flag = bypass_default
    end if

    do recon_count = 1, max_recon_iters
      ! Save the old value of the scalar flux
      old_phi = phi
      old_psi = psi

      ! Expand the fluxes into moment form
      call compute_flux_moments()

      ! Compute the cross section moments
      call compute_xs_moments()

      ! Fill the initial moments
      d_phi = phi_m_zero
      d_psi = psi_m_zero
      phi = 0.0
      psi = 0.0

      ! Solve for order 0
      do i = 0, expansion_order

        ! Set Incoming to the proper order
        call compute_incoming_flux(i, old_psi)

        ! Compute the cross section moments
        call slice_xs_moments(order=i)

        ! Converge the 0th order flux moments
        if (i == 0) then
          call solve()

          ! Save the new flux moments as the zeroth moments
          phi_m_zero = d_phi
          psi_m_zero = d_psi
        ! Converge the higher order moments
        else
          call solve(.true.)
        end if

        ! Print the moments if verbose printing is on
        if (recon_print > 1) then
          print *, i, d_phi
        end if

        ! Unfold ith order flux
        call unfold_flux_moments(i, d_psi, phi, psi)
      end do

      ! Update flux using krasnoselskii iteration
      phi = (1.0 - lamb) * old_phi + lamb * phi
      if (store_psi) then
        psi = (1.0 - lamb) * old_psi + lamb * psi
      end if

      call normalize_flux(phi, psi)

      ! Compute the fission density
      call update_fission_density()

      ! Update the error
      recon_error = sum(abs(old_phi - phi))

      ! Print output
      if (recon_print > 0) then
        write(*, 1001) recon_count, recon_error, d_keff
        1001 format ( "recon: ", i4, " Error: ", es12.5E2, " eigenvalue: ", f12.9)
        if (recon_print > 1) then
          !call output_moments()
        end if
      end if

      ! Check if tolerance is reached
      if (recon_error < recon_tolerance .or. bypass_flag) then
        exit
      end if

    end do

    if (recon_count == max_recon_iters) then
      if (.not. ignore_warnings) then
        ! Warning if more iterations are required
        write(*, 1002) recon_count
        1002 format ('recon iteration did not converge in ', i4, ' iterations')
      end if
    end if

  end subroutine dgmsolve

  subroutine unfold_flux_moments(order, psi_moment, phi_new, psi_new)
    ! ##########################################################################
    ! Unfold the fluxes from the moments
    ! ##########################################################################

    ! Use Statements
    use control, only : number_fine_groups, number_angles, number_cells
    use angle, only : p_leg, wt
    use control, only : store_psi, number_angles, number_legendre
    use dgm, only : energyMesh, basis

    ! Variable definitions
    integer, intent(in) :: &
        order         ! Expansion order
    double precision, intent(in), dimension(:,:,:) :: &
        psi_moment    ! Angular flux moments
    double precision, intent(inout), dimension(:,:,:) :: &
        psi_new,    & ! Scalar flux for current iteration
        phi_new       ! Angular flux for current iteration
    double precision, allocatable, dimension(:) :: &
        M                    ! Legendre polynomial integration vector
    integer :: &
        a,          & ! Angle index
        c,          & ! Cell index
        cg,         & ! Coarse group index
        g,          & ! Fine group index
        an            ! Global angle index
    double precision :: &
        val           ! Variable to hold a double value

    allocate(M(number_legendre))

    ! Recover the angular flux from moments
    do g = 1, number_fine_groups
      ! Get the coarse group index
      cg = energyMesh(g)
      do a = 1, number_angles * 2
        ! legendre polynomial integration vector
        an = merge(a, 2 * number_angles - a + 1, a <= number_angles)
        M = wt(an) * p_leg(:,a)
        do c = 1, number_cells
          ! Unfold the moments
          val = basis(g, order) * psi_moment(c, a, cg)
          if (store_psi) then
            psi_new(c, a, g) = psi_new(c, a, g) + val
          end if
          phi_new(:, c, g) = phi_new(:, c, g) + M(:) * val
        end do
      end do
    end do

    deallocate(M)

  end subroutine unfold_flux_moments

  subroutine dgmoutput()
    ! ##########################################################################
    ! Save the output for the fluxes to file
    ! ##########################################################################

    ! Use Statements
    use state, only : phi, psi, output_state

    print *, phi
    print *, psi
    call output_state()

  end subroutine dgmoutput

  subroutine finalize_dgmsolver()
    ! ##########################################################################
    ! Deallocate used arrays
    ! ##########################################################################

    ! Use Statements
    use state, only : finalize_state

    call finalize_state()

  end subroutine finalize_dgmsolver

  subroutine compute_flux_moments()
    ! ##########################################################################
    ! Expand the flux moments using the basis functions
    ! ##########################################################################

    ! Use Statements
    use control, only : number_angles, number_fine_groups, number_cells
    use state, only : phi, psi
    use mesh, only : mMap
    use dgm, only : phi_m_zero, psi_m_zero, basis, energyMesh

    ! Variable definitions
    integer :: &
        a,   & ! Angle index
        c,   & ! Cell index
        cg,  & ! Outer coarse group index
        g,   & ! Outer fine group index
        mat    ! Material index

    ! initialize all moments to zero
    phi_m_zero = 0.0
    psi_m_zero = 0.0

    ! Get moments for the fluxes
    do g = 1, number_fine_groups
      cg = energyMesh(g)
      do a = 1, number_angles * 2
        do c = 1, number_cells
          ! Scalar flux
          if (a == 1) then
            phi_m_zero(:, c, cg) = phi_m_zero(:, c, cg) + basis(g, 0) * phi(:, c, g)
          end if
          ! Angular flux
          psi_m_zero(c, a, cg) = psi_m_zero(c, a, cg) +  basis(g, 0) * psi(c, a, g)
        end do
      end do
    end do

  end subroutine compute_flux_moments

  subroutine compute_incoming_flux(order, psi)
    ! ##########################################################################
    ! Compute the incident angular flux at the boundary for the given order
    ! ##########################################################################

    ! Use Statements
    use control, only : number_angles, number_fine_groups
    use state, only : d_incoming
    use dgm, only : basis, energyMesh

    ! Variable definitions
    double precision, intent(in), dimension(:,:,:) :: &
      psi      ! Angular flux
    integer :: &
      order, & ! Expansion order
      a,     & ! Angle index
      g,     & ! Fine group index
      cg       ! Coarse group index

    d_incoming = 0.0
    do g = 1, number_fine_groups
      cg = energyMesh(g)
      do a = 1, number_angles
        d_incoming(a, cg) = d_incoming(a, cg) + basis(g, order) * psi(1, a + number_angles, g)
      end do
    end do

  end subroutine compute_incoming_flux

  subroutine slice_xs_moments(order)
    ! ##########################################################################
    ! Fill the XS containers with precomputed values for higher moments
    ! ##########################################################################

    ! Use Statements
    use state, only : d_source, d_chi, d_sig_s, d_nu_sig_f, d_keff
    use angle, only : p_leg, wt
    use dgm, only : source_m, delta_m, chi_m, sig_s_m, phi_m_zero, psi_m_zero
    use control, only : number_legendre, number_angles, number_cells, number_groups, &
                        allow_fission

    ! Variable definitions
    integer, intent(in) :: &
        order  ! Expansion order
    integer :: &
        a, & ! Angle index
        c, & ! Cell index
        cg, & ! Coarse group index
        cgp, & ! Alternate coarse group index
        l      ! Legendre index

    ! Get the external and delta sources
    d_source(:, :, :) = source_m(:, :, :, order) - delta_m(:, :, :, order) * psi_m_zero(:, :, :)

    if (order > 0) then
      ! Get the combined fixed source
      do cg = 1, number_groups
        do cgp = 1, number_groups
          do a = 1, number_angles * 2
            do c = 1, number_cells
              if (allow_fission) then
                d_source(c, a, cg) = d_source(c, a, cg) + &
                                     0.5 * chi_m(c, cg, order) * d_nu_sig_f(c, cgp) * phi_m_zero(0, c, cgp) / d_keff
              end if
              do l = 0, number_legendre
                ! Get the scattering source
                d_source(c, a, cg) = d_source(c, a, cg) + &
                                     0.5 / (2 * l + 1) * p_leg(l, a) * sig_s_m(l, c, cgp, cg, order) * phi_m_zero(l, c, cgp)
              end do
            end do
          end do
        end do
      end do

      d_chi = 0.0
      d_sig_s = 0.0
    else
      ! Compute the source as normal
      d_chi(:, :) = chi_m(:, :, order)
      d_sig_s(:,:,:,:) = sig_s_m(:, :, :, :, order)
    end if

  end subroutine slice_xs_moments

  subroutine compute_xs_moments()
    ! ##########################################################################
    ! Expand the cross section moments using the basis functions
    ! ##########################################################################

    ! Use Statements
    use control, only : number_angles, number_fine_groups, number_cells, number_legendre, number_groups, &
                        ignore_warnings
    use state, only : d_sig_t, d_nu_sig_f, phi, psi
    use material, only : sig_t, nu_sig_f, sig_s
    use mesh, only : mMap
    use dgm, only : source_m, chi_m, phi_m_zero, psi_m_zero, energyMesh, basis, &
                    sig_s_m, delta_m, expansion_order

    ! Variable definitions
    integer :: &
        o,   & ! Order index
        a,   & ! Angle index
        c,   & ! Cell index
        cg,  & ! Outer coarse group index
        cgp, & ! Inner coarse group index
        g,   & ! Outer fine group index
        gp,  & ! Inner fine group index
        l,   & ! Legendre moment index
        mat    ! Material index

    if (allocated(delta_m)) then
      deallocate(delta_m)
    end if
    if (allocated(sig_s_m)) then
      deallocate(sig_s_m)
    end if

    allocate(delta_m(number_cells, 2 * number_angles, number_groups, 0:expansion_order))
    allocate(sig_s_m(0:number_legendre, number_cells, number_groups, number_groups, 0:expansion_order))

    ! initialize all moments to zero
    sig_s_m = 0.0
    d_sig_t = 0.0
    delta_m = 0.0
    d_nu_sig_f = 0.0

    do o = 0, expansion_order
      do g = 1, number_fine_groups
        cg = energyMesh(g)
        if (o == 0) then
          do c = 1, number_cells
            ! get the material for the current cell
            mat = mMap(c)
            ! Check if producing nan and not computing with a nan
            if (phi_m_zero(0, c, cg) /= phi_m_zero(0, c, cg)) then
              ! Detected NaN
              if (.not. ignore_warnings) then
                print *, "NaN detected, limiting"
              end if
              phi_m_zero(0, c, cg) = 1.0
            else if (phi_m_zero(0, c, cg) /= 0.0)  then
              ! total cross section moment
              d_sig_t(c, cg) = d_sig_t(c, cg) + basis(g, 0) * sig_t(g, mat) * phi(0, c, g) / phi_m_zero(0, c, cg)
              ! fission cross section moment
              d_nu_sig_f(c, cg) = d_nu_sig_f(c, cg) + nu_sig_f(g, mat) * phi(0, c, g) / phi_m_zero(0, c, cg)
            end if
          end do
        end if

        ! Scattering cross section moment
        do gp = 1, number_fine_groups
          cgp = energyMesh(gp)
          do c = 1, number_cells
            ! get the material for the current cell
            mat = mMap(c)
            do l = 0, number_legendre
              ! Check if producing nan
              if (phi_m_zero(l, c, cgp) /= phi_m_zero(l, c, cgp)) then
                ! Detected NaN
                if (.not. ignore_warnings) then
                  print *, "NaN detected, limiting"
                end if
                phi_m_zero(l, c, cgp) = 1.0
              else if (phi_m_zero(l, c, cgp) /= 0.0) then
                sig_s_m(l, c, cgp, cg, o) = sig_s_m(l, c, cgp, cg, o) &
                                       + basis(g, o) * sig_s(l, gp, g, mat) * phi(l, c, gp) / phi_m_zero(l, c, cgp)
              end if
            end do
          end do
        end do
      end do

      do g = 1, number_fine_groups
        cg = energyMesh(g)
        ! Add angular total cross section moment (delta) to the external source
        do a = 1, number_angles * 2
          do c = 1, number_cells
            ! get the material for the current cell
            mat = mMap(c)
            ! Check if producing nan and not computing with a nan
            if (psi_m_zero(c, a, cg) /= psi_m_zero(c, a, cg)) then
              ! Detected NaN
                if (.not. ignore_warnings) then
                  print *, "NaN detected, limiting"
                end if
                psi_m_zero(c, a, cg) = 1.0
            else if (psi_m_zero(c, a, cg) /= 0.0) then
              delta_m(c, a, cg, o) = delta_m(c, a, cg, o) + basis(g, o) * (sig_t(g, mat) &
                                  - d_sig_t(c, cg)) * psi(c, a, g) / psi_m_zero(c, a, cg)
            end if
          end do
        end do
      end do
    end do

  end subroutine compute_xs_moments

  subroutine compute_source_moments()
    ! ##########################################################################
    ! Expand the source and chi using the basis functions
    ! ##########################################################################

    ! Use Statements
    use control, only : number_cells, number_fine_groups, number_angles, number_groups
    use material, only : chi
    use state, only : source
    use mesh, only : mMap
    use dgm, only : chi_m, source_m, expansion_order, energyMesh, basis

    ! Variable definitions
    integer :: &
        order, & ! Expansion order index
        c,     & ! Cell index
        a,     & ! Angle index
        g,     & ! Fine group index
        cg,    & ! Coarse group index
        mat      ! Material index

    allocate(chi_m(number_cells, number_groups, 0:expansion_order))
    allocate(source_m(number_cells, number_angles * 2, number_groups, 0:expansion_order))

    chi_m = 0.0
    source_m = 0.0

    do order = 0, expansion_order
      do g = 1, number_fine_groups
        cg = energyMesh(g)
        do a = 1, number_angles * 2
          do c = 1, number_cells
            mat = mMap(c)
            if (a == 1) then
              ! chi moment
              chi_m(c, cg, order) = chi_m(c, cg, order) + basis(g, order) * chi(g, mat)
            end if

            ! Source moment
            source_m(c, a, cg, order) = source_m(c, a, cg, order) + basis(g, order) * source(c, a, g)
          end do
        end do
      end do
    end do

  end subroutine compute_source_moments

end module dgmsolver
