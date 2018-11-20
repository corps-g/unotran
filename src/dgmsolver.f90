module dgmsolver
  ! ############################################################################
  ! Solve discrete ordinates using the Discrete Generalized Multigroup Method
  ! ############################################################################

  implicit none

  contains
  
  subroutine initialize_dgmsolver()
    ! ##########################################################################
    ! Initialize all of the variables and solvers
    ! The mg containers in state are set to coarse group size
    ! ##########################################################################

    ! Use Statements
    use state, only : initialize_state
    use state, only : mg_mMap
    use control, only : homogenization_map

    ! allocate the solutions variables
    call initialize_state()

    ! Fill the multigroup material map
    mg_mMap = homogenization_map

    call compute_source_moments()

  end subroutine initialize_dgmsolver

  subroutine dgmsolve(bypass_arg)
    ! ##########################################################################
    ! Interate DGM equations to convergance
    ! ##########################################################################

    ! Use Statements
    use control, only : max_recon_iters, recon_print, recon_tolerance, store_psi, &
                        ignore_warnings, lamb, number_cells, number_fine_groups, &
                        number_legendre, number_angles, min_recon_iters
    use state, only : keff, phi, psi, mg_phi, mg_psi, normalize_flux, &
                      update_fission_density, output_moments
    use dgm, only : expansion_order, phi_m_zero, psi_m_zero, dgm_order
    use solver, only : solve

    ! Variable definitions
    logical, intent(in), optional :: &
        bypass_arg        ! Allow the dgmsolver to do one recon loop selectively
    logical, parameter :: &
        bypass_default = .false.  ! Default value of bypass_arg
    logical :: &
        bypass_flag       ! Local variable to signal an eigen loop bypass
    integer :: &
        recon_count       ! Iteration counter
    double precision :: &
        recon_error       ! Error between successive iterations
    double precision, dimension(0:number_legendre, number_fine_groups, number_cells) :: &
        old_phi           ! Scalar flux from previous iteration
    double precision, dimension(number_fine_groups, 2 * number_angles, number_cells) :: &
        old_psi           ! Angular flux from previous iteration

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
      mg_phi = phi_m_zero
      mg_psi = psi_m_zero
      phi = 0.0
      psi = 0.0

      ! Solve for order 0
      do dgm_order = 0, expansion_order

        ! Set Incoming to the proper order
        call compute_incoming_flux(dgm_order, old_psi)

        ! Compute the cross section moments
        call slice_xs_moments(order=dgm_order)

        call solve()

        ! Converge the 0th order flux moments
        if (dgm_order == 0) then
          ! Save the new flux moments as the zeroth moments
          phi_m_zero = mg_phi
          psi_m_zero = mg_psi
        end if

        ! Print the moments if verbose printing is on
        if (recon_print > 1) then
          print *, dgm_order, mg_phi
        end if

        ! Unfold ith order flux
        call unfold_flux_moments(dgm_order, mg_psi, phi, psi)
      end do

      ! Update flux using krasnoselskii iteration
      phi = (1.0 - lamb) * old_phi + lamb * phi
      if (store_psi) then
        psi = (1.0 - lamb) * old_psi + lamb * psi
      end if

      ! Compute the fission density
      call update_fission_density()

      ! Update the error
      recon_error = maxval(abs(old_phi - phi))

      ! Print output
      if (recon_print > 0) then
        write(*, 1001) recon_count, recon_error, keff
        1001 format ( "recon: ", i4, " Error: ", es12.5E2, " eigenvalue: ", f12.9)
        if (recon_print > 1) then
          call output_moments()
        end if
      end if

      ! Check if tolerance is reached
      if ((recon_error < recon_tolerance .and. recon_count >= min_recon_iters) .or. bypass_flag) then
        exit
      end if

    end do

    ! Do final normalization
    call normalize_flux(phi, psi)

    ! Compute the fission density based on the fine-group flux
    call update_fission_density(.true.)

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
    use control, only : number_fine_groups, number_angles, number_cells, energy_group_map
    use angle, only : p_leg, wt
    use control, only : store_psi, number_angles, number_legendre
    use dgm, only : basis

    ! Variable definitions
    integer, intent(in) :: &
        order         ! Expansion order
    double precision, intent(in), dimension(:,:,:) :: &
        psi_moment    ! Angular flux moments
    double precision, intent(inout), dimension(:,:,:) :: &
        psi_new,    & ! Scalar flux for current iteration
        phi_new       ! Angular flux for current iteration
    double precision, dimension(0:number_legendre) :: &
        M                    ! Legendre polynomial integration vector
    integer :: &
        a,          & ! Angle index
        c,          & ! Cell index
        g,          & ! Fine group index
        an            ! Global angle index
    double precision :: &
        val           ! Variable to hold a double value

    ! Recover the angular flux from moments
    do c = 1, number_cells
      do a = 1, number_angles * 2
        ! legendre polynomial integration vector
        an = merge(a, 2 * number_angles - a + 1, a <= number_angles)
        M = wt(an) * p_leg(:,a)
        do g = 1, number_fine_groups
          ! Unfold the moments
          val = basis(g, order) * psi_moment(energy_group_map(g), a, c)
          if (store_psi) then
            psi_new(g, a, c) = psi_new(g, a, c) + val
          end if
          phi_new(:, g, c) = phi_new(:, g, c) + M(:) * val
        end do
      end do
    end do

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
    use control, only : number_angles, number_fine_groups, number_cells, &
                        energy_group_map, delta_legendre_order, truncate_delta
    use state, only : phi, psi
    use dgm, only : phi_m_zero, psi_m_zero, basis
    use angle, only : p_leg

    ! Variable definitions
    integer :: &
        a,    & ! Angle index
        c,    & ! Cell index
        cg,   & ! Outer coarse group index
        ord,  & ! Delta truncation order
        g       ! Outer fine group index
    double precision :: &
        tmp_psi ! temporary angular


    ! initialize all moments to zero
    phi_m_zero = 0.0
    psi_m_zero = 0.0

    ! Get moments for the Angular flux
    do c = 1, number_cells
      do a = 1, number_angles * 2
        do g = 1, number_fine_groups
          cg = energy_group_map(g)

          if (truncate_delta) then
            ! If we are truncating the delta term, then first truncate
            ! the angular flux (because the idea is that we would only store
            ! the angular moments and then the discrete delta term would be
            ! generated on the fly from the corresponding delta moments)
            ord = delta_legendre_order
            tmp_psi = dot_product(p_leg(:ord, a), phi(:ord, g, c))
          else
            tmp_psi = psi(g, a, c)
          end if

          psi_m_zero(cg, a, c) = psi_m_zero(cg, a, c) + basis(g, 0) * tmp_psi
        end do
      end do
    end do

    !TODO: Integrate psi_m_zero over angle to get phi_m_zero

    ! Get moments for the Scalar flux
    do c = 1, number_cells
      do g = 1, number_fine_groups
        cg = energy_group_map(g)
        phi_m_zero(:, cg, c) = phi_m_zero(:, cg, c) + basis(g, 0) * phi(:, g, c)
      end do
    end do

  end subroutine compute_flux_moments

  subroutine compute_incoming_flux(order, psi)
    ! ##########################################################################
    ! Compute the incident angular flux at the boundary for the given order
    ! ##########################################################################

    ! Use Statements
    use control, only : number_angles, number_fine_groups, energy_group_map
    use state, only : mg_incident
    use dgm, only : basis

    ! Variable definitions
    double precision, intent(in), dimension(:,:,:) :: &
      psi      ! Angular flux
    integer :: &
      order, & ! Expansion order
      a,     & ! Angle index
      g,     & ! Fine group index
      cg       ! Coarse group index

    mg_incident = 0.0
    do a = 1, number_angles
      do g = 1, number_fine_groups
        cg = energy_group_map(g)
        mg_incident(cg, a) = mg_incident(cg, a) + basis(g, order) * psi(g, a + number_angles, 1)
      end do
    end do

  end subroutine compute_incoming_flux

  subroutine slice_xs_moments(order)
    ! ##########################################################################
    ! Fill the XS containers with precomputed values for higher moments
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_chi, mg_sig_s
    use dgm, only : chi_m, sig_s_m

    ! Variable definitions
    integer, intent(in) :: &
        order  ! Expansion order

    mg_chi(:, :) = chi_m(:, :, order)
    mg_sig_s(:,:,:,:) = sig_s_m(:, :, :, :, order)

  end subroutine slice_xs_moments

  subroutine compute_xs_moments()
    ! ##########################################################################
    ! Expand the cross section moments using the basis functions
    ! ##########################################################################

    ! Use Statements
    use control, only : number_angles, number_fine_groups, number_cells, &
                        number_legendre, number_groups, &
                        delta_legendre_order, truncate_delta, number_regions, &
                        energy_group_map, scatter_legendre_order
    use state, only : mg_sig_t, mg_nu_sig_f, phi, psi, mg_mMap
    use material, only : sig_t, nu_sig_f, sig_s
    use mesh, only : mMap, dx
    use dgm, only : phi_m_zero, psi_m_zero, basis, sig_s_m, delta_m, expansion_order
    use angle, only : p_leg

    ! Variable definitions
    integer :: &
        o,           & ! Order index
        a,           & ! Angle index
        c,           & ! Cell index
        cg,          & ! Outer coarse group index
        cgp,         & ! Inner coarse group index
        g,           & ! Outer fine group index
        gp,          & ! Inner fine group index
        l,           & ! Legendre moment index
        r,           & ! Region index
        ord,         & ! Delta truncation order
        mat            ! Material index
    double precision :: &
        tmp_psi,     & ! temporary angular
        tmp_psi_m_zero ! flux arrays
    double precision, dimension(0:number_legendre, number_groups, number_regions) :: &
        homog_phi      ! Homogenization container

    if (allocated(delta_m)) then
      deallocate(delta_m)
    end if
    if (allocated(sig_s_m)) then
      deallocate(sig_s_m)
    end if

    allocate(delta_m(number_groups, 2 * number_angles, number_regions, 0:expansion_order))
    allocate(sig_s_m(0:scatter_legendre_order, number_groups, number_groups, number_regions, 0:expansion_order))

    ! initialize all moments and mg containers to zero
    sig_s_m = 0.0
    mg_sig_t = 0.0
    delta_m = 0.0
    mg_nu_sig_f = 0.0

    ! Compute the denominator for spatial homogenization
    homog_phi = 0.0
    do c = 1, number_cells
      r = mg_mMap(c)
      homog_phi(0:, :, r) = homog_phi(0:, :, r) + dx(c) * phi_m_zero(0:, :, c)
    end do

    ! Compute the total and fission moments
    do c = 1, number_cells
      do g = 1, number_fine_groups
        cg = energy_group_map(g)
        ! get the material for the current cell
        mat = mMap(c)
        r = mg_mMap(c)
        ! Check if producing nan and not computing with a nan
        if (phi_m_zero(0, cg, c) /= 0.0)  then
          ! total cross section moment
          mg_sig_t(cg, r) = mg_sig_t(cg, r) + dx(c) * basis(g, 0) * sig_t(g, mat) * phi(0, g, c) / homog_phi(0, cg, r)
          ! fission cross section moment
          mg_nu_sig_f(cg, r) = mg_nu_sig_f(cg, r) + dx(c) * nu_sig_f(g, mat) * phi(0, g, c) / homog_phi(0, cg, r)
        end if
      end do
    end do

    do o = 0, expansion_order
      do c = 1, number_cells
        do g = 1, number_fine_groups
          cg = energy_group_map(g)

          ! Scattering cross section moment
          do gp = 1, number_fine_groups
            cgp = energy_group_map(gp)
            ! get the material for the current cell
            mat = mMap(c)
            r = mg_mMap(c)
            do l = 0, scatter_legendre_order
              ! Check if producing nan
              if (phi_m_zero(l, cgp, c) /= 0.0) then
                sig_s_m(l, cgp, cg, r, o) = sig_s_m(l, cgp, cg, r, o) &
                                       + dx(c) * basis(g, o) * sig_s(l, gp, g, mat) * phi(l, gp, c) / homog_phi(l, cgp, r)
              end if
            end do
          end do
        end do
      end do

      ord = delta_legendre_order
      ! Add angular total cross section moment (delta) to the external source
      do c = 1, number_cells
        do a = 1, number_angles * 2
          do g = 1, number_fine_groups
            cg = energy_group_map(g)
            ! get the material for the current cell
            mat = mMap(c)
            r = mg_mMap(c)
            if (truncate_delta) then
              ! If we are truncating the delta term, then first truncate
              ! the angular flux (because the idea is that we would only store
              ! the angular moments and then the discrete delta term would be
              ! generated on the fly from the corresponding delta moments)
              tmp_psi = dot_product(p_leg(:ord, a), phi(:ord, g, c))
              tmp_psi_m_zero = dot_product(p_leg(:ord, a), phi_m_zero(:ord, cg, c))
            else
              tmp_psi = psi(g, a, c)
              tmp_psi_m_zero = psi_m_zero(cg, a, c)
            end if
            ! Check if producing nan and not computing with a nan
            if (tmp_psi_m_zero /= 0.0) then
              delta_m(cg, a, r, o) = delta_m(cg, a, r, o) + dx(c) * basis(g, o) * (sig_t(g, mat) &
                                  - mg_sig_t(cg, r)) * tmp_psi / tmp_psi_m_zero &
                                  * phi_m_zero(0, cg, c) / homog_phi(0, cg, r)
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
    use control, only : number_cells, number_fine_groups, &
                        number_groups, energy_group_map
    use material, only : chi
    use state, only : mg_constant_source
    use mesh, only : mMap
    use dgm, only : chi_m, source_m, expansion_order, basis

    ! Variable definitions
    integer :: &
        order, & ! Expansion order index
        c,     & ! Cell index
        g,     & ! Fine group index
        cg,    & ! Coarse group index
        mat      ! Material index

    allocate(chi_m(number_groups, number_cells, 0:expansion_order))
    allocate(source_m(number_groups, 0:expansion_order))

    chi_m = 0.0
    source_m = 0.0

    ! chi moment
    do order = 0, expansion_order
      do c = 1, number_cells
        mat = mMap(c)
        do g = 1, number_fine_groups
          cg = energy_group_map(g)
          chi_m(cg, c, order) = chi_m(cg, c, order) + basis(g, order) * chi(g, mat)
        end do
      end do
    end do

    ! Source moment
    do order = 0, expansion_order
      do g = 1, number_fine_groups
        cg = energy_group_map(g)
        source_m(cg, order) = source_m(cg, order) + basis(g, order) * mg_constant_source
      end do
    end do

  end subroutine compute_source_moments

end module dgmsolver
