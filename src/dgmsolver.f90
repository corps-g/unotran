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
                        ignore_warnings, lamb, number_cells, &
                        number_legendre, number_angles, min_recon_iters, number_coarse_groups, &
                        truncate_delta, delta_legendre_order
    use state, only : keff, phi, psi, mg_phi, mg_psi, normalize_flux, &
                      update_fission_density, output_moments, mg_incident, &
                      recon_convergence_rate
    use dgm, only : expansion_order, phi_m, psi_m, dgm_order
    use angle, only : p_leg
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
        a                 ! Angle index
    double precision :: &
        recon_error       ! Error between successive iterations
    double precision, dimension(0:expansion_order, 0:number_legendre, number_coarse_groups, number_cells) :: &
        old_phi_m         ! Scalar flux from previous iteration
    double precision, dimension(0:expansion_order, number_coarse_groups, 2 * number_angles, number_cells) :: &
        old_psi_m         ! Angular flux from previous iteration
    double precision, dimension(3) :: &
        past_error        ! Container to hold the error for last 3 iterations

    if (present(bypass_arg)) then
      bypass_flag = bypass_arg
    else
      bypass_flag = bypass_default
    end if

    ! Expand the fluxes into moment form
    call compute_flux_moments()

    past_error = 0.0

    do recon_count = 1, max_recon_iters
      ! Save the old value of the scalar flux
      old_phi_m = phi_m
      old_psi_m = psi_m

      ! Compute the cross section moments
      call compute_xs_moments()

      ! Fill the initial moments
      mg_phi = phi_m(0,:,:,:)
      mg_psi = psi_m(0,:,:,:)

      ! Solve for each order
      do dgm_order = 0, expansion_order

        ! Set incident flux to the proper order
        if (truncate_delta) then
          do a = 1, number_angles
            mg_incident(:, a) = matmul(transpose(phi_m(dgm_order, :delta_legendre_order, :, 1)), &
                                       p_leg(:delta_legendre_order, a+number_angles))
          end do
        else
          mg_incident(:, :) = psi_m(dgm_order, :, (number_angles+1):, 1)
        end if

        ! Compute the cross section moments
        call slice_xs_moments(order=dgm_order)

        call solve()

        ! Save the new flux moments
        phi_m(dgm_order,:,:,:) = mg_phi
        psi_m(dgm_order,:,:,:) = mg_psi

        ! Print the moments if verbose printing is on
        if (recon_print > 1) then
          print *, dgm_order, mg_phi
        end if

      end do

      ! Update flux using krasnoselskii iteration
      phi_m = (1.0 - lamb) * old_phi_m + lamb * phi_m
      if (store_psi) then
        psi_m = (1.0 - lamb) * old_psi_m + lamb * psi_m
      end if

      ! Compute the fission density
      call update_fission_density()

      ! Update the error
      recon_error = maxval(abs(old_phi_m - phi_m))
      past_error(3) = past_error(2)
      past_error(2) = past_error(1)
      past_error(1) = log10(recon_error)
      recon_convergence_rate = exp(sum(past_error) / 3.0)

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

    ! Unfold to fine-group flux
    call unfold_flux_moments()

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

  subroutine unfold_flux_moments()
    ! ##########################################################################
    ! Unfold the fluxes from the moments
    ! ##########################################################################

    ! Use Statements
    use control, only : number_fine_groups, number_angles, number_cells, energy_group_map
    use angle, only : p_leg
    use control, only : store_psi, number_angles, truncate_delta, &
                        delta_legendre_order
    use dgm, only : basis, expansion_order, phi_m, psi_m
    use state, only : phi, psi

    ! Variable definitions
    double precision, dimension(0:expansion_order) :: &
        tmp_psi_m     ! Legendre polynomial integration vector
    integer :: &
        a,          & ! Angle index
        c,          & ! Cell index
        g             ! Fine group index

    ! Get scalar flux from moments
    do c = 1, number_cells
      do g = 1, number_fine_groups
        phi(:, g, c) = matmul(basis(g, :), phi_m(:, :, energy_group_map(g), c))
      end do
    end do

    if (store_psi) then
      do c = 1, number_cells
        do a = 1, number_angles * 2
          do g = 1, number_fine_groups
            if (truncate_delta) then
              tmp_psi_m = matmul(phi_m(:, :delta_legendre_order, energy_group_map(g), c), &
                                 p_leg(:delta_legendre_order, a))
            else
              tmp_psi_m = psi_m(:, energy_group_map(g), a, c)
            end if
            psi(g, a, c) = dot_product(basis(g, :), tmp_psi_m)
          end do
        end do
      end do
    end if

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
    use dgm, only : phi_m, psi_m, basis, expansion_order
    use angle, only : p_leg

    ! Variable definitions
    integer :: &
        a,    & ! Angle index
        c,    & ! Cell index
        j,    & ! Expansion order index
        cg,   & ! Outer coarse group index
        ord,  & ! Delta truncation order
        g       ! Outer fine group index
    double precision :: &
        tmp_psi ! temporary angular


    ! initialize all moments to zero
    phi_m = 0.0
    psi_m = 0.0

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
          do j = 0, expansion_order
            psi_m(j, cg, a, c) = psi_m(j, cg, a, c) + basis(g, j) * tmp_psi
          end do
        end do
      end do
    end do

    !TODO: Integrate psi_m_zero over angle to get phi_m_zero

    ! Get moments for the Scalar flux
    do c = 1, number_cells
      do g = 1, number_fine_groups
        do j = 0, expansion_order
          cg = energy_group_map(g)
          phi_m(j, :, cg, c) = phi_m(j, :, cg, c) + basis(g, j) * phi(:, g, c)
        end do
      end do
    end do

  end subroutine compute_flux_moments

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
    use control, only : number_angles, number_cells, number_legendre, number_groups, &
                        delta_legendre_order, truncate_delta, number_regions, &
                        scatter_legendre_order, number_coarse_groups
    use state, only : mg_sig_t, mg_nu_sig_f, mg_mMap
    use mesh, only : mMap, dx
    use dgm, only : phi_m, psi_m, sig_s_m, delta_m, expansion_order, &
                    expanded_sig_t, expanded_nu_sig_f, expanded_sig_s
    use angle, only : p_leg

    ! Variable definitions
    integer :: &
        o,           & ! Order index
        a,           & ! Angle index
        c,           & ! Cell index
        cg,          & ! Outer coarse group index
        cgp,         & ! Inner coarse group index
        l,           & ! Legendre moment index
        r,           & ! Region index
        ord,         & ! Delta truncation order
        mat            ! Material index
    double precision :: &
        float          ! temporary double precision number
    double precision, dimension(0:expansion_order) :: &
        tmp_psi_m      ! flux arrays
    double precision, dimension(0:number_legendre, number_groups, number_regions) :: &
        homog_phi      ! Homogenization container

    ! Reset the moment containers if need be
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
      homog_phi(0:, :, r) = homog_phi(0:, :, r) + dx(c) * phi_m(0, 0:, :, c)
    end do

    ! Compute the total cross section moments
    mg_sig_t = 0.0
    do c = 1, number_cells
      mat = mMap(c)
      r = mg_mMap(c)
      do cg = 1, number_coarse_groups
        float = dot_product(phi_m(:, 0, cg, c), expanded_sig_t(:, cg, mat, 0))
        if (homog_phi(0, cg, r) /= 0.0)  then
          mg_sig_t(cg, r) = mg_sig_t(cg, r) + dx(c) * float / homog_phi(0, cg, r)
        end if
      end do
    end do

    ! Compute the fission cross section moments
    mg_nu_sig_f = 0.0
    do c = 1, number_cells
      mat = mMap(c)
      r = mg_mMap(c)
      do cg = 1, number_coarse_groups
        float = dot_product(phi_m(:, 0, cg, c), expanded_nu_sig_f(:, cg, mat))
        if (homog_phi(0, cg, r) /= 0.0)  then
          mg_nu_sig_f(cg, r) = mg_nu_sig_f(cg, r) + dx(c) * float / homog_phi(0, cg, r)
        end if
      end do
    end do

    ! Compute the scattering cross section moments
    do o = 0, expansion_order
      do c = 1, number_cells
        ! get the material for the current cell
        mat = mMap(c)
        r = mg_mMap(c)
        do cg = 1, number_coarse_groups
          do cgp = 1, number_coarse_groups
            do l = 0, scatter_legendre_order
              float = dot_product(phi_m(:, l, cgp, c), expanded_sig_s(:, l, cgp, cg, mat, o))
              if (homog_phi(l, cgp, r) /= 0.0) then
                  sig_s_m(l, cgp, cg, r, o) = sig_s_m(l, cgp, cg, r, o) &
                                            + dx(c) * float / homog_phi(l, cgp, r)
              end if
            end do
          end do
        end do
      end do
    end do

    ! Compute delta
    ord = delta_legendre_order
    do o = 0, expansion_order
      ! Add angular total cross section moment (delta) to the external source
      do c = 1, number_cells
        ! get the material for the current cell
        mat = mMap(c)
        r = mg_mMap(c)
        do a = 1, number_angles * 2
          do cg = 1, number_coarse_groups
            if (truncate_delta) then
              ! If we are truncating the delta term, then first truncate
              ! the angular flux (because the idea is that we would only store
              ! the angular moments and then the discrete delta term would be
              ! generated on the fly from the corresponding delta moments)
              tmp_psi_m = matmul(phi_m(:, :ord, cg, c), p_leg(:ord, a))
            else
              tmp_psi_m = psi_m(:, cg, a, c)
            end if
            ! Check if producing nan and not computing with a nan
            float = dot_product(tmp_psi_m(:), expanded_sig_t(:, cg, mat, o))
            float = float - mg_sig_t(cg, r) * tmp_psi_m(o)
            if ((homog_phi(0, cg, r) /= 0.0) .and. (tmp_psi_m(0) /= 0)) then
              delta_m(cg, a, r, o) = delta_m(cg, a, r, o) &
                                   + dx(c) * float / tmp_psi_m(0) * phi_m(0, 0, cg, c) / homog_phi(0, cg, r)
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
    use control, only : number_cells, number_regions, number_fine_groups, &
                        number_groups, energy_group_map
    use material, only : chi
    use state, only : mg_constant_source, mg_mMap
    use mesh, only : mMap, dx
    use dgm, only : chi_m, source_m, expansion_order, basis

    ! Variable definitions
    integer :: &
        order, & ! Expansion order index
        c,     & ! Cell index
        g,     & ! Fine group index
        cg,    & ! Coarse group index
        r,     & ! Region index
        mat      ! Material index
    double precision, dimension(number_regions) :: &
        lengths  ! Length of each region

    allocate(chi_m(number_groups, number_regions, 0:expansion_order))
    allocate(source_m(number_groups, 0:expansion_order))

    chi_m = 0.0
    source_m = 0.0

    ! Get the length of each region
    lengths = 0.0
    do c = 1, number_cells
      r = mg_mMap(c)
      lengths(r) = lengths(r) + dx(c)
    end do

    ! chi moment
    do order = 0, expansion_order
      do c = 1, number_cells
        mat = mMap(c)
        r = mg_mMap(c)
        do g = 1, number_fine_groups
          cg = energy_group_map(g)
          chi_m(cg, r, order) = chi_m(cg, r, order) + basis(g, order) * chi(g, mat) * dx(c) / lengths(r)
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
