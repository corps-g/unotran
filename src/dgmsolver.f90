module dgmsolver
  ! ############################################################################
  ! Solve discrete ordinates using the Discrete Generalized Multigroup Method
  ! ############################################################################

  use control, only : dp

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
                        ignore_warnings, lamb, number_cells, spatial_dimension, &
                        number_moments, number_angles_per_octant, min_recon_iters, number_coarse_groups
    use state, only : keff, phi, psi, mg_phi, mg_psi, normalize_flux, &
                      update_fission_density, output_moments, recon_convergence_rate, &
                      mg_incident_x, mg_incident_y, eigen_count, recon_count, exit_status
    use dgm, only : expansion_order, phi_m, psi_m, dgm_order
    use solver, only : solve
    use omp_lib, only : omp_get_wtime

    ! Variable definitions
    logical, intent(in), optional :: &
        bypass_arg        ! Allow the dgmsolver to do one recon loop selectively
    logical, parameter :: &
        bypass_default = .false.  ! Default value of bypass_arg
    logical :: &
        bypass_flag       ! Local variable to signal an eigen loop bypass
    real(kind=dp) :: &
        recon_error,    & ! Error between successive iterations
        start,          & ! Start time of the sweep function
        past_error,     & ! The error for last iteration
        past_error2,    & ! The error for two iterations back
        ave_sweep_time    ! Average time in seconds per sweep
    real(kind=dp), dimension(0:expansion_order, 0:number_moments, number_coarse_groups, number_cells) :: &
        old_phi_m         ! Scalar flux from previous iteration
    real(kind=dp), dimension(0:expansion_order, number_coarse_groups, 2 * spatial_dimension * &
                                number_angles_per_octant, number_cells) :: &
        old_psi_m         ! Angular flux from previous iteration
    integer :: &
        recon_estimate    ! Estimate for the total number of iterations
        

    if (present(bypass_arg)) then
      bypass_flag = bypass_arg
    else
      bypass_flag = bypass_default
    end if

    ! Expand the fluxes into moment form
    call compute_flux_moments()

    past_error = 0.0_8
    ave_sweep_time = 0.0_8

    do recon_count = 1, max_recon_iters
      start = omp_get_wtime()
    
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

        ! Reset the incident conditions to vacuum
        mg_incident_x(:,:,:,:) = 0.0_8
        mg_incident_y(:,:,:,:) = 0.0_8

        ! Compute the cross section moments
        call slice_xs_moments(order=dgm_order)

        call solve()
        
        if (exit_status == 1) then
            return
        end if

        ! Save the new flux moments
        phi_m(dgm_order,:,:,:) = mg_phi
        psi_m(dgm_order,:,:,:) = mg_psi

        ! Print the moments if verbose printing is on
        if (recon_print > 1) then
          print *, dgm_order, mg_phi
        end if

      end do  ! End dgm_order loop

      ! Update flux using krasnoselskii iteration
      phi_m = (1.0_8 - lamb) * old_phi_m + lamb * phi_m
      if (store_psi) then
        psi_m = (1.0_8 - lamb) * old_psi_m + lamb * psi_m
      end if

      ! Compute the fission density
      call update_fission_density()

      ! Update the error
      past_error2 = past_error
      past_error = recon_error
      recon_error = maxval(abs(old_phi_m - phi_m))
      
      recon_convergence_rate = log10(recon_error ** 3 / past_error ** 4 * past_error2) / 2
      recon_estimate = int(log10(recon_tolerance / recon_error) / recon_convergence_rate) + recon_count
      if (recon_estimate < 0) then
        recon_estimate = 1000000
      end if

      ave_sweep_time = ((recon_count - 1) * ave_sweep_time + (omp_get_wtime() - start)) / recon_count

      ! Print output
      if (recon_print > 0) then
        write(*, 1001) recon_count, recon_error, keff, eigen_count, ave_sweep_time, &
                       recon_convergence_rate, recon_estimate
        1001 format ("recon: ", i5, " Error: ", es10.4E2, " k: ", f12.10, &
                     " eSweeps: ", i5, " sweepTime: ", f6.1, " s", &
                     " rate: ", f6.3, " iterEst: ", i5)
        if (recon_print > 1) then
          call output_moments()
        end if
      end if

      ! Check if tolerance is reached
      if ((recon_error < recon_tolerance .and. recon_count >= min_recon_iters) .or. bypass_flag) then
        exit
      end if

    end do  ! End recon_count loop

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
    use control, only : number_fine_groups, number_angles, number_cells, &
                        energy_group_map, store_psi, truncate_delta, delta_leg_order
    use angle, only : p_leg
    use dgm, only : basis, expansion_order, phi_m, psi_m
    use state, only : phi, psi

    ! Variable definitions
    real(kind=dp), dimension(0:expansion_order) :: &
        tmp_psi_m     ! Legendre polynomial integration vector
    integer :: &
        a,          & ! Angle index
        c,          & ! Cell index
        g             ! Fine group index

    ! Get scalar flux from moments
    do c = 1, number_cells
      do g = 1, number_fine_groups
        phi(:, g, c) = matmul(basis(g, :), phi_m(:, :, energy_group_map(g), c))
      end do  ! End g loop
    end do  ! End c loop

    if (store_psi) then
      do c = 1, number_cells
        do a = 1, number_angles
          do g = 1, number_fine_groups
            if (truncate_delta) then
              tmp_psi_m = matmul(phi_m(:, :delta_leg_order, energy_group_map(g), c), &
                                 p_leg(:delta_leg_order, a))
            else
              tmp_psi_m = psi_m(:, energy_group_map(g), a, c)
            end if
            psi(g, a, c) = dot_product(basis(g, :), tmp_psi_m)
          end do  ! End g loop
        end do  ! End a loop
      end do  ! End c loop
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
                        energy_group_map, delta_leg_order, truncate_delta
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
    real(kind=dp) :: &
        tmp_psi ! temporary angular


    ! initialize all moments to zero
    phi_m = 0.0_8
    psi_m = 0.0_8

    ! Get moments for the Angular flux
    do c = 1, number_cells
      do a = 1, number_angles
        do g = 1, number_fine_groups
          cg = energy_group_map(g)

          if (truncate_delta) then
            ! If we are truncating the delta term, then first truncate
            ! the angular flux (because the idea is that we would only store
            ! the angular moments and then the discrete delta term would be
            ! generated on the fly from the corresponding delta moments)
            ord = delta_leg_order
            tmp_psi = dot_product(p_leg(:ord, a), phi(:ord, g, c))
          else
            tmp_psi = psi(g, a, c)
          end if
          do j = 0, expansion_order
            psi_m(j, cg, a, c) = psi_m(j, cg, a, c) + basis(g, j) * tmp_psi
          end do  ! End j loop
        end do  ! End g loop
      end do  ! End a loop
    end do  ! End c loop

    !TODO: Integrate psi_m_zero over angle to get phi_m_zero

    ! Get moments for the Scalar flux
    do c = 1, number_cells
      do g = 1, number_fine_groups
        cg = energy_group_map(g)
        do j = 0, expansion_order
          phi_m(j, :, cg, c) = phi_m(j, :, cg, c) + basis(g, j) * phi(:, g, c)
        end do  ! End j loop
      end do  ! End g loop
    end do  ! End c loop

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
    use control, only : number_angles, number_cells_x, number_cells_y, number_moments, &
                        number_groups, delta_leg_order, truncate_delta, number_regions, &
                        scatter_leg_order, number_coarse_groups
    use state, only : mg_sig_t, mg_nu_sig_f, mg_mMap
    use mesh, only : mMap, dx, dy
    use dgm, only : phi_m, psi_m, sig_s_m, delta_m, expansion_order, &
                    expanded_sig_t, expanded_nu_sig_f, expanded_sig_s
    use angle, only : p_leg

    ! Variable definitions
    integer :: &
        o,           & ! Order index
        a,           & ! Angle index
        c,           & ! Cell index
        cx,          & ! Cell index for x cells
        cy,          & ! Cell index for y cells
        cg,          & ! Outer coarse group index
        cgp,         & ! Inner coarse group index
        l,           & ! Legendre moment index
        r,           & ! Region index
        ord,         & ! Delta truncation order
        mat            ! Material index
    real(kind=dp) :: &
        tolerance,   & ! Variable to hold the tolerance check for dividing by zero
        float          ! temporary double precision number
    real(kind=dp), dimension(0:expansion_order) :: &
        tmp_psi_m      ! flux arrays
    real(kind=dp), dimension(0:number_moments, number_groups, number_regions) :: &
        homog_phi      ! Homogenization container

    ! Reset the moment containers if need be
    if (allocated(delta_m)) then
      deallocate(delta_m)
    end if
    if (allocated(sig_s_m)) then
      deallocate(sig_s_m)
    end if

    allocate(delta_m(number_groups, number_angles, number_regions, 0:expansion_order))
    allocate(sig_s_m(0:scatter_leg_order, number_groups, number_groups, number_regions, 0:expansion_order))

    ! initialize all moments and mg containers to zero
    sig_s_m = 0.0_8
    delta_m = 0.0_8

    ! Set the tolerance as the smallest number of the same type as homog_phi
    tolerance = tiny(homog_phi)

    ! Compute the denominator for spatial homogenization
    homog_phi = 0.0_8
    c = 1
    do cy = 1, number_cells_y
      do cx = 1, number_cells_x
        r = mg_mMap(c)
        homog_phi(0:, :, r) = homog_phi(0:, :, r) + dx(cx) * dy(cy) * phi_m(0, 0:, :, c)
        c = c + 1
      end do  ! End cx loop
    end do  ! End cy loop

    ! Compute the total cross section moments
    mg_sig_t = 0.0_8
    c = 1
    do cy = 1, number_cells_y
      do cx = 1, number_cells_x
        mat = mMap(c)
        r = mg_mMap(c)
        do cg = 1, number_coarse_groups
          float = dot_product(phi_m(:, 0, cg, c), expanded_sig_t(:, cg, mat, 0))
          ! Avoid dividing by zero
          if (abs(homog_phi(0, cg, r)) > tolerance) then
            mg_sig_t(cg, r) = mg_sig_t(cg, r) + dx(cx) * dy(cy) * float / homog_phi(0, cg, r)
          end if
        end do  ! End cg loop
        c = c + 1
      end do  ! End cx loop
    end do  ! End cy loop

    ! Compute the fission cross section moments
    mg_nu_sig_f = 0.0_8
    c = 1
    do cy = 1, number_cells_y
      do cx = 1, number_cells_x
        mat = mMap(c)
        r = mg_mMap(c)
        do cg = 1, number_coarse_groups
          float = dot_product(phi_m(:, 0, cg, c), expanded_nu_sig_f(:, cg, mat))
          ! Avoid dividing by zero
          if (abs(homog_phi(0, cg, r)) > tolerance) then
            mg_nu_sig_f(cg, r) = mg_nu_sig_f(cg, r) + dx(cx) * dy(cy) * float / homog_phi(0, cg, r)
          end if
        end do  ! End cg loop
        c = c + 1
      end do  ! End cx loop
    end do  ! End cy loop

    ! Compute the scattering cross section moments
    do o = 0, expansion_order
      c = 1
      do cy = 1, number_cells_y
        do cx = 1, number_cells_x
          ! get the material for the current cell
          mat = mMap(c)
          r = mg_mMap(c)
          do cg = 1, number_coarse_groups
            do cgp = 1, number_coarse_groups
              do l = 0, scatter_leg_order
                float = dot_product(phi_m(:, l, cgp, c), expanded_sig_s(:, l, cgp, cg, mat, o))
                ! Avoid dividing by zero
                if (abs(homog_phi(l, cgp, r)) > tolerance) then
                    sig_s_m(l, cgp, cg, r, o) = sig_s_m(l, cgp, cg, r, o) &
                                              + dx(cx) * dy(cy) * float / homog_phi(l, cgp, r)
                end if
              end do  ! End l loop
            end do  ! End cgp loop
          end do  ! End cg loop
          c = c + 1
        end do  ! End cx loop
      end do  ! End cy loop
    end do  ! End o loop

    ! Compute delta
    ord = delta_leg_order
    do o = 0, expansion_order
      ! Add angular total cross section moment (delta) to the external source
      c = 1
      do cy = 1, number_cells_y
        do cx = 1, number_cells_x
          ! get the material for the current cell
          mat = mMap(c)
          r = mg_mMap(c)
          do a = 1, number_angles
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
              ! Avoid dividing by zero
              if (abs(homog_phi(0, cg, r)) > tolerance .and. (abs(tmp_psi_m(0)) > tolerance)) then
                delta_m(cg, a, r, o) = delta_m(cg, a, r, o) &
                                     + dx(cx) * dy(cy) * float / tmp_psi_m(0) * phi_m(0, 0, cg, c) / homog_phi(0, cg, r)
              end if
            end do  ! End cg loop
          end do  ! End a loop
          c = c + 1
        end do  ! End cx loop
      end do  ! End cy loop
    end do  ! End o loop


  end subroutine compute_xs_moments

  subroutine compute_source_moments()
    ! ##########################################################################
    ! Expand the source and chi using the basis functions
    ! ##########################################################################

    ! Use Statements
    use control, only : number_cells_x, number_cells_y, number_regions, number_fine_groups, &
                        number_groups, energy_group_map
    use material, only : chi
    use state, only : mg_constant_source, mg_mMap
    use mesh, only : mMap, dx, dy
    use dgm, only : chi_m, source_m, expansion_order, basis

    ! Variable definitions
    integer :: &
        order, & ! Expansion order index
        c,     & ! Cell index
        cx,    & ! Cell index for x cells
        cy,    & ! Cell index for y cells
        g,     & ! Fine group index
        cg,    & ! Coarse group index
        r,     & ! Region index
        mat      ! Material index
    real(kind=dp), dimension(number_regions) :: &
        lengths  ! Length of each region

    allocate(chi_m(number_groups, number_regions, 0:expansion_order))
    allocate(source_m(number_groups, 0:expansion_order))

    chi_m = 0.0_8
    source_m = 0.0_8

    ! Get the length of each region
    lengths = 0.0_8
    c = 1
    do cy = 1, number_cells_y
      do cx = 1, number_cells_x
        r = mg_mMap(c)
        lengths(r) = lengths(r) + dx(cx) * dy(cy)
        c = c + 1
      end do  ! End cx loop
    end do  ! End cy loop

    ! chi moment
    do order = 0, expansion_order
      c = 1
      do cy = 1, number_cells_y
        do cx = 1, number_cells_x
          mat = mMap(c)
          r = mg_mMap(c)
          do g = 1, number_fine_groups
            cg = energy_group_map(g)
            chi_m(cg, r, order) = chi_m(cg, r, order) + basis(g, order) * chi(g, mat) * dx(cx) * dy(cy) / lengths(r)
          end do  ! End g loop
          c = c + 1
        end do  ! End cx loop
      end do  ! End cy loop
    end do  ! End order loop

    ! Source moment
    do order = 0, expansion_order
      do g = 1, number_fine_groups
        cg = energy_group_map(g)
        source_m(cg, order) = source_m(cg, order) + basis(g, order) * mg_constant_source
      end do  ! End g loop
    end do  ! End order loop

  end subroutine compute_source_moments

end module dgmsolver
