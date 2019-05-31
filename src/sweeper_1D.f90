module sweeper_1D

  use control, only : dp

  implicit none

  contains
  
  subroutine apply_transport_operator_1D(phi)
    ! ##########################################################################
    ! Sweep over each cell, angle, and octant
    ! ##########################################################################

    ! Use Statements
    use angle, only : p_leg, wt, mu
    use mesh, only : dx
    use control, only : store_psi, number_angles_per_octant, number_cells, scatter_leg_order, &
                        number_legendre, number_groups, use_DGM, boundary_east, number_angles, &
                        boundary_west
    use state, only : mg_sig_t, sweep_count, mg_mMap, mg_incident_x, mg_psi, &
                      mg_source, sigphi
    use sources, only : compute_source
    use omp_lib, only : omp_get_wtime
    use dgm, only : delta_m, psi_m, dgm_order

    ! Variable definitions
    real(kind=dp), intent(inout), dimension(:,:,:) :: &
        phi           ! Scalar flux for current iteration and group g
    real(kind=dp), dimension(0:number_legendre, number_groups, number_cells) :: &
        phi_update    ! Container to hold the updated scalar flux
    integer :: &
        g,          & ! Group index
        o,          & ! Octant index
        c,          & ! Cell index
        mat,        & ! Material index
        a,          & ! Angle index
        an,         & ! Global angle index
        cmin,       & ! Lower cell number
        cmax,       & ! Upper cell number
        cstep,      & ! Cell stepping direction
        amin,       & ! Lower angle number
        amax,       & ! Upper angle number
        astep         ! Angle stepping direction
    real(kind=dp), dimension(0:number_legendre) :: &
        M           ! Legendre polynomial integration vector
    real(kind=dp), dimension(number_groups) :: &
        psi_center, & ! Angular flux at cell center
        source        ! Fission, In-Scattering, External source in group g
    logical :: &
        octant        ! Positive/Negative octant flag

    ! Increment the sweep counter
    sweep_count = sweep_count + 1

    ! Reset phi
    phi_update = 0.0_8

    ! Update the forcing function
    call compute_source()

    do o = 1, 2  ! Sweep over octants
      ! Sweep in the correct direction within the octant
      octant = o == 1
      amin = merge(1, number_angles_per_octant, octant)
      amax = merge(number_angles_per_octant, 1, octant)
      astep = merge(1, -1, octant)
      cmin = merge(1, number_cells, octant)
      cmax = merge(number_cells, 1, octant)
      cstep = merge(1, -1, octant)

      ! set boundary conditions
      if (o == 1) then
        mg_incident_x = boundary_west * mg_incident_x  ! Set albedo conditions
      else
        mg_incident_x = boundary_east * mg_incident_x  ! Set albedo conditions
      end if

      do c = cmin, cmax, cstep  ! Sweep over cells
        mat = mg_mMap(c)

        do a = amin, amax, astep  ! Sweep over angle
          ! Get the correct angle index
          an = merge(a, number_angles - a + 1, octant)

          ! legendre polynomial integration vector
          M = wt(a) * p_leg(:, an)

          ! Get the source in this cell, group, and angle
          source(:) = mg_source(:, c)
          source(:) = source(:) + matmul(transpose(sigphi(:scatter_leg_order,:,c)), p_leg(:scatter_leg_order,an))
          if (use_DGM) then
            source(:) = source(:) - delta_m(:, an, mg_mMap(c), dgm_order) * psi_m(0, :, an, c)
          end if

          call computeEQ(source(:), mg_sig_t(:, mat), dx(c), mu(a), mg_incident_x(:, a, 1, 1), psi_center)

          if (store_psi) then
            mg_psi(:, an, c) = psi_center(:)
          end if

          ! Loop over the energy groups
          do g = 1, number_groups

            ! Increment the legendre expansions of the scalar flux
            phi_update(:, g, c) = phi_update(:, g, c) + M(:) * psi_center(g)
          end do  ! End g loop
        end do  ! End a loop
      end do  ! End c loop
    end do  ! End o loop

    phi = phi_update

  end subroutine apply_transport_operator_1D
  
  subroutine computeEQ(S, sig, dx, mua, incident, cellPsi)
    ! ##########################################################################
    ! Compute the value for the closure relationship
    ! ##########################################################################

    ! Use statements
    use control, only : equation_type

    ! Variable definitions
    real(kind=dp), dimension(:), intent(inout) :: &
        incident ! Angular flux incident on the cell
    real(kind=dp), dimension(:), intent(in) :: &
        S,     & ! Source within the cell
        sig      ! Total cross section within the cell
    real(kind=dp), intent(in) :: &
        dx,    & ! Width of the cell
        mua      ! Angle for the cell
    real(kind=dp), dimension(:), intent(inout) :: &
        cellPsi  ! Angular flux at cell center
    real(kind=dp), allocatable, dimension(:) :: &
        tau,   & ! Parameter used in step characteristics
        A        ! Parameter used in step characteristics
    real(kind=dp) :: &
        invmu    ! Parameter used in diamond and step differences

    if (equation_type == 'DD') then
      ! Diamond Difference relationship
      invmu = dx / (2.0_8 * abs(mua))
      cellPsi(:) = (incident(:) + invmu * S(:)) / (1.0_8 + invmu * sig(:))
      incident(:) = 2.0_8 * cellPsi(:) - incident(:)
    else if (equation_type == 'SC') then
      ! Step Characteristics relationship
      allocate(tau(size(sig)), A(size(sig)))
      tau = sig(:) * dx / mua
      A = exp(-tau)
      cellPsi = incident * (1.0_8 - A) / tau + S * (sig * dx + mua * (A - 1.0_8)) / (sig ** 2.0_8 * dx)
      incident = A * incident + S * (1.0_8 - A) / sig
      deallocate(tau, A)
    else if (equation_type == 'SD') then
      ! Step Difference relationship
      invmu = dx / (abs(mua))
      cellPsi = (incident + invmu * S) / (1.0_8 + invmu * sig)
      incident = cellPsi
    else
      print *, 'ERROR : Equation not implemented'
      stop
    end if

  end subroutine computeEQ
  
end module sweeper_1D
