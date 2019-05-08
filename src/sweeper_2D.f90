module sweeper_2D

  implicit none

  contains
  
  subroutine apply_transport_operator_2D(phi)
    ! ##########################################################################
    ! Sweep over each cell, angle, and octant
    ! ##########################################################################

    ! Use Statements
    use angle, only : p_leg, wt, mu, eta, PI
    use mesh, only : dx, dy
    use control, only : store_psi, number_angles, number_cells, number_cells_x, number_cells_y, &
                        number_legendre, number_groups, use_DGM, scatter_leg_order, &
                        boundary_east, boundary_west, boundary_north, boundary_south, number_moments
    use state, only : mg_sig_t, sweep_count, mg_mMap, mg_incident_x, mg_incident_y, &
                      mg_psi, mg_source, sigphi, scaling
    use sources, only : compute_source
    use omp_lib, only : omp_get_wtime
    use dgm, only : delta_m, psi_m, dgm_order

    ! Variable definitions
    double precision, intent(inout), dimension(:,:,:) :: &
        phi           ! Scalar flux for current iteration and group g
    double precision, dimension(0:number_moments, number_groups, number_cells) :: &
        phi_update    ! Container to hold the updated scalar flux
    integer :: &
        g,          & ! Group index
        o,          & ! Octant index
        c,          & ! Cell index
        cx,         & ! x cell index
        cy,         & ! y cell index
        mat,        & ! Material index
        a,          & ! Angle index
        an,         & ! Global angle index
        l,          & ! Degree index for spherical harmonics
        m,          & ! Order index for spherical harmonics
        ll,         & ! Basis index
        src_x,      & ! Source index for cell boundary condition in x direction
        src_y,      & ! Source index for cell boundary condition in y direction
        dst_x,      & ! Destination index for cell boundary condition in x direction
        dst_y,      & ! Destination index for cell boundary condition in y direction
        cx_start,   & ! Lower cell number in x direction
        cx_stop,    & ! Upper cell number in x direction
        cx_step,    & ! Cell stepping direction in x direction
        cy_start,   & ! Lower cell number in y direction
        cy_stop,    & ! Upper cell number in y direction
        cy_step       ! Cell stepping direction in y direction
    double precision, dimension(number_groups) :: &
        psi_center, & ! Angular flux at cell center
        source        ! Fission, In-Scattering, External source in group g

    ! Increment the sweep counter
    sweep_count = sweep_count + 1

    ! Reset phi
    phi_update = 0.0

    ! Update the forcing function
    call compute_source()

    do o = 1, 4  ! Sweep over octants

      ! Get sweep direction and set boundary condition for x cells
      if (o == 1 .or. o == 2) then
        cx_start = 1
        cx_stop = number_cells_x
        cx_step = 1
        src_x = merge(1, 2, o == 1)
        dst_x = merge(4, 3, o == 1)
        mg_incident_x(:,:,:,dst_x) = boundary_west * mg_incident_x(:,:,:,src_x)
      else
        cx_start = number_cells_x
        cx_stop = 1
        cx_step = -1
        src_x = merge(3, 4, o == 3)
        dst_x = merge(2, 1, o == 3)
        mg_incident_x(:,:,:,dst_x) = boundary_east * mg_incident_x(:,:,:,src_x)
      end if

      ! Get sweep direction and set boundary condition for y cells
      if (o == 1 .or. o == 4) then
        cy_start = 1
        cy_stop = number_cells_y
        cy_step = 1
        src_y = merge(1, 4, o == 1)
        dst_y = merge(2, 3, o == 1)
        mg_incident_y(:,:,:,dst_y) = boundary_north * mg_incident_y(:,:,:,src_y)
      else
        cy_start = number_cells_y
        cy_stop = 1
        cy_step = -1
        src_y = merge(2, 3, o == 2)
        dst_y = merge(1, 4, o == 2)
        mg_incident_y(:,:,:,dst_y) = boundary_south * mg_incident_y(:,:,:,src_y)
      end if

      do cy = cy_start, cy_stop, cy_step  ! Sweep over cells in y direction
        do cx = cx_start, cx_stop, cx_step  ! Sweep over cells in x direction
          c = (cy - 1) * number_cells_x + cx
          mat = mg_mMap(c)

          do a = 1, number_angles  ! Sweep over angle
            an = (o - 1) * number_angles + a

            ! Get the source in this cell, group, and angle
            source(:) = mg_source(:, c)
            ll = 0
            do l = 0, scatter_leg_order
              do m = -l, l
                source(:) = source(:) + sigphi(ll,:,c) * p_leg(ll,an) * (2 * l + 1)
                ll = ll + 1
              end do
            end do

            if (use_DGM) then
              source(:) = source(:) - delta_m(:, an, mg_mMap(c), dgm_order) * psi_m(0, :, an, c)
            end if

            ! Solve the equation for psi_center
            call computeEQ(source(:), mg_sig_t(:, mat), dx(cx), dy(cy), mu(a), eta(a), &
                           mg_incident_x(:, a, cy, dst_x), mg_incident_y(:, a, cx, dst_y), psi_center)

            ! Store psi if desired
            if (store_psi) then
              mg_psi(:, an, c) = psi_center(:)
            end if

            ! Increment the legendre expansions of the scalar flux
            do ll = 0, number_moments
              phi_update(ll, :, c) = phi_update(ll, :, c) + wt(a) * p_leg(ll,an) * psi_center(:)
            end do
          end do
        end do
      end do

    end do

    phi = phi_update

  end subroutine apply_transport_operator_2D
  
  subroutine computeEQ(S, sig, dx, dy, mua, eta, inc_x, inc_y, cellPsi)
    ! ##########################################################################
    ! Compute the value for the closure relationship
    ! ##########################################################################

    ! Use statements
    use control, only : equation_type

    ! Variable definitions
    double precision, dimension(:), intent(inout) :: &
        inc_x,   & ! Angular flux incident on the cell from x direction
        inc_y      ! Angular flux incident on the cell from y direction
    double precision, dimension(:), intent(in) :: &
        S,            & ! Source within the cell
        sig             ! Total cross section within the cell
    double precision, intent(in) :: &
        dx,           & ! Width of the cell in x direction
        dy,           & ! Width of the cell in y direction
        eta,          & ! Angle for the cell
        mua             ! Angle for the cell
    double precision, dimension(:), intent(inout) :: &
        cellPsi         ! Angular flux at cell center
    double precision, allocatable, dimension(:) :: &
        tau,          & ! Parameter used in step characteristics
        A               ! Parameter used in step characteristics
    double precision :: &
        coef_x,       & ! Parameter used in diamond and step differences
        coef_y,       & ! Parameter used in diamond and step differences
        coef(size(sig)) ! Parameter used in diamond and step differences

    if (equation_type == 'DD') then
      ! Diamond Difference relationship
      coef_x = 2.0 * mua / dx
      coef_y = 2.0 * eta / dy
      coef = 1.0 / (sig(:) + coef_x + coef_y)
      cellPsi(:) = coef * (S(:) + coef_x * inc_x(:) + coef_y * inc_y(:))
      inc_x(:) = 2 * cellPsi(:) - inc_x(:)
      inc_y(:) = 2 * cellPsi(:) - inc_y(:)
!    else if (equation_type == 'SC') then
!      ! Step Characteristics relationship
!      allocate(tau(size(sig)), A(size(sig)))
!      tau = sig(:) * dx / mua
!      A = exp(-tau)
!      cellPsi = incident * (1.0 - A) / tau + S * (sig * dx + mua * (A - 1.0)) / (sig ** 2 * dx)
!      incident = A * incident + S * (1.0 - A) / sig
!      deallocate(tau, A)
!    else if (equation_type == 'SD') then
!      ! Step Difference relationship
!      invmu = dx / (abs(mua))
!      cellPsi = (incident + invmu * S) / (1 + invmu * sig)
!      incident = cellPsi
    else
      print *, 'ERROR : Equation not implemented'
      stop
    end if

  end subroutine computeEQ
  
end module sweeper_2D
