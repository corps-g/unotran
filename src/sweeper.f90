module sweeper
  ! ############################################################################
  ! Sweep through the cells,angles,groups to implement discrete ordinates
  ! ############################################################################

  use control, only : boundary_type, store_psi, equation_type
  use material, only : number_groups, sig_t, number_legendre
  use mesh, only : dx, number_cells, mMap
  use angle, only : number_angles, p_leg, wt, mu
  use state, only : d_source, d_nu_sig_f, d_chi, d_sig_s, d_phi, d_delta, &
                    d_sig_t, d_psi, d_keff, d_incoming

  implicit none
  
  contains
  
  subroutine sweep(number_energy_groups, phi, psi)
    ! ##########################################################################
    ! Sweep over each cell, angle, group, and ocatant
    ! ##########################################################################

    integer, intent(in) :: &
        number_energy_groups ! Number of energy groups
    double precision, intent(inout), dimension(:,:,:) :: &
        phi,               & ! Scalar flux for current iteration
        psi                  ! Angular flux for current iteration
    integer :: &
        o,                 & ! Octant index
        c,                 & ! Cell index
        a,                 & ! Angle index
        g,                 & ! Group index
        an,                & ! Global angle index
        cmin,              & ! Lower cell number
        cmax,              & ! Upper cell number
        cstep,             & ! Cell stepping direction
        amin,              & ! Lower angle number
        amax,              & ! Upper angle number
        astep                ! Angle stepping direction
    double precision, allocatable, dimension(:,:,:) :: &
        Q                    ! Source
    double precision, allocatable, dimension(:) :: &
        M                    ! Legendre polynomial integration vector
    double precision :: &
        Ps                   ! Angular flux at cell center
    logical :: &
        octant               ! Positive/Negative octant flag

    allocate(Q(number_energy_groups, number_angles * 2, number_cells))
    allocate(M(0:number_legendre))

    phi = 0.0  ! Reset phi

    ! Update the right hand side
    call updateRHS(Q, number_energy_groups)

    do o = 1, 2  ! Sweep over octants
      ! Sweep in the correct direction within the octant
      octant = o == 1
      cmin = merge(1, number_cells, octant)
      cmax = merge(number_cells, 1, octant)
      cstep = merge(1, -1, octant)
      amin = merge(1, number_angles, octant)
      amax = merge(number_angles, 1, octant)
      astep = merge(1, -1, octant)
      
      ! set boundary conditions
      d_incoming = boundary_type(o) * d_incoming  ! Set albedo conditions

      do c = cmin, cmax, cstep  ! Sweep over cells
        do a = amin, amax, astep  ! Sweep over angle
          ! Get the correct angle index
          an = merge(a, number_angles + a, octant)

          ! legendre polynomial integration vector
          M = wt(a) * p_leg(:,an)

          do g = 1, number_energy_groups  ! Sweep over group
            ! Use the specified equation.  Defaults to DD
            call computeEQ(Q(g,an,c), d_incoming(g,a), d_sig_t(g, c), dx(c), mu(a), Ps)

            if (store_psi) then
              psi(g,an,c) = Ps
            end if

            ! Increment the legendre expansions of the scalar flux
            phi(:,g,c) = phi(:,g,c) + M(:) * Ps
          end do
        end do
      end do
    end do

    deallocate(Q, M)

  end subroutine sweep
  
  subroutine computeEQ(S, incoming, sig, dx, mua, cellPsi)
    ! ##########################################################################
    ! Compute the value for the closure reslationship
    ! ##########################################################################

    double precision, intent(inout) :: &
        incoming ! Angular flux incident on the cell
    double precision, intent(in) :: &
        S,     & ! Source within the cell
        sig,   & ! Total cross section within the cell
        dx,    & ! Width of the cell
        mua      ! Angle for the cell
    double precision, intent(out) :: &
        cellPsi  ! Angular flux at cell center
    double precision :: &
        tau,   & ! Parameter used in step characteristics
        A,     & ! Parameter used in step characteristics
        invmu    ! Parameter used in diamond and step differences

    select case (equation_type)
    case ('DD')
      ! Diamond Difference relationship
      invmu = dx / (2 * abs(mua))
      cellPsi = (incoming + invmu * S) / (1 + invmu * sig)
      incoming = 2 * cellPsi - incoming
    case ('SC')
      ! Step Characteristics relationship
      tau = sig * dx / mua
      A = exp(-tau)
      cellPsi = incoming * (1.0 - A) / tau + S * (sig * dx + mua * (A - 1.0)) / (sig ** 2 * dx)
      incoming = A * incoming + S * (1.0 - A) / sig
    case ('SD')
      ! Step Difference relationship
      invmu = dx / (abs(mua))
      cellPsi = (incoming + invmu * S) / (1 + invmu * sig)
      incoming = cellPsi
    case default
      print *, 'ERROR : Equation not implemented'
    end select
  
  end subroutine computeEQ
  
  subroutine updateRHS(Q, number_groups)
    ! ##########################################################################
    ! Update the source including external, scattering, and fission
    ! ##########################################################################

    integer, intent(in) :: &
        number_groups ! Number of energy groups
    double precision, intent(inout), dimension(:,:,:) :: &
        Q             ! Container for total source
    double precision, dimension(number_groups) :: &
        source,     & ! Array to hold the source per group
        scat          ! Array to hold scattering source per group
    integer :: &
        o,          & ! Octant index
        c,          & ! Cell index
        a,          & ! Angle index
        l,          & ! Legendre index
        an,         & ! Global angle index
        cmin,       & ! Lower cell number
        cmax,       & ! Upper cell number
        cstep,      & ! Cell stepping direction
        amin,       & ! Lower angle number
        amax,       & ! Upper angle number
        astep         ! Angle stepping direction
    logical :: &
        octant        ! Positive/Negative octant flag

    ! Initialize the source to zero
    Q = 0.0

    do o = 1, 2  ! Sweep over octants
      ! Sweep in the correct direction in the octant
      octant = o == 1
      cmin = merge(1, number_cells, octant)
      cmax = merge(number_cells, 1, octant)
      cstep = merge(1, -1, octant)
      amin = merge(1, number_angles, octant)
      amax = merge(number_angles, 1, octant)
      astep = merge(1, -1, octant)
      do c = cmin, cmax, cstep  ! Sweep over cells
        do a = amin, amax, astep  ! Sweep over angle
          ! Get the correct angle index
          an = merge(a, 2 * number_angles - a + 1, octant)

          ! Get the External source and possibly the angular total XS moment source
          if (allocated(d_psi)) then
            source(:) = d_source(:,an,c) - d_delta(:,an,c)  * d_psi(:,an,c)
          else
            source(:) = d_source(:,an,c)
          end if

          ! Include the external source and the fission source
          Q(:,an,c) = 0.5 * (source(:) + d_chi(:,c) / d_keff * dot_product(d_nu_sig_f(:,c), d_phi(0,:,c)))

          ! Add the scattering source for each Legendre moment
          do l = 0, number_legendre
            scat(:) = 0.5 / (2 * l + 1) * p_leg(l, an) * matmul(transpose(d_sig_s(l, :, :, c)), d_phi(l,:,c))
            Q(:,an,c) = Q(:,an,c) + scat(:)
          end do
        end do
      end do
    end do

  end subroutine updateRHS
  
end module sweeper
