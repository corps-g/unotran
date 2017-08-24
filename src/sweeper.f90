module sweeper
  use control, only : boundary_type, store_psi, equation_type
  use material, only : number_groups, sig_t, number_legendre
  use mesh, only : dx, number_cells, mMap
  use angle, only : number_angles, p_leg, wt, mu
  use state, only : d_source, d_nu_sig_f, d_chi, d_sig_s, d_phi, d_delta, d_sig_t, d_psi

  implicit none
  
  contains
  
  subroutine sweep(number_energy_groups, phi, psi, incoming)
    integer :: o, c, a, g, l, an, cmin, cmax, cstep, amin, amax, astep
    integer, intent(in) :: number_energy_groups
    double precision :: Q(number_energy_groups, number_angles * 2, number_cells), Ps, invmu, fiss
    double precision :: M(0:number_legendre)
    double precision, intent(inout) :: phi(:,:,:), incoming(:,:), psi(:,:,:)
    logical :: octant

    phi = 0.0  ! Reset phi

    ! Update the right hand side
    call updateRHS(Q, number_energy_groups)

    do o = 1, 2  ! Sweep over octants
      ! Sweep in the correct direction in the octant
      octant = o == 1
      cmin = merge(1, number_cells, octant)
      cmax = merge(number_cells, 1, octant)
      cstep = merge(1, -1, octant)
      amin = merge(1, number_angles, octant)
      amax = merge(number_angles, 1, octant)
      astep = merge(1, -1, octant)
      
      ! set boundary conditions
      incoming = boundary_type(o) * incoming  ! Set albedo conditions

      do c = cmin, cmax, cstep  ! Sweep over cells
        do a = amin, amax, astep  ! Sweep over angle
          ! Get the correct angle index
          an = merge(a, 2 * number_angles - a + 1, octant)
          ! get a common fraction
          invmu = dx(c) / (2 * abs(mu(a)))

          ! legendre polynomial integration vector
          M = 0.5 * wt(a) * p_leg(:,an)

          do g = 1, number_energy_groups  ! Sweep over group
            ! Use the specified equation.  Defaults to DD
            call computeEQ(Q(g,an,c), incoming(g,an), d_sig_t(g, c), invmu, Ps)

            if (store_psi) then
              psi(g,an,c) = Ps
            end if

            ! Increment the legendre expansions of the scalar flux
            phi(:,g,c) = phi(:,g,c) + M(:) * Ps
          end do
        end do
      end do
    end do
  end subroutine sweep
  
  subroutine computeEQ(S, incoming, sig, invmu, cellPsi)
    implicit none
    double precision, intent(inout) :: incoming
    double precision, intent(in) :: S, sig, invmu
    double precision, intent(out) :: cellPsi
    if (equation_type == 'DD') then
      ! Diamond Difference relationship
      cellPsi = (incoming + invmu * S) / (1 + invmu * sig)
      incoming = 2 * cellPsi - incoming
    else
      print *, 'ERROR : Equation not implemented'
    end if
  
  end subroutine computeEQ
  
  subroutine updateRHS(Q, number_groups)

    integer, intent(in) :: number_groups
    double precision, intent(inout) :: Q(:, :, :)
    double precision :: source(number_groups), scat(number_groups)
    integer :: o, c, a, g, l, an, cmin, cmax, cstep, amin, amax, astep
    logical :: octant

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

          if (allocated(d_psi)) then
            source(:) = d_source(:,an,c) - d_delta(:,an,c)  * d_psi(:,an,c)
          else
            source(:) = d_source(:,an,c)
          end if

          ! Include the external source and the fission source
          Q(:,an,c) = source(:) + d_chi(:,c) * dot_product(d_nu_sig_f(:,c), d_phi(0,:,c))

          ! Add the scattering source for each legendre moment
          do l = 0, number_legendre
            scat(:) = (2 * l + 1) * p_leg(l, an) * matmul(transpose(d_sig_s(l, :, :, c)), d_phi(l,:,c))
            Q(:,an,c) = Q(:,an,c) + scat(:)
          end do
        end do
      end do
    end do

  end subroutine updateRHS
  
end module sweeper
