module sweeper
  use control, only : boundary_type, store_psi, equation_type
  use material, only : number_groups, sig_s, sig_t, nu_sig_f, chi, number_legendre
  use mesh, only : dx, number_cells, mMap
  use angle, only : number_angles, p_leg, wt, mu
  use state, only : source

  implicit none
  
  contains
  
  subroutine sweep(phi, psi, incoming)
    integer :: o, c, a, g, gp, l, an, cmin, cmax, cstep, amin, amax, astep
    double precision :: Q(number_groups), Ps, invmu, fiss
    double precision :: phi_old(0:number_legendre,number_groups,number_cells)
    double precision :: M(0:number_legendre)
    double precision, intent(inout) :: phi(:,:,:), incoming(:,:), psi(:,:,:)
    logical :: octant
    
    phi_old = phi
    phi = 0.0  ! Reset phi
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

          ! Update the right hand side
          Q = updateSource(number_groups, source(:,an,c), phi_old(:,:,c), an, &
                           sig_s(:,:,:,mMap(c)), nu_sig_f(:,mMap(c)), chi(:,mMap(c)))
          
          do g = 1, number_groups  ! Sweep over group
            ! Use the specified equation.  Defaults to DD
            call computeEQ(Q(g), incoming(g,an), sig_t(g, mMap(c)), invmu, Ps)
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
  
  function updateSource(number_groups, Sg, Phig, angle, sig_s, nu_sig_f, chi)
    integer, intent(in) :: number_groups
    double precision :: updateSource(number_groups)
    double precision, intent(in) :: Sg(:), Phig(0:number_legendre, number_groups), nu_sig_f(:), chi(:)
    double precision, intent(in) :: sig_s(0:number_legendre,number_groups,number_groups)
    double precision :: scat(number_groups)
    integer, intent(in) :: angle
    integer :: l
    
    ! Include the external source and the fission source
    updateSource(:) = Sg(:) + chi(:) * dot_product(nu_sig_f(:), phig(0,:))

    ! Add the scattering source for each legendre moment
    do l = 0, number_legendre
      scat(:) = (2 * l + 1) * p_leg(l, angle) * matmul(transpose(sig_s(l, :, :)), phig(l,:))
      updateSource(:) = updateSource(:) + scat(:)
    end do
    
  end function updateSource
  
end module sweeper
