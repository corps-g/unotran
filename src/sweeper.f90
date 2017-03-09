module sweeper
  ! Uses diamond difference
  ! Assumes only vacuum conditions
  use material, only: sig_s, sig_t, vsig_f, chi, number_groups, number_legendre
  use mesh, only: dx, number_cells, mMap
  use angle, only: number_angles, p_leg, wt, mu
  use state, only: phi, psi, source, store_psi, equation
  implicit none
  
  contains
  ! Define source
  
  subroutine sweep()
    integer :: o, c, a, g, gp, l, an, cmin, cmax, cstep, amin, amax, astep
    double precision :: incoming(number_angles*2), Q, Ps, invmu
    double precision :: phi_old(0:number_legendre,number_groups,number_cells)
    
    do o = 1, 2  ! Sweep over octants
      ! Sweep in the correct direction in the octant
      if (o .eq. 1) then
        cmin = 1
        cmax = number_cells
        cstep = 1
        amin = 1
        amax = number_angles
        astep = 1
      else
        cmin = number_cells
        cmax = 1
        cstep = -1
        amin = number_angles
        amax = 1
        astep = -1
      end if
      
      phi_old = phi
      phi = 0.0  ! Reset phi
      incoming = 0.0  ! Set vacuum conditions for both faces
      do c = cmin, cmax, cstep  ! Sweep over cells
        do g = 1, number_groups  ! Sweep over group
          do a = amin, amax, astep  ! Sweep over angle
            ! Get the correct angle index
            if (o .eq. 1) then
              an = a
            else
              an = 2 * number_angles - a + 1
            end if
          
            ! get a common fraction
            invmu = dx(c) / (2 * abs(mu(a)))
            ! Update the right hand side
            Q = source(g, an, c)
            do gp = 1, number_groups
              Q = Q + chi(g, mMap(c)) * vsig_f(gp, mMap(c)) * phi_old(0, gp, c)
              do l = 0, number_legendre
                Q = Q + (2 * l + 1) * p_leg(l, an) * sig_s(l, g, gp, mMap(c)) * phi_old(l, gp, c)
              end do
            end do
            
            call computeEQ(Q, incoming(an), sig_t(g, mMap(c)), invmu, incoming(an), Ps)
            
            if (store_psi) then
              psi(g,an,c) = Ps
            end if
            
            do l = 0, number_legendre
              phi(l,g,c) = phi(l,g,c) + 0.5 * wt(a) * p_leg(l, an) * Ps
            end do
          end do
        end do
      end do
    end do
  end subroutine sweep
  
  subroutine computeEQ(Q, incoming, sig, invmu, outgoing, Ps)
    implicit none
    double precision, intent(in) :: Q, incoming, sig, invmu
    double precision, intent(out) :: outgoing, Ps
    
    if (equation .eq. 'DD') then
      Ps = (incoming + invmu * Q) / (1 + invmu * sig)
      outgoing = 2 * Ps - incoming
    end if
  
  end subroutine computeEQ
  
end module sweeper
