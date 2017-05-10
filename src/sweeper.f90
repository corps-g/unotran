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
    double precision :: incoming(number_groups,2*number_angles), Q(number_groups), Ps, invmu, fiss
    double precision :: phi_old(0:number_legendre,number_groups,number_cells)
    logical :: octant
    
    phi_old = phi
    phi = 0.0  ! Reset phi
    do o = 1, 2  ! Sweep over octants
      ! Sweep in the correct direction in the octant
      octant = o .eq. 1
      cmin = merge(1, number_cells, octant)
      cmax = merge(number_cells, 1, octant)
      cstep = merge(1, -1, octant)
      amin = merge(1, number_angles, octant)
      amax = merge(number_angles, 1, octant)
      astep = merge(1, -1, octant)
      
      
      incoming = 0.0  ! Set vacuum conditions for both faces
      do c = cmin, cmax, cstep  ! Sweep over cells
        do a = amin, amax, astep  ! Sweep over angle
          ! Get the correct angle index
          an = merge(a, 2 * number_angles - a + 1, octant)
          ! get a common fraction
          invmu = dx(c) / (2 * abs(mu(a)))
          
          ! Update the right hand side
          Q = updateSource(source(:,an,c), phi_old(:,:,c), c, an)
          
          do g = 1, number_groups  ! Sweep over group
            ! Use the specified equation.  Defaults to DD
            call computeEQ(Q(g), incoming(g,an), sig_t(g, mMap(c)), invmu, incoming(g,an), Ps)
            
            if (store_psi) then
              psi(g,an,c) = Ps
            end if
            
            ! Increment the legendre expansions of the scalar flux
            phi(:,g,c) = phi(:,g,c) + 0.5 * wt(a) * p_leg(:, an) * Ps
          end do
        end do
      end do
    end do
  end subroutine sweep
  
  subroutine computeEQ(Qg, incoming, sig, invmu, outgoing, cellPsi)
    implicit none
    double precision, intent(in) :: Qg, incoming, sig, invmu
    double precision, intent(out) :: outgoing, cellPsi
    
    if (equation .eq. 'DD') then
      ! Diamond Difference relationship
      cellPsi = (incoming + invmu * Qg) / (1 + invmu * sig)
      outgoing = 2 * cellPsi - incoming
    else
      print *, 'ERROR : Equation not implemented'
    end if
  
  end subroutine computeEQ
  
  function updateSource(Sg, Phig, cell, angle)
    double precision :: updateSource(number_groups)
    double precision, intent(in) :: Sg(number_groups), Phig(0:number_legendre, number_groups)
    double precision :: scat(number_groups)
    integer, intent(in) :: cell, angle
    integer :: l
    
    ! Include the external source and the fission source
    updateSource(:) = Sg(:) + chi(:, mMap(cell)) * dot_product(vsig_f(:, mMap(cell)), phig(0,:))
    
    ! Add the scattering source for each legendre moment
    do l = 0, number_legendre
      scat(:) = (2 * l + 1) * p_leg(l, angle) * matmul(sig_s(l, :, :, mMap(cell)), phig(l,:))
      updateSource(:) = updateSource(:) + scat(:)
    end do
    
  end function updateSource
  
end module sweeper
