module dgmsweeper
  ! Uses diamond difference
  ! Assumes only vacuum conditions
  use material, only: sig_s, sig_t, vsig_f, chi, number_groups, number_legendre
  use mesh, only: dx, number_cells, mMap
  use angle, only: number_angles, p_leg, wt, mu
  use state, only: phi, psi, source, store_psi, equation
  use dgm

  implicit none
  
  contains
  ! Define source
  
  subroutine dgmsweep()
    integer :: o, c, a, g, gp, l, an, cmin, cmax, cstep, amin, amax, astep, cg, i
    double precision :: incoming(number_groups,expansion_order,2*number_angles), Q(number_groups, expansion_order), Ps, invmu
    double precision :: phi_old(0:number_legendre,number_groups,number_cells)
    double precision :: S(number_course_groups,expansion_order), delta
    logical :: octant
    
    phi_old = phi
    phi = 0.0  ! Reset phi
    if (store_psi) then
      psi = 0.0
    end if

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
          Q = updateSource(c, an)

          do i = 1, expansion_order
            do g = 1, number_groups  ! Sweep over group
              cg = energyMesh(g)
              if (i > order(cg)) then
                cycle
              end if
              ! Use the specified equation.  Defaults to DD
              delta = delta_moment(cg,i,an,c) * psi_0_moment(cg, an, c)
              call computeEQ(Q(cg,i), incoming(cg,i,an), sig_t_moment(cg, mMap(c)), delta, invmu, incoming(cg,i,an), Ps)
            
              if (store_psi) then
                psi(g,an,c) = psi(g,an,c) + basis(g,i) * Ps
              end if
            
              ! Increment the legendre expansions of the scalar flux
              phi(:,g,c) = phi(:,g,c) + 0.5 * wt(a) * p_leg(:, an) * basis(g,i) * Ps
            end do
          end do
        end do
      end do
    end do
  end subroutine dgmsweep
  
  subroutine computeEQ(Qg, incoming, sig, delta, invmu, outgoing, Ps)
    implicit none
    double precision, intent(in) :: Qg, incoming, sig, invmu, delta
    double precision, intent(out) :: outgoing, Ps
    
    if (equation .eq. 'DD') then
      ! Diamond Difference relationship
      Ps = (incoming + invmu * (Qg - delta)) / (1 + invmu * sig)
      outgoing = 2 * Ps - incoming
    end if
  
  end subroutine computeEQ
  

  ! Compute the updated source term
  function updateSource(cell, angle)
    double precision :: updateSource(number_course_groups,expansion_order)
    double precision :: num
    integer, intent(in) :: cell, angle
    integer :: l, i, g, gp, cg
    
    ! Include the external source and the fission source
    updateSource(:,:) = source_moment(:,:,angle,cell)
    
    ! Add the scattering source for each legendre moment
    do i = 1, expansion_order
      do g = 1, number_course_groups
        cg = energyMesh(g)
        if (i > order(cg)) then
          cycle
        end if
        do l = 0, number_legendre
          do gp = 1, number_course_groups
            num = (2 * l + 1) * p_leg(l, angle) * sig_s_moment(l, gp, g, i, cell) * phi_moment(l, gp, i, cell)
            updateSource(g,i) = updateSource(g,i) + num
          end do
        end do
      end do
    end do


  end function updateSource
  
end module dgmsweeper
