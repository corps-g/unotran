module dgmsweeper
  ! Uses diamond difference
  ! Assumes only vacuum conditions
  use material, only: sig_s, sig_t, vsig_f, chi, number_groups, number_legendre
  use mesh, only: dx, number_cells, mMap
  use angle, only: number_angles, p_leg, wt, mu
  use state, only: phi, psi, source, store_psi, equation
  use sweeper, only : computeEQ
  use dgm

  implicit none
  
  contains
  ! Define source
  
  subroutine dgmsweep(lambda)
    integer :: o, c, a, g, gp, l, an, cmin, cmax, cstep, amin, amax, astep, cg, i
    double precision :: incoming(expansion_order,number_groups,2*number_angles), Q(expansion_order,number_groups), invmu
    double precision :: phi_old(0:number_legendre,number_groups,number_cells)
    double precision :: delta, num, M(0:number_legendre), Ps(expansion_order, number_course_groups)
    double precision, intent(in) :: lambda
    logical :: octant
    
    phi = (1.0 - lambda) * phi  ! Reset phi
    psi = (1.0 - lambda) * psi  ! Reset psi

    do o = 1, 2  ! Sweep over octants
      ! Sweep in the correct direction in the octant
      octant = o .eq. 1
      ! merge means : num = v1 if condition else v2
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
          
          ! legendre polynomial integration vector
          M = 0.5 * wt(a) * p_leg(:,an)

          ! Update the right hand side
          Q = updateSource(c, an)
          do cg = 1, number_course_groups  ! sweep oper course group
            do i = 1, order(cg)
              ! compute psi_moment for given order and course group
              ! Use the specified equation.  Defaults to DD
              ! fill Ps with the cell angular flux
              call computeEQ(Q(i,cg), incoming(i,cg,an), sig_t_moment(cg, c), invmu, incoming(i,cg,an), Ps(i,cg))
            end do
          end do

          do g = 1, number_groups  ! Sweep over group
            cg = energyMesh(g)  ! get course group
            num = 0.0
            do i = 1, order(cg)  ! sweep over expansion order
              num = num + lambda * basis(i,g) * Ps(i,cg)
            end do
            ! compute the angular flux
            psi(g,an,c) = psi(g,an,c) + num

            ! Increment the legendre expansions of the scalar flux
            phi(:,g,c) = phi(:,g,c) + M(:) * num
          end do
        end do
      end do
    end do

  end subroutine dgmsweep
  
  ! Compute the updated source term
  function updateSource(cell, angle)
    double precision :: updateSource(expansion_order,number_course_groups)
    double precision :: num
    integer, intent(in) :: cell, angle
    integer :: l, i, cg, cgp
    
    ! Include the external source and the fission source
    ! Add the scattering source for each legendre moment
    do cg = 1, number_course_groups
      updateSource(:,cg) = source_moment(:,cg,angle,cell) - delta_moment(:,cg,angle,cell)
      do cgp = 1, number_course_groups
        do i = 1, order(cg)
          num = 0.0
          do l = 0, number_legendre
            num = num + (2 * l + 1) * p_leg(l, angle) * sig_s_moment(l, i, cgp, cg, cell)
          end do
          updateSource(i,cg) = updateSource(i,cg) + num
        end do
      end do
    end do


  end function updateSource
  
end module dgmsweeper
