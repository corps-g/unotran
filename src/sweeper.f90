module sweeper
  ! Uses diamond difference
  ! Assumes only vacuum conditions
  use material, only: sig_s, sig_t, vsig_f, chi, number_groups, number_legendre
  use mesh, only: dx, number_cells, mMap
  use angle, only: number_angles, p_leg, wt, mu
  use state, only: phi, psi, source, phistar, internal_source
  implicit none
  
  contains
  ! Define source
  
  subroutine sweep()
    integer :: c, a, g, l, an
    double precision :: incoming, scat, invmu
    
    ! get the source array
    call update_source()
    
    incoming = 0.0
    ! Sweep in positive mu direction
    do c = 1, number_cells
      do a = 1, number_angles
        do g = 1, number_groups
          invmu = dx(c) / (2 * abs(mu(a)))
          scat = 0.0
          do l = 0, number_legendre
            scat = scat + (2 * l + 1) * p_leg(l, a) * sig_s(mMap(c), l, g, g) * phistar(c, l, g)
          end do
          !psi(c, a, g) = (incoming + invmu * (scat + source(c, a, g))) / (1 + sig_t(c, g) * invmu)
          phi(c, g) = phi(c,g) + wt(a)*(incoming+invmu*(scat+internal_source(c, a, g)))/(1+sig_t(mMap(c), g)*invmu)
          incoming = 2 * psi(c, a, g) - incoming
        end do
      end do
    end do
    
    incoming = 0.0
    ! Sweep in negative mu direction
    do c = number_cells, 1, -1
      do a = number_angles, 1, -1
        do g = 1, number_groups
          invmu = dx(c) / (2 * abs(mu(a)))
          scat = 0.0
          do l = 0, number_legendre
            scat = scat + (2 * l + 1) * p_leg(l, 2*number_angles - a + 1) * sig_s(mMap(c), l, g, g) * phistar(c, l, g)
          end do
          !psi(c, a, g) = (incoming + invmu * (scat + source(c, a, g))) / (1 + sig_t(c, g) * invmu)
          an = 2 * number_angles - a + 1
          phi(c, g) = phi(c,g) + wt(a) * (incoming + invmu * (scat + internal_source(c, an, g))) / (1 + sig_t(mMap(c), g) * invmu)
          incoming = 2 * psi(c, an, g) - incoming
        end do
      end do
    end do
        
  
  end subroutine sweep
  
  subroutine update_source()
    integer :: c, a, g, gp, l
    double precision :: out_scatter, fission
    call update_phistar()
  
    do c = 1, number_cells
      do a = 1, number_angles * 2
        do g = 1, number_groups
          out_scatter = 0.0
          fission = 0.0
          do gp = 1, number_groups
            ! Get the scattering source
            if (gp .ne. g) then  ! Only care about groups outside of the current group
              do l = 0, number_legendre
                out_scatter = out_scatter + (2 * l + 1) * p_leg(l, a) * sig_s(mMap(c), l, g, gp) * phistar(c, l, gp)
              end do
            end if
            ! Get the fission source
            fission = fission + vsig_f(mMap(c), g) * phi(c, g)
          end do
          fission = fission * chi(mMap(c), g)
          ! Save the right hand side of the equation
          internal_source(c, a, g) = out_scatter + fission + source(c, a, g)
        end do
      end do
    end do
  end subroutine update_source
  
  subroutine update_phistar()
    integer :: c, l, g, a, an
    do c = 1, number_cells
      do l = 0, number_legendre
        do g = 1, number_groups
          phistar(c, l, g) = 0.0
          do a = 1, number_angles
            ! for positive mu
            phistar(c, l, g) = phistar(c, l, g) + 0.5 * wt(a) * p_leg(l, a) * psi(c, a, g)
            ! for negative mu
            an = 2 * number_angles - a + 1
            phistar(c, l, g) = phistar(c, l, g) + 0.5 * wt(a) * p_leg(l, an) * psi(c, an, g)
          end do
        end do
      end do
    end do
  end subroutine update_phistar
  
end module sweeper
