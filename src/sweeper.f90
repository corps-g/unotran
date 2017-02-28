module sweeper
  ! Uses diamond difference
  ! Assumes only vacuum conditions
  use material, only: sig_s, sig_t, vsig_f, chi, number_groups, number_legendre
  use mesh, only: dx, number_cells, mMap
  use angle, only: number_angles
  use state, only: phi, psi, source, phistar, internal_source
  implicit none
  
  contains
  ! Define phistar, P, source
  
  subroutine sweep()
    ! Sweep in positive mu direction
    integer :: c, a, g, l
    double precision :: incoming = 0.0
    
    ! get the source array
    call update_source()
    
    do c = 1, number_cells
      do a = 1, number_angles
        do g = 1, number_groups
          invmu = dx(c) / (2 * abs(mu[a]))
          scat = 0.0
          do l = 0, number_legendre
            scat = scat + (2 * l + 1) * P(l, a) * sig_s(mMap(c), l, g, g) * phistar(c, l, g)
          end do
          !psi(c, a, g) = (incoming + invmu * (scat + source(c, a, g))) / (1 + sig_t(c, g) * invmu)
          phi(c, g) = phi(c,g) + w[a] * (incoming + invmu * (scat + internal_source(c, a, g))) / (1 + sig_t(mMap(c), g) * invmu)
          incoming = 2 * psi(c, a, g) - incoming
        end do
      end do
    end do
    
    incoming = 0.0
    ! Sweep in negative mu direction
    do c = number_cells, 1, -1
      do a = 1, number_angles
        do g = 1, number_groups
          invmu = dx(c) / (2 * abs(mu[a]))
          scat = 0.0
          do l = 0, nLegendre
            scat = scat + (2 * l + 1) * P(l, a) * sig_s(mMap(c), l, g, g) * phistar(c, l, g)
          end do
          !psi(c, a, g) = (incoming + invmu * (scat + source(c, a, g))) / (1 + sig_t(c, g) * invmu)
          phi(c, g) = phi(c,g) + w[a] * (incoming + invmu * (scat + internal_source(c, a, g))) / (1 + sig_t(mMap(c), g) * invmu)
          incoming = 2 * psi(c, a, g) - incoming
        end do
      end do
    end do
        
  
  end subroutine sweep
  
  subroutine update_source()
    double precision :: out_scatter, fission
  
    do c = 1, number_cells
      do a = 1, number_angles
        do g = 1, number_groups
          out_scatter = 0.0
          fission = 0.0
          do gp = 1, number_groups
            ! Get the scattering source
            if gp /= g  ! Only care about groups outside of the current group
              do l = 0, number_legendre
                out_scatter = out_scatter + (2 * l + 1) * P(l, a) * sig_s(mMap(c), l, g, gp) * phistar(c, l, gp)
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
  
end module sweeper
