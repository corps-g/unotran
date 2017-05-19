module dgmsweeper
  use material, only: number_groups, number_legendre
  use mesh, only: dx, number_cells, mMap, bounds
  use angle, only: number_angles, p_leg, wt, mu
  use state, only: store_psi, equation
  use sweeper, only : computeEQ, updateSource
  use dgm

  implicit none
  
  contains

  subroutine sweep(phi_new, psi_new, incoming, eps, print_option)
    double precision, intent(in) :: eps
    double precision, intent(inout) :: incoming(:,:), phi_new(:,:,:), psi_new(:,:,:)
    logical, intent(in) :: print_option
    double precision :: norm, inner_error, hold, lambda
    double precision, allocatable :: phi_m(:,:,:), psi_m(:,:,:)
    integer :: counter, i

    allocate(phi_m(0:number_legendre,number_course_groups,number_cells))
    allocate(psi_m(number_course_groups,number_angles*2,number_cells))
    phi_new = 0.0
    psi_new = 0.0
    phi_m = 0.0
    psi_m = 0.0

    ! Get the first guess for the flux moments
    call compute_flux_moments()


    do i = 0, expansion_order
      ! Initialize iteration variables
      inner_error = 1.0
      hold = 0.0
      counter = 1

      ! Compute the order=0 cross section moments
      call compute_xs_moments(order=0)

      ! Update the external source to include the delta term
      source_moment(:,:,:) = source_moment(:,:,:) + delta_moment(:,:,:) * psi_0_moment(:,:,:)

      ! Converge the 0th order flux moments
      do while (inner_error .gt. eps)  ! Interate to convergance tolerance
        ! Sweep through the mesh
        call moment_sweep(phi_m, psi_m, source_moment, incoming)

        ! Store norm of scalar flux
        norm = norm2(phi_m)
        ! error is the difference in the norm of phi for successive iterations
        inner_error = abs(norm - hold)
        ! Keep the norm for the next iteration
        hold = norm
        ! output the current error and iteration number
        if (print_option) then
          print *, inner_error, counter
        end if
        ! increment the iteration
        counter = counter + 1
      end do

      ! Update the 0th order moments if working on converging zeroth moment
      if (i == 0) then
        phi_0_moment = phi_m
        psi_0_moment = psi_m
      end if

      ! Unfold 0th order flux
      call unfold_flux_moments(i, phi_m, psi_m, phi_new, psi_new)
    end do

    deallocate(phi_m, psi_m)
  end subroutine sweep

  ! Unfold the flux moments
  subroutine unfold_flux_moments(order, phi_moment, psi_moment, phi_new, psi_new)
    integer, intent(in) :: order
    double precision, intent(in) :: phi_moment(:,:,:), psi_moment(:,:,:)
    double precision, intent(out) :: psi_new(:,:,:), phi_new(:,:,:)
    integer :: a, c, cg, g, mat

    do c = 1, number_cells
      ! get the material for the current cell
      mat = mMap(c)

      do a = 1, number_angles * 2
        do g = 1, number_groups
          cg = energyMesh(g)
          ! Scalar flux
          if (a == 1) then
            phi_new(:, g, c) = phi_new(:, g, c) + basis(g, order) * phi_moment(:, cg, c)
          end if
          ! Angular flux
          psi_new(g, a, c) = psi_new(g, a, c) +  basis(g, order) * psi_moment(cg, a, c)
        end do
      end do
    end do
  end subroutine unfold_flux_moments
  
  subroutine moment_sweep(phi_m, psi_m, source, incoming)
    integer :: o, c, a, g, gp, l, an, cmin, cmax, cstep, amin, amax, astep
    double precision :: Q(number_course_groups), Ps, invmu, fiss
    double precision :: phi_old(0:number_legendre,number_course_groups,number_cells), M(0:number_legendre)
    double precision, intent(inout) :: phi_m(:,:,:), source(:,:,:), incoming(:,:), psi_m(:,:,:)
    logical :: octant

    phi_old = phi_m
    phi_m = 0.0  ! Reset phi
    do o = 1, 2  ! Sweep over octants
      ! Sweep in the correct direction in the octant
      octant = o .eq. 1
      cmin = merge(1, number_cells, octant)
      cmax = merge(number_cells, 1, octant)
      cstep = merge(1, -1, octant)
      amin = merge(1, number_angles, octant)
      amax = merge(number_angles, 1, octant)
      astep = merge(1, -1, octant)

      ! set boundary conditions
      incoming = bounds(o) * incoming  ! Set albedo conditions

      do c = cmin, cmax, cstep  ! Sweep over cells
        do a = amin, amax, astep  ! Sweep over angle
          ! Get the correct angle index
          an = merge(a, 2 * number_angles - a + 1, octant)
          ! get a common fraction
          invmu = dx(c) / (2 * abs(mu(a)))

          ! legendre polynomial integration vector
          M = 0.5 * wt(a) * p_leg(:, an)

          ! Update the right hand side
          Q = updateSource(number_course_groups, source_moment(:, an, c), phi_0_moment(:,:,c), an, &
                           sig_s_moment(:,:,:,c), nu_sig_f_moment(:,c), chi_moment(:,c))

          do g = 1, number_course_groups  ! Sweep over group
            ! Use the specified equation.  Defaults to DD
            call computeEQ(Q(g), incoming(g, an), sig_t_moment(g, c), invmu, incoming(g, an), Ps)

            psi_m(g,an,c) = Ps

            ! Increment the legendre expansions of the scalar flux
            phi_m(:,g,c) = phi(:,g,c) + M(:) * Ps
          end do
        end do
      end do
    end do
  end subroutine moment_sweep
  
end module dgmsweeper
