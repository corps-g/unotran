module dgmsweeper
  use control, only : boundary_type, inner_print, inner_tolerance, lambda, use_recondensation, store_psi
  use material, only : number_groups, number_legendre
  use mesh, only : dx, number_cells, mMap
  use angle, only : number_angles
  use sweeper, only : sweep
  use state, only : d_source, d_nu_sig_f, d_chi, d_sig_s, d_phi, d_delta, d_sig_t, d_psi
  use dgm, only : number_course_groups, expansion_order, &
                  energymesh, basis, compute_xs_moments, compute_flux_moments

  implicit none
  
  contains

  subroutine dgmsweep(phi_new, psi_new, incoming)
    double precision, intent(inout) :: incoming(number_course_groups,2 * number_angles,0:expansion_order)
    double precision, intent(inout) :: phi_new(:,:,:), psi_new(:,:,:)
    double precision :: norm, inner_error, hold
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
      call compute_xs_moments(order=i)

      ! Converge the 0th order flux moments
      do while (inner_error > inner_tolerance)  ! Interate to convergance tolerance
        ! Sweep through the mesh

        ! Use discrete ordinates to sweep over the moment equation
        call sweep(number_course_groups, phi_m, psi_m, incoming(:,:,i))

        ! Store norm of scalar flux
        norm = norm2(phi_m)
        ! error is the difference in the norm of phi for successive iterations
        inner_error = abs(norm - hold)
        ! Keep the norm for the next iteration
        hold = norm
        ! output the current error and iteration number
        if (inner_print) then
          print *, '    ', 'eps = ', inner_error, ' counter = ', counter, ' order = ', i, phi_m(0,:,1)
        end if
        ! increment the iteration
        counter = counter + 1

        ! Update the 0th order moments if working on converging zeroth moment
        if (i == 0) then
          !d_phi = (1.0 - lambda) * d_phi + lambda * phi_m
          !d_psi = (1.0 - lambda) * d_psi + lambda * psi_m
          d_phi = phi_m
          d_psi = psi_m
        end if

        if (i > 0) then
          exit
        end if

        ! If recondensation is active, break out of loop early
        if (use_recondensation) then
          call compute_xs_moments(order=i)
        end if

      end do

      ! Unfold 0th order flux
      call unfold_flux_moments(i, phi_m, psi_m, phi_new, psi_new)

    end do

    deallocate(phi_m, psi_m)
  end subroutine dgmsweep

  ! Unfold the flux moments
  subroutine unfold_flux_moments(order, phi_moment, psi_moment, phi_new, psi_new)
    integer, intent(in) :: order
    double precision, intent(in) :: phi_moment(:,:,:), psi_moment(:,:,:)
    double precision, intent(out) :: psi_new(:,:,:), phi_new(:,:,:)
    integer :: a, c, cg, g, mat

    do c = 1, number_cells
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
  
end module dgmsweeper
