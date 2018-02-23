module dgmsolver
  ! ############################################################################
  ! Solve discrete ordinates using the Discrete Generalized Multigroup Method
  ! ############################################################################

!  use control
!  use material, only : create_material, number_legendre, &
!                       number_groups, finalize_material
!  use angle, only : initialize_angle, p_leg, number_angles, &
!                    initialize_polynomials, finalize_angle
!  use mesh, only : create_mesh, number_cells, finalize_mesh
!  use state, only : initialize_state, phi, source, psi, finalize_state, &
!                    output_state, d_keff
!  use dgm, only : number_coarse_groups, initialize_moments, initialize_basis, &
!                  finalize_moments, expansion_order, compute_source_moments
!  use dgmsweeper, only : dgmsweep

  implicit none
  
  contains
  
  subroutine initialize_dgmsolver()
    ! ##########################################################################
    ! Initialize all of the variables and solvers
    ! ##########################################################################

    ! Use Statements
    use mesh, only : create_mesh
    use material, only : create_material, number_legendre
    use angle, only : initialize_angle, initialize_polynomials
    use state, only : initialize_state
    use dgm, only : initialize_moments, initialize_basis, compute_source_moments

    ! initialize the mesh
    call create_mesh()
    ! read the material cross sections
    call create_material()
    ! initialize the angle quadrature
    call initialize_angle()
    ! get the basis vectors
    call initialize_polynomials(number_legendre)
    ! allocate the solutions variables
    call initialize_state()
    ! Initialize DGM moments
    call initialize_moments()
    call initialize_basis()
    call compute_source_moments()

  end subroutine initialize_dgmsolver

  subroutine dgmsolve()
    ! ##########################################################################
    ! Interate DGM equations to convergance
    ! ##########################################################################

    use control, only : max_recon_iters

!    do recon_count = 1, max_recon_iters
!      ! Save the old value of the scalar flux
!      old_phi = phi
!
!      ! Expand the fluxes into moment form
!      call compute_flux_moments()
!      call compute_source_moments
!
!
!    end do


!    double precision :: &
!        outer_error ! Error between successive iterations
!    double precision, allocatable, dimension(:,:,:) :: &
!        phi_new,  & ! Scalar flux for current iteration
!        psi_new     ! Angular flux for current iteration
!    integer :: &
!        counter     ! Iteration counter
!
!    ! Initialize the flux containers
!    allocate(phi_new(0:number_legendre, number_groups, number_cells))
!    allocate(psi_new(number_groups, 2 * number_angles, number_cells))
!
!    ! Error of current iteration
!    outer_error = 1.0
!
!    ! interation number
!    counter = 1
!
!    ! Interate to convergance
!    do while (outer_error > outer_tolerance)
!      ! Sweep through the mesh
!      call dgmsweep(phi_new, psi_new)
!
!      ! error is the difference in phi between successive iterations
!      outer_error = sum(abs(phi - phi_new))
!
!      ! output the current error and iteration number
!      if (outer_print) then
!        if (solver_type == 'eigen') then
!          print *, 'OuterError = ', outer_error, 'Iteration = ', counter, &
!                   'Eigenvalue = ', d_keff
!        else if (solver_type == 'fixed') then
!          print *, 'OuterError = ', outer_error, 'Iteration = ', counter
!        end if
!      end if
!
!      ! Compute the eigenvalue
!      if (solver_type == 'eigen') then
!        ! Compute new eigenvalue if eigen problem
!        d_keff = d_keff * sum(abs(phi_new(0,:,:))) / sum(abs(phi(0,:,:)))
!      end if
!
!      ! Update flux using krasnoselskii iteration
!      phi = (1.0 - lamb) * phi + lamb * phi_new
!      psi = (1.0 - lamb) * psi + lamb * psi_new
!
!      ! increment the iteration
!      counter = counter + 1
!
!      ! Break out of loop if exceeding maximum outer iterations
!      if (counter > max_outer_iters) then
!        if (.not. ignore_warnings) then
!          print *, 'warning: exceeded maximum outer iterations'
!        end if
!        exit
!      end if
!
!    end do

  end subroutine dgmsolve

  subroutine unfold_flux_moments(order, psi_moment, phi_new, psi_new)
    ! ##########################################################################
    ! Unfold the fluxes from the moments
    ! ##########################################################################

    ! Use Statements
    use angle, only : p_leg, number_angles, wt
    use mesh, only : number_cells
    use material, only : number_groups, number_legendre
    use control, only : store_psi
    use dgm, only : energyMesh, basis

    ! Variable definitions
    integer, intent(in) :: &
        order         ! Expansion order
    double precision, intent(in), dimension(:,:,:) :: &
        psi_moment    ! Angular flux moments
    double precision, intent(inout), dimension(:,:,:) :: &
        psi_new,    & ! Scalar flux for current iteration
        phi_new       ! Angular flux for current iteration
    double precision, allocatable, dimension(:) :: &
        M                    ! Legendre polynomial integration vector
    integer :: &
        a,          & ! Angle index
        c,          & ! Cell index
        cg,         & ! Coarse group index
        g,          & ! Fine group index
        an            ! Global angle index
    double precision :: &
        val           ! Variable to hold a double value

    allocate(M(0:number_legendre))

    ! Recover the angular flux from moments
    do g = 1, number_groups
      ! Get the coarse group index
      cg = energyMesh(g)
      do a = 1, number_angles * 2
        ! legendre polynomial integration vector
        an = merge(a, 2 * number_angles - a + 1, a <= number_angles)
        M = wt(an) * p_leg(:,a)
        do c = 1, number_cells
          ! Unfold the moments
          val = basis(g, order) * psi_moment(c, a, cg)
          if (store_psi) then
            psi_new(c, a, g) = psi_new(c, a, g) + val
          end if
          phi_new(:, c, g) = phi_new(:, c, g) + M(:) * val
        end do
      end do
    end do

    deallocate(M)

  end subroutine unfold_flux_moments

  subroutine dgmoutput()
    ! ##########################################################################
    ! Save the output for the fluxes to file
    ! ##########################################################################

    ! Use Statements
    use state, only : phi, psi, output_state

    print *, phi
    print *, psi
    call output_state()

  end subroutine dgmoutput

  subroutine finalize_dgmsolver()
    ! ##########################################################################
    ! Deallocate used arrays
    ! ##########################################################################

    ! Use Statements
    use angle, only : finalize_angle
    use mesh, only : finalize_mesh
    use material, only : finalize_material
    use state, only : finalize_state
    use dgm, only : finalize_moments

    call finalize_angle()
    call finalize_mesh()
    call finalize_material()
    call finalize_state()
    call finalize_moments()

  end subroutine finalize_dgmsolver

end module dgmsolver
