module dgmsolver
  ! ############################################################################
  ! Solve discrete ordinates using the Discrete Generalized Multigroup Method
  ! ############################################################################

  use control
  use material, only : create_material, number_legendre, &
                       number_groups, finalize_material
  use angle, only : initialize_angle, p_leg, number_angles, &
                    initialize_polynomials, finalize_angle
  use mesh, only : create_mesh, number_cells, finalize_mesh
  use state, only : initialize_state, phi, source, psi, finalize_state, &
                    output_state, d_keff
  use dgm, only : number_coarse_groups, initialize_moments, initialize_basis, &
                  finalize_moments, expansion_order, compute_source_moments
  use dgmsweeper, only : dgmsweep

  implicit none
  
  double precision, allocatable, dimension(:,:,:) :: &
      incoming ! Angular flux incident on a cell

  contains
  
  subroutine initialize_dgmsolver()
    ! ##########################################################################
    ! Initialize all of the variables and solvers
    ! ##########################################################################

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

    allocate(incoming(number_coarse_groups, number_angles, 0:expansion_order))

    incoming = 0.0

  end subroutine initialize_dgmsolver

  subroutine dgmsolve()
    ! ##########################################################################
    ! Interate DGM equations to convergance
    ! ##########################################################################

    double precision :: &
        outer_error ! Error between successive iterations
    double precision, allocatable, dimension(:,:,:) :: &
        phi_new,  & ! Scalar flux for current iteration
        psi_new     ! Angular flux for current iteration
    integer :: &
        counter     ! Iteration counter

    ! Initialize the flux containers
    allocate(phi_new(0:number_legendre, number_groups, number_cells))
    allocate(psi_new(number_groups, 2 * number_angles, number_cells))

    ! Error of current iteration
    outer_error = 1.0

    ! interation number
    counter = 1

    ! Interate to convergance
    do while (outer_error > outer_tolerance)
      ! Sweep through the mesh
      call dgmsweep(phi_new, psi_new, incoming)

      ! error is the difference in phi between successive iterations
      outer_error = sum(abs(phi - phi_new))

      ! output the current error and iteration number
      if (outer_print) then
        if (solver_type == 'eigen') then
          print *, 'OuterError = ', outer_error, 'Iteration = ', counter, &
                   'Eigenvalue = ', d_keff
        else if (solver_type == 'fixed') then
          print *, 'OuterError = ', outer_error, 'Iteration = ', counter
        end if
      end if

      ! Update flux using krasnoselskii iteration
      phi = (1.0 - lamb) * phi + lamb * phi_new
      psi = (1.0 - lamb) * psi + lamb * psi_new

      ! increment the iteration
      counter = counter + 1

      ! Break out of loop if exceeding maximum outer iterations
      if (counter > max_outer_iters) then
        if (.not. ignore_warnings) then
          print *, 'warning: exceeded maximum outer iterations'
        end if
        exit
      end if

    end do

  end subroutine dgmsolve

  subroutine dgmoutput()
    ! ##########################################################################
    ! Save the output for the fluxes to file
    ! ##########################################################################

    call output_state()

  end subroutine dgmoutput

  subroutine finalize_dgmsolver()
    ! ##########################################################################
    ! Deallocate used arrays
    ! ##########################################################################

    call finalize_angle()
    call finalize_mesh()
    call finalize_material()
    call finalize_state()
    call finalize_moments()
    if (allocated(incoming)) then
      deallocate(incoming)
    end if
  end subroutine finalize_dgmsolver

end module dgmsolver
