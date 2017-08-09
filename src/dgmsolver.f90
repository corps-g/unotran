module dgmsolver
  use control
  use material, only : create_material, number_legendre, number_groups, finalize_material
  use angle, only : initialize_angle, p_leg, number_angles, initialize_polynomials, finalize_angle
  use mesh, only : create_mesh, number_cells, finalize_mesh
  use state, only : initialize_state, phi, source, psi, finalize_state, output_state
  use dgm, only : number_course_groups, initialize_moments, initialize_basis, finalize_moments, expansion_order
  use dgmsweeper, only : sweep

  implicit none
  
  logical :: printOption, use_fission
  double precision, allocatable :: incoming(:,:,:)

  contains
  
  ! Initialize all of the variables and solvers
  subroutine initialize_dgmsolver()
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

    allocate(incoming(number_course_groups, number_angles * 2, 0:expansion_order))

    incoming  = 0.0

  end subroutine initialize_dgmsolver

  ! Interate equations until convergance
  subroutine dgmsolve()
    double precision :: norm, outer_error, hold
    double precision :: phi_new(0:number_legendre,number_groups,number_cells), psi_new(number_groups,2*number_angles,number_cells)
    integer :: counter

    ! Error of current iteration
    outer_error = 1.0
    ! 2 norm of the scalar flux
    norm = norm2(phi)
    ! interation number
    counter = 1
    do while (outer_error > outer_tolerance)  ! Interate to convergance tolerance
      ! Sweep through the mesh
      call sweep(phi_new, psi_new, incoming)
      ! Store norm of scalar flux
      hold = norm2(phi_new)
      ! error is the difference in the norm of phi for successive iterations
      outer_error = abs(norm - hold)
      ! Keep the norm for the next iteration
      norm = hold
      ! output the current error and iteration number
      if (outer_print) then
        print *, outer_error, counter
      end if
      ! increment the iteration
      counter = counter + 1
      phi = (1.0 - lambda) * phi + lambda * phi_new
      psi = (1.0 - lambda) * psi + lambda * psi_new
    end do



  end subroutine dgmsolve

  subroutine dgmoutput()

    call output_state()

  end subroutine dgmoutput

  subroutine finalize_dgmsolver()
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
