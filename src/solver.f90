module solver
  use control, only : lambda, outer_print, outer_tolerance, store_psi, solver_type
  use material, only : create_material, number_legendre, number_groups, finalize_material, &
                       sig_t, sig_s, nu_sig_f, chi
  use angle, only : initialize_angle, p_leg, number_angles, initialize_polynomials, finalize_angle
  use mesh, only : create_mesh, number_cells, finalize_mesh, mMap
  use state, only : initialize_state, phi, psi, source, finalize_state, output_state, &
                    d_source, d_nu_sig_f, d_delta, d_phi, d_chi, d_sig_s, d_sig_t, d_psi, d_keff
  use sweeper, only : sweep

  implicit none
  
  logical :: printOption, use_fission
  double precision, allocatable :: incoming(:,:)

  contains
  
  ! Initialize all of the variables and solvers
  subroutine initialize_solver()
    integer :: c

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

    ! Fill container arrays
    do c = 1, number_cells
      d_nu_sig_f(:, c) = nu_sig_f(:, mMap(c))
      d_chi(:, c) = chi(:, mMap(c))
      d_sig_s(:, :, :, c) = sig_s(:, :, :, mMap(c))
      d_sig_t(:, c) = sig_t(:, mMap(c))
    end do
    d_source(:, :, :) = source(:, :, :)
    d_delta(:, :, :) = 0.0
    d_phi(:, :, :) = phi(:, :, :)
    if (store_psi) then
      d_psi(:, :, :) = psi(:, :, :)
    end if

    ! todo: move to state
    allocate(incoming(number_groups, number_angles * 2))
    incoming = 0.0

  end subroutine initialize_solver

  ! Interate equations until convergance
  subroutine solve()
    double precision :: norm, error, hold
    integer :: counter

    ! Error of current iteration
    error = 1.0

    ! interation number
    counter = 1

    do while (error > outer_tolerance)  ! Interate to convergance tolerance
      ! save phi from previous iteration
      d_phi = phi

      ! Sweep through the mesh
      call sweep(number_groups, phi, psi, incoming)

      ! error is the difference in phi between successive iterations
      error = sum(abs(phi - d_phi))

      ! Compute new eigenvalue if eigen problem
      if (solver_type == 'eigen') then
        d_keff = d_keff * sum(abs(phi)) / sum(abs(d_phi))
      end if

      ! output the current error and iteration number
      if (outer_print) then
        if (solver_type == 'eigen') then
          print *, 'Error = ', error, 'Iteration = ', counter, 'Eigenvalue = ', d_keff
        else if (solver_type == 'fixed') then
          print *, 'Error = ', error, 'Iteration = ', counter
        end if
      end if

      ! increment the iteration
      counter = counter + 1
    end do

  end subroutine solve

  subroutine output()

    call output_state()

  end subroutine output

  subroutine finalize_solver()
    call finalize_angle()
    call finalize_mesh()
    call finalize_material()
    call finalize_state()
    if (allocated(incoming)) then
      deallocate(incoming)
    end if
  end subroutine

end module solver
