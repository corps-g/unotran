module dgmsolver
  use material, only : create_material, number_legendre, number_groups, finalize_material
  use angle, only : initialize_angle, p_leg, number_angles, initialize_polynomials, finalize_angle
  use mesh, only : create_mesh, number_cells, finalize_mesh
  use state, only : initialize_state, phi, source, psi, finalize_state
  use dgm, only : number_course_groups, initialize_moments, initialize_basis, finalize_moments
  use dgmsweeper, only : sweep

  implicit none
  
  logical :: printOption, use_fission
  double precision, allocatable :: incoming(:,:)

  contains
  
  ! Initialize all of the variables and solvers
  subroutine initialize_solver(fineMesh, courseMesh, materialMap, fileName, angle_order, &
                               angle_option, boundary, store, EQ, energyMap, basisName, &
                               truncation, print_level, fission_option)
    ! Inputs :
    !   fineMesh : vector of int for number of fine mesh divisions per cell
    !   courseMap : vector of float with bounds for course mesh regions
    !   materialMap : vector of int with material index for each course region
    !   fileName : file where cross sections are stored
    !   angle_order : number of angles per octant
    !   angle_option : type of quadrature, defined in angle.f90
    !   boundary : length 2 array containing boundary.  0=vacuum, 1=reflect.  values 0<=x<=1 are accepted.  [left, right]
    !   store (optional) : boolian for option to store angular flux
    !   EQ (optional) : Define the type of closure relation used.  default is DD
    !   energyMap (optional) : Required if using DGM.  Sets the course group struc.
    !   truncation (optional) : provides the expansion order for the dgm expansion.  full order assumed if not given
    !   silent (optional) : boolian that will show iteration prints or not

    integer, intent(in) :: fineMesh(:), materialMap(:), angle_order, angle_option
    double precision, intent(in) :: courseMesh(:), boundary(2)
    character(len=*), intent(in) :: fileName
    logical, intent(in), optional :: store
    character(len=2), intent(in), optional :: EQ
    logical :: store_psi
    character(len=2) :: equation
    integer, intent(in) :: energyMap(:)
    character(len=*), intent(in) :: basisName
    integer, intent(in), optional :: truncation(:)
    logical, intent(in), optional :: print_level
    logical, intent(in), optional :: fission_option
    
    store_psi = .true.
    
    ! Check if the optional argument EQ is given
    if (present(EQ)) then
      equation = EQ  ! Set the option to the given parameter
    else
      equation = 'DD'  ! Default to diamond difference
    end if
    
    ! Activate print statements depending on output flag
    if (present(print_level)) then
      printOption = print_level
    else
      printOption = .true.
    end if

    ! Deactivate fission if option is selected
    if (present(fission_option)) then
      use_fission = fission_option
    else
      use_fission = .true.
    end if

    ! initialize the mesh
    call create_mesh(fineMesh, courseMesh, materialMap, boundary)
    ! read the material cross sections
    call create_material(filename, use_fission)
    ! initialize the angle quadrature
    call initialize_angle(angle_order, angle_option)
    ! get the basis vectors
    call initialize_polynomials(number_legendre)
    ! allocate the solutions variables
    call initialize_state(store_psi, equation)

    ! make the energy mesh
    ! Pass the truncation array to dgm if provided
    if (present(truncation)) then
      call initialize_moments(energyMap, truncation)
    else
      call initialize_moments(energyMap)
    end if
    call initialize_basis(basisName)

    allocate(incoming(number_course_groups, number_angles * 2))

  end subroutine initialize_solver

  ! Interate equations until convergance
  subroutine solve(eps, lambda_arg)
    ! Inputs
    !   eps : error tolerance for convergance

    double precision, intent(in) :: eps
    double precision, intent(in), optional :: lambda_arg
    double precision :: norm, outer_error, hold, lambda
    integer :: counter

    if (present(lambda_arg)) then
      lambda = lambda_arg
    else
      lambda = 1.0
    end if

    ! Error of current iteration
    outer_error = 1.0
    ! 2 norm of the scalar flux
    norm = norm2(phi)
    ! interation number
    counter = 1
    do while (outer_error .gt. eps)  ! Interate to convergance tolerance
      ! Sweep through the mesh
      call sweep(phi, psi, incoming, eps, .true.)
      ! Store norm of scalar flux
      hold = norm2(phi)
      ! error is the difference in the norm of phi for successive iterations
      outer_error = abs(norm - hold)
      ! Keep the norm for the next iteration
      norm = hold
      ! output the current error and iteration number
      if (printOption) then
        print *, outer_error, counter
      end if
      ! increment the iteration
      counter = counter + 1

    end do

  end subroutine solve

  subroutine finalize_solver()
    call finalize_angle()
    call finalize_mesh()
    call finalize_material()
    call finalize_state()
    call finalize_moments()
    if (allocated(incoming)) then
      deallocate(incoming)
    end if
  end subroutine

end module dgmsolver
