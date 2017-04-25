module solver
  use material, only : create_material, number_legendre, number_groups
  use angle, only : initialize_angle, p_leg, number_angles, initialize_polynomials
  use mesh, only : create_mesh, number_cells
  use state, only : initialize_state, phi, source
  use sweeper, only : sweep
  use dgm, only : initialize_moments, initialize_basis

  implicit none
  
  contains
  
  ! Initialize all of the variables and solvers
  subroutine initialize_solver(fineMesh, courseMesh, materialMap, fileName, angle_order, &
                               angle_option, store, EQ, energyMap, basisName)
    ! Inputs :
    !   fineMesh : vector of int for number of fine mesh divisions per cell
    !   courseMap : vector of float with bounds for course mesh regions
    !   materialMap : vector of int with material index for each course region
    !   fileName : file where cross sections are stored
    !   angle_order : number of angles per octant
    !   angle_option : type of quadrature, defined in angle.f90
    !   store (optional) : boolian for option to store angular flux
    !   EQ (optional) : Define the type of closure relation used.  default is DD
    !   energyMap (optional) : Required if using DGM.  Sets the course group struc.

    integer, intent(in) :: fineMesh(:), materialMap(:), angle_order, angle_option
    double precision, intent(in) :: courseMesh(:)
    character(len=*), intent(in) :: fileName
    logical, intent(in), optional :: store
    character(len=2), intent(in), optional :: EQ
    logical :: store_psi, useDGM
    character(len=2) :: equation
    integer, intent(in), optional :: energyMap(:)
    character(len=*), intent(in) :: basisName
    
    ! Check if the optional argument store is given
    if (present(store)) then
      store_psi = store  ! Set the option to the given parameter
    else
      store_psi = .false.  ! Default to not storing the angular flux
    end if
    
    ! Check if the optional argument EQ is given
    if (present(EQ)) then
      equation = EQ  ! Set the option to the given parameter
    else
      equation = 'DD'  ! Default to diamond difference
    end if
    
    ! initialize the mesh
    call create_mesh(fineMesh, courseMesh, materialMap)
    ! read the material cross sections
    call create_material(filename)
    ! initialize the angle quadrature
    call initialize_angle(angle_order, angle_option)
    ! get the basis vectors
    call initialize_polynomials(number_legendre)
    ! allocate the solutions variables
    call initialize_state(store_psi, equation)

    ! make the energy mesh if energyMap was given
    if (present(energyMap)) then
      useDGM = .true.
      call initialize_moments(energyMap)
      call initialize_basis(basisName)
    else
      useDGM = .false.
    end if

  
  end subroutine initialize_solver

  ! Interate equations until convergance
  subroutine solve(eps)
    ! Inputs
    !   eps : error tolerance for convergance

    double precision, intent(in) :: eps
    double precision :: norm, error, hold
    integer :: counter

    ! Error of current iteration
    error = 1.0
    ! 2 norm of the scalar flux
    norm = 0.0
    ! interation number
    counter = 1
    do while (error .gt. eps)  ! Interate to convergance tolerance
      ! Sweep through the mesh
      call sweep()
      ! Store norm of scalar flux
      hold = norm2(phi)
      ! error is the difference in the norm of phi for successive iterations
      error = abs(norm - hold)
      ! Keep the norm for the next iteration
      norm = hold
      ! output the current error and iteration number
      print *, error, counter
      ! increment the iteration
      counter = counter + 1
    end do

  end subroutine solve

end module solver
