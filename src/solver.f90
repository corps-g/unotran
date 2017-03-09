module solver
  !use material, only : create_material, number_legendre, number_groups
  !use angle, only : initialize_angle, p_leg, number_angles, initialize_polynomials
  !use mesh, only : create_mesh, number_cells
  !use state, only : initialize_state, phi, source
  !use sweeper, only : sweep
  use material
  use angle
  use mesh
  use state
  use sweeper
  implicit none
  !f2py intent(out) phi
  
  save
  
  contains
  
  subroutine initialize_solver(fineMesh, courseMesh, materialMap, fileName, angle_order, &
                               angle_option, store, EQ)
    integer, intent(in) :: fineMesh(:), materialMap(:), angle_order, angle_option
    double precision, intent(in) :: courseMesh(:)
    character(len=*), intent(in) :: fileName
    logical, intent(in), optional :: store
    character(len=2), intent(in), optional :: EQ
    logical :: store_psi
    character(len=2) :: equation
    
    if (present(store)) then
      store_psi = store  ! Set the option to the given parameter
    else
      store_psi = .false.  ! Default to not storing the angular flux
    end if
    
    ! Check if the optional argument EQ is given
    if (present(EQ)) then
      equation = EQ  ! Set the option to the given parameter
    else
      equation = 'DD'  ! Default to not storing the angular flux
    end if
    
    call create_mesh(fineMesh, courseMesh, materialMap)
    call create_material(filename)
    call initialize_angle(angle_order, angle_option)
    call initialize_polynomials(number_legendre)
    call initialize_state(store_psi, equation)
  
  end subroutine initialize_solver

end module solver
