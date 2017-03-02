module state
  use material, only: number_groups, number_legendre
  use mesh, only: number_cells
  use angle, only: number_angles
  implicit none
  double precision, allocatable, dimension(:,:) :: phi
  double precision, allocatable, dimension(:,:,:) :: psi, source, phistar, internal_source
  
  save
  
  contains
  
  subroutine initialize_state()
    allocate(phi(number_cells,number_groups))
    allocate(psi(number_cells,number_angles*2,number_groups))
    allocate(source(number_cells,number_angles*2,number_groups))
    allocate(phistar(number_cells,number_legendre,number_groups))
    allocate(internal_source(number_cells,number_angles*2,number_groups))
    
    phi = 0.0
    psi = 0.0
    source = 0.0
    phistar = 0.0
    internal_source = 0.0
    
  end subroutine initialize_state
  
end module state
