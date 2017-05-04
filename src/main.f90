program main
  use solver

  implicit none
  
  ! initialize types
  integer :: l, c, a, g, n
  ! fineMesh : vector of int for number of fine mesh divisions per cell
  ! materialMap : vector of int with material index for each course region
  integer, allocatable :: fm(:), mm(:), em(:)
  ! courseMap : vector of float with bounds for course mesh regions
  double precision, allocatable :: cm(:)
  
  ! Define file for cross sections
  character(len=10) :: fileName = 'test.anlxs'
  character(len=5) :: basisName = 'basis'
  
  ! define energy map
  allocate(em(1))
  em(1) = 4

  ! number of pins in model
  n = 10
  
  ! fill the mesh containers for the n-pin problem
  allocate(fm(3*n), cm(3*n+1), mm(3*n))
  
  ! initialize the mesh for the problem using mesh containers
  call get_mesh(n, fm, cm, mm)
  
  ! initialize the variables necessary to solve the problem
  call initialize_solver(fineMesh=fm, courseMesh=cm, materialMap=mm, fileName=fileName, &
                         angle_order=10, angle_option=1, energyMap=em, basisName=basisName)

  ! add source to all cells in 1st energy group
  source(:,:,:) = 1.0

  ! call the solver with error tolerance
  call solve(1e-8_8)

  ! output the resulting scalar flux
  !print *, phi(0,:,:)
  
end program main

! Fill the mesh containers for a 2 material, n-pin problem
subroutine get_mesh(n, fm, cm, mt)
  implicit none
  ! number of pins
  integer, intent(in) :: n
  ! fine mesh container, material map container
  integer, intent(out) :: fm(3*n), mt(3*n)
  ! course mesh container
  double precision, intent(out) :: cm(3*n+1)
  ! Divisions for a course mesh pin cell
  double precision :: cm_pin(3)
  integer :: fm_pin(3), i, j
  
  ! Define the "pin" coarse mesh edges.
  cm_pin = [0.09, 1.17, 1.26]
  
  ! Define the base fine mesh count per coarse mesh.
  fm_pin = [3, 22, 3]
  
  ! Build the mesh containers
  cm(:) = 0.0
  do i = 1, n  ! loop through pins
    do j = 1, 3  ! loop through pin course mesh edges
      cm(3 * (i-1) + j + 1) = cm_pin(j) + 1.26 * (i-1)
      fm(3 * (i-1) + j) = fm_pin(j)
    end do
  end do
  
  ! If only 1 cell use only UO2, else have half UO2 and half MOX
  if (n .eq. 1) then
    mt = [6, 1, 6]
  else
    ! fill with UO2
    do i = 1, n / 2
      mt(3 * (i - 1) + 1) = 6
      mt(3 * (i - 1) + 2) = 1
      mt(3 * (i - 1) + 3) = 6
    end do
    ! fill with MOX
    do i = n / 2 + 1, n
      mt(3 * (i - 1) + 1) = 6
      mt(3 * (i - 1) + 2) = 3
      mt(3 * (i - 1) + 3) = 6
    end do
  end if
end subroutine get_mesh
