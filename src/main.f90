program main
  use solver
  implicit none
  
  ! initialize types
  integer :: l, c, a, g, counter, n
  double precision :: norm, error
  integer, allocatable :: fineMesh(:), materialMap(:)
  double precision, allocatable :: courseMesh(:)
  
  ! Define problem parameters
  character(len=10) :: filename = 'test.anlxs'
  
  ! number of pins in model
  n = 10
  
  allocate(fineMesh(3*n), courseMesh(3*n+1), materialMap(3*n))
  
  call get_mesh(n, fineMesh, courseMesh, materialMap)
  
  call initialize_solver(fineMesh, courseMesh, materialMap,filename, 10, 1)

  !source = 1.0
  ! Source only in first group
  source(:,:,:) = 1.0

  error = 1.0
  norm = 0.0
  counter = 1
  do while (error .gt. 1e-8)
    call sweep()
    error = abs(norm - norm2(phi))
    norm = norm2(phi)
    print *, error, counter
    counter = counter + 1
  end do
  print *, phi(0,:,:)
  
end program main

subroutine get_mesh(n, fm, cm, mt)
  implicit none
  integer, intent(in) :: n
  integer, intent(out) :: fm(3*n), mt(3*n)
  double precision, intent(out) :: cm(3*n+1)
  double precision :: cm_pin(3)
  integer :: fm_pin(3), i, j
  
  ! Define the "pin coarse mesh edges.
  cm_pin = [0.09, 1.17, 1.26]
  
  ! Define the base fine mesh count per coarse mesh.
  fm_pin = [3, 22, 3]
  
  cm(:) = 0.0
  do i = 1, n
    do j = 1, 3
      cm(3 * (i-1) + j + 1) = cm_pin(j) + 1.26 * (i-1)
      fm(3 * (i-1) + j) = fm_pin(j)
    end do
  end do
  
  if (n .eq. 1) then
    mt = [6, 1, 6]
  else
    do i = 1, n / 2
      mt(3 * (i - 1) + 1) = 6
      mt(3 * (i - 1) + 2) = 1
      mt(3 * (i - 1) + 3) = 6
    end do
    do i = n / 2 + 1, n
      mt(3 * (i - 1) + 1) = 6
      mt(3 * (i - 1) + 2) = 3
      mt(3 * (i - 1) + 3) = 6
    end do
  end if
end subroutine get_mesh
