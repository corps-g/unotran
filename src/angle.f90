module angle

integer :: number_angles ! number angles per half space
double precision, allocatable, dimension(:) :: mu ! cosine of angles
double precision, allocatable, dimension(:) :: wt ! quadrature weight (sum to 1)

integer, parameter :: GL = 1, DGL = 2

contains

subroutine initialize(n, option) 
  integer, intent(in) :: n, option
  integer :: i
  number_angles = n
  allocate(mu(number_angles), wt(number_angles))
  
  if (option .eq. GL) then
    call fill_gl()
  else if (option .eq. DGL) then
    call fill_dgl()
  else
    stop "Invalid quadrature option."
  end if
  
end subroutine initialize

subroutine finalize()
  if (allocated(mu)) then
    deallocate(mu)
  end if
  if (allocated(wt)) then
    deallocate(wt)
  end if
end subroutine finalize

subroutine fill_gl
  ! put algorithm here to determine legendre zeros over -1, 1
end subroutine fill_gl

subroutine fill_dgl
  ! shift and scale gl numbers
end subroutine fill_dgl


end module angle
