module angle
implicit none

! number angles per *half space*
integer :: number_angles 
! cosine of angles
double precision, allocatable, dimension(:) :: mu 
! quadrature weight (sum to 1)
double precision, allocatable, dimension(:) :: wt 
! Polynomial container for the legendre polynomials
double precision, allocatable, dimension(:,:) :: p_leg

integer, parameter :: GL = 1, DGL = 2

double precision, parameter :: PI = 3.141592653589793116_8

contains

! Allocate quadrature arrays
subroutine initialize_angle(n, option) 
  integer, intent(in) :: n, option
  integer :: i
  number_angles = n
  allocate(mu(number_angles), wt(number_angles))
  if (option .eq. GL) then
    call generate_gl_parameters(2*number_angles, mu, wt)
  else if (option .eq. DGL) then
    call generate_gl_parameters(number_angles, mu, wt)
    mu = 0.5*mu + 0.5
    wt = 0.5*wt
    do i = 1, number_angles/2
      mu(number_angles-i+1) = 1-mu(i)
      wt(number_angles-i+1) = wt(i) 
    end do
  else
    stop "Invalid quadrature option."
  end if
end subroutine initialize_angle

! Deallocated quadrature arrays
subroutine finalize_angle()
  if (allocated(mu)) then
    deallocate(mu)
  end if
  if (allocated(wt)) then
    deallocate(wt)
  end if
end subroutine finalize_angle

! Generate Gauss-Legendre parameters.
subroutine generate_gl_parameters(m, x, w)
  implicit none
  integer, intent(in) :: m
  double precision, intent(inout), dimension(:) :: x, w
  ! Local Variables
  integer :: i, j
  double precision :: p, dp
  ! The roots are symmetric, so we only find half of them.
  do i = 1, int(m/2)+mod(m,2)
    ! Asymptotic approx. due to F. Tricomi, Ann. Mat. Pura Appl., 31 (1950)
    x(i) = cos(pi*(i-0.25)/(m+0.5)) * (1-0.125*(1_8/m**2-1_8/m**3))
    do j = 1, 100
      p = legendre_p(m, x(i))
      dp = d_legendre_p(m, x(i))
      if (abs(p/dp) .lt. 1e-16) then
        exit
      end if
      ! Newton step
      x(i) = x(i) - p/dp 
    end do
    w(i) = 2.0_8/((1.0 - x(i)**2)*d_legendre_p(m, x(i))**2)
  end do
end subroutine generate_gl_parameters

subroutine initialize_polynomials(number_legendre)
  integer, intent(in) :: number_legendre
  integer :: l, a
  allocate(p_leg(0:number_legendre, number_angles * 2))
  do a = 1, number_angles
    do l = 0, number_legendre
      p_leg(l,a) = legendre_p(l,mu(a))
      p_leg(l,2 * number_angles - a + 1) = legendre_p(l,-mu(a))
    end do
  end do
end subroutine initialize_polynomials

! Compute P_l(x) using recursion relationship
double precision function legendre_p(l, x)
  implicit none
  integer, intent(in) :: l
  double precision, intent(in) :: x
  ! local variables
  double precision :: P_0, P_1, P_2
  integer :: m
  if (l .eq. 0) then
    P_0 = 1.0_8
  else if (l .eq. 1) then
    P_0 = x
  else
    P_1 = 1.0_8 ! P(l-2, x)
    P_0 = x     ! P(l-1, x)
    do m = 2, l
      P_2 = P_1
      P_1 = P_0
      P_0 = ((2*m-1)*x*P_1-(m-1)*P_2)/m
    end do
  end if
  legendre_p = P_0
end function legendre_p

! Compute dP_l/dx using recursion relationship
double precision function d_legendre_p(l, x)
  implicit none
  integer, intent(in) :: l
  double precision, intent(in) :: x
  ! local variables
  if (l .eq. 0) then
    d_legendre_p = 0.0_8
  else
    d_legendre_p = (legendre_p(l-1, x)-x*legendre_p(l, x))*l/(1-x**2)
  end if
end function d_legendre_p

end module angle
