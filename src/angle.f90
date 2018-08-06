module angle
  ! ############################################################################
  ! Setup the angular quadrature
  ! ############################################################################

  implicit none

  double precision, allocatable, dimension(:) :: &
      mu,                       & ! cosine of angles
      wt                          ! quadrature weight (sum to 1)
  double precision, allocatable, dimension(:,:) :: &
      p_leg                       ! Container for the legendre polynomials
  integer, parameter :: &
      GL = 1, &                   ! Gauss Legendre quadrature
      DGL = 2                     ! Double Gauss Legendre quadrature
  double precision, parameter :: &
      PI = 3.141592653589793116_8 ! Setting the value of pi

  contains

  subroutine initialize_angle()
    ! ##########################################################################
    ! Allocate quadrature arrays
    ! ##########################################################################

    ! Use Statements
    use control, only : number_angles, angle_order, angle_option

    ! Variable definitions
    integer :: &
        a  ! Angle index

    number_angles = angle_order
    allocate(mu(number_angles), wt(number_angles))

    ! Generate the Legendre quadrature
    if (angle_option == GL) then
      call generate_gl_parameters(2 * number_angles, mu, wt)
    else if (angle_option == DGL) then
      call generate_gl_parameters(number_angles, mu, wt)
      mu = 0.5 * mu + 0.5
      wt = 0.5 * wt
      do a = 1, number_angles / 2
        mu(number_angles - a + 1) = 1 - mu(a)
        wt(number_angles - a + 1) = wt(a)
      end do
    else
      stop "Invalid quadrature option."
    end if

  end subroutine initialize_angle

  subroutine finalize_angle()
    ! ##########################################################################
    ! Deallocated quadrature arrays
    ! ##########################################################################

    if (allocated(mu)) then
      deallocate(mu)
    end if
    if (allocated(wt)) then
      deallocate(wt)
    end if
    if (allocated(p_leg)) then
      deallocate(p_leg)
    end if
  end subroutine finalize_angle

  subroutine generate_gl_parameters(m, x, w)
    ! ##########################################################################
    ! Generate Gauss-Legendre parameters.
    ! ##########################################################################

    ! Variable definitions
    integer, intent(in) :: &
        m    ! Number of angles
    double precision, intent(inout), dimension(:) :: &
        x, & ! Quadrature points
        w    ! Quadrature weights
    integer :: &
        i, & ! Looping variable
        j    ! Looping variable
    double precision :: &
        p, & ! Legendre polynomial value
        dp   ! Double Legendre polynomial value

    ! The roots are symmetric, so we only find half of them.
    do i = 1, int(m / 2) + mod(m, 2)
      ! Asymptotic approx. due to F. Tricomi, Ann. Mat. Pura Appl., 31 (1950)
      x(i) = cos(pi * (i - 0.25) / (m + 0.5)) * (1 - 0.125 * (1_8 / m ** 2 - 1_8 / m ** 3))
      do j = 1, 100
        p = legendre_p(m, x(i))
        dp = d_legendre_p(m, x(i))
        if (abs(p/dp) < 1e-16) then
          exit
        end if
        ! Newton step
        x(i) = x(i) - p / dp
      end do
      w(i) = 2.0_8 / ((1.0 - x(i) ** 2) * d_legendre_p(m, x(i)) ** 2)
    end do

  end subroutine generate_gl_parameters

  subroutine initialize_polynomials()
    ! ##########################################################################
    ! Fill p_leg with the discrete column vectors of DLP
    ! ##########################################################################

    ! Use Statements
    use control, only : number_angles, number_legendre

    ! Variable definitions
    integer :: &
        l,             & ! Legendre order index
        a                ! Angle index

    allocate(p_leg(0:number_legendre, number_angles * 2))

    do a = 1, number_angles
      do l = 0, number_legendre
        p_leg(l, a) = legendre_p(l, mu(a))
        p_leg(l, 2 * number_angles - a + 1) = legendre_p(l, -mu(a))
      end do
    end do

  end subroutine initialize_polynomials

  double precision function legendre_p(l, x)
    ! ##########################################################################
    ! Compute P_l(x) using recursion relationship
    ! ##########################################################################

    integer, intent(in) :: &
        l      ! Order of legendre polynomial
    double precision, intent(in) :: &
        x      ! Coordinate for polynomial value
    double precision :: &
        P_0, & ! Polynomial for order l
        P_1, & ! Polynomial for order l-1
        P_2    ! Polynomial for order l-2
    integer :: &
        m      ! Order index

    if (l == 0) then
      P_0 = 1.0_8
    else if (l == 1) then
      P_0 = x
    else
      P_1 = 1.0_8 ! P(l-2, x)
      P_0 = x     ! P(l-1, x)
      do m = 2, l
        P_2 = P_1
        P_1 = P_0
        P_0 = ((2 * m - 1) * x * P_1 - (m - 1) * P_2) / m
      end do
    end if
    legendre_p = P_0

  end function legendre_p

  double precision function d_legendre_p(l, x)
    ! ##########################################################################
    ! Compute dP_l/dx using recursion relationship
    ! ##########################################################################

    integer, intent(in) :: &
        l ! Order of legendre polynomial
    double precision, intent(in) :: &
        x ! Coordinate for polynomial value

    if (l == 0) then
      d_legendre_p = 0.0_8
    else
      d_legendre_p = (legendre_p(l - 1, x) - x * legendre_p(l, x)) * l / (1 - x ** 2)
    end if

  end function d_legendre_p

end module angle
