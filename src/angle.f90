module angle
  ! ############################################################################
  ! Setup the angular quadrature
  ! ############################################################################

  implicit none

  double precision, allocatable, dimension(:) :: &
      mu,                       & ! cosine of angles
      eta,                      & ! sine of angles
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
    use control, only : number_angles_azi, number_angles_pol, spatial_dimension, &
                        angle_option, number_angles, angle_order

    ! Variable definitions
    integer :: &
        a, &  ! Angle index
        p
    double precision :: &
        phia(number_angles_azi), &
        wt_azi(number_angles_azi), &
        xi(2 * number_angles_pol), &
        wt_pol(2 * number_angles_pol)

    if (spatial_dimension == 1) then

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
    else if (spatial_dimension == 2) then
      ! Verify that the angles are specified for 2D
      if (number_angles_azi < 1) then
        print *, 'Number of azimuthal angles must be at least 1'
        stop
      end if
      if (number_angles_pol < 1) then
        print *, 'Number of polar angles must be at least 1'
        stop
      end if

      number_angles = number_angles_azi * number_angles_pol

      do a = 0, number_angles_azi - 1
        phia(a + 1) = PI * (2 * a + 1) / (4 * number_angles_azi)
        wt_azi(a + 1) = PI / (2 * number_angles_azi)
      end do

      call generate_gl_parameters(2 * number_angles_pol, xi, wt_pol)

      do a = 1, number_angles_pol
        xi(2 * number_angles_pol - a + 1) = -xi(a)
        wt_pol(2 * number_angles_pol - a + 1) = wt_pol(a)
      end do

      xi = -xi / 2 + 0.5

      allocate(mu(number_angles), eta(number_angles), wt(number_angles))

      do p = 1, number_angles_pol
        do a = 1, number_angles_azi
          wt(a + (p - 1) * number_angles_azi) = wt_pol(p) * wt_azi(a)
          mu(a + (p - 1) * number_angles_azi) = cos(phia(a)) * sqrt(1 - xi(p) ** 2)
          eta(a + (p - 1) * number_angles_azi) = sin(phia(a)) * sqrt(1 - xi(p) ** 2)
        end do
      end do

    end if

  end subroutine initialize_angle

  subroutine finalize_angle()
    ! ##########################################################################
    ! Deallocated quadrature arrays
    ! ##########################################################################

    if (allocated(mu)) then
      deallocate(mu)
    end if
    if (allocated(eta)) then
      deallocate(eta)
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

  double precision function generate_y_lm(l, m, mu, eta, xi)
    ! ##########################################################################
    ! Calculate the spherical harmonic of degree l, order m
    ! ##########################################################################

    integer, intent(in) :: &
        l,   & ! Legendre order (or spherical harmonic degree)
        m      ! Spherical harmonic order
    double precision, intent(in) :: &
        mu,  & ! direction cosine w/r to x axis
        eta, & ! direction cosine w/r to y axis
        xi     ! direction cosine w/r to z axis

    if (l < 0) then
      print *, 'Bad value for l in angle.f90 (l < 0)'
      stop
    else if (abs(m) > l) then
      print *, 'Bad value for m in angle.f90 (abs(m) > 0)'
      stop
    else if (abs(mu) > 1) then
      print *, 'Bad value for mu in angle.f90 (abs(mu) > 1)'
      stop
    else if (abs(eta) > 1) then
      print *, 'Bad value for eta in angle.f90 (abs(eta) > 1)'
      stop
    else if (abs(xi) > 1) then
      print *, 'Bad value for xi in angle.f90 (abs(xi) > 1)'
      stop
    else if (mu ** 2 + eta ** 2 + xi ** 2 - 1 > 1e-5) then
      print *, 'Angles do not math to unity'
      stop
    end if

    if (l == 0) then
      ! R_0^0 = sqrt(2)
      generate_y_lm = 1.41421356237309505
    else if (l == 1) then
      if (m == -1) then
        ! R_1^-1 = sqrt(2)*eta/2
        generate_y_lm = 0.707106781186547524*eta
      else if (m == 0) then
        ! R_1^0 = sqrt(2)*xi
        generate_y_lm = 1.41421356237309505*xi
      else if (m == 1) then
        ! R_1^1 = sqrt(2)*mu/2
        generate_y_lm = 0.707106781186547524*mu
      end if
    else if (l == 2) then
      if (m == -2) then
        ! R_2^-2 = 0.5*sqrt(6)*eta*mu
        generate_y_lm = 1.22474487139158905*eta*mu
      else if (m == -1) then
        ! R_2^-1 = 0.5*sqrt(6)*eta*xi
        generate_y_lm = 1.22474487139158905*eta*xi
      else if (m == 0) then
        ! R_2^0 = sqrt(2)*(1.5*xi**2 - 0.5)
        generate_y_lm = 2.12132034355964257*xi**2 - 0.707106781186547524
      else if (m == 1) then
        ! R_2^1 = 0.5*sqrt(6)*mu*xi
        generate_y_lm = 1.22474487139158905*mu*xi
      else if (m == 2) then
        ! R_2^2 = 0.25*sqrt(6)*(-eta**2 + mu**2)
        generate_y_lm = -0.612372435695794525*eta**2 + 0.612372435695794525*mu**2
      end if
    else if (l == 3) then
      if (m == -3) then
        ! R_3^-3 = sqrt(5)*eta*(-0.25*eta**2 + 0.75*mu**2)
        generate_y_lm = 2.2360679774997897*eta*(-0.25*eta**2 + 0.75*mu**2)
      else if (m == -2) then
        ! R_3^-2 = 0.5*sqrt(30)*eta*mu*xi
        generate_y_lm = 2.73861278752583057*eta*mu*xi
      else if (m == -1) then
        ! R_3^-1 = sqrt(3)*eta*(1.25*xi**2 - 0.25)
        generate_y_lm = 1.73205080756887729*eta*(1.25*xi**2 - 0.25)
      else if (m == 0) then
        ! R_3^0 = sqrt(2)*xi*(2.5*xi**2 - 1.5)
        generate_y_lm = 1.41421356237309505*xi*(2.5*xi**2 - 1.5)
      else if (m == 1) then
        ! R_3^1 = sqrt(3)*mu*(1.25*xi**2 - 0.25)
        generate_y_lm = 1.73205080756887729*mu*(1.25*xi**2 - 0.25)
      else if (m == 2) then
        ! R_3^2 = 0.25*sqrt(30)*xi*(-eta**2 + mu**2)
        generate_y_lm = 1.36930639376291528*xi*(-eta**2 + mu**2)
      else if (m == 3) then
        ! R_3^3 = sqrt(5)*mu*(-0.75*eta**2 + 0.25*mu**2)
        generate_y_lm = 2.2360679774997897*mu*(-0.75*eta**2 + 0.25*mu**2)
      end if
    else if (l == 4) then
      if (m == -4) then
        ! R_4^-4 = 0.25*sqrt(70)*eta*mu*(-eta**2 + mu**2)
        generate_y_lm = 2.09165006633518887*eta*mu*(-eta**2 + mu**2)
      else if (m == -3) then
        ! R_4^-3 = sqrt(35)*eta*xi*(-0.25*eta**2 + 0.75*mu**2)
        generate_y_lm = 5.91607978309961604*eta*xi*(-0.25*eta**2 + 0.75*mu**2)
      else if (m == -2) then
        ! R_4^-2 = sqrt(10)*eta*mu*(1.75*xi**2 - 0.25)
        generate_y_lm = 3.16227766016837933*eta*mu*(1.75*xi**2 - 0.25)
      else if (m == -1) then
        ! R_4^-1 = sqrt(5)*eta*xi*(1.75*xi**2 - 0.75)
        generate_y_lm = 2.2360679774997897*eta*xi*(1.75*xi**2 - 0.75)
      else if (m == 0) then
        ! R_4^0 = sqrt(2)*(4.375*xi**4 - 3.75*xi**2 + 0.375)
        generate_y_lm = 6.18718433538229084*xi**4 - 5.30330085889910643*xi**2 + 0.530330085889910643
      else if (m == 1) then
        ! R_4^1 = sqrt(5)*mu*xi*(1.75*xi**2 - 0.75)
        generate_y_lm = 2.2360679774997897*mu*xi*(1.75*xi**2 - 0.75)
      else if (m == 2) then
        ! R_4^2 = sqrt(10)*(-eta**2 + mu**2)*(52.5*xi**2 - 7.5)/60
        generate_y_lm = 0.0527046276694729889*(-eta**2 + mu**2)*(52.5*xi**2 - 7.5)
      else if (m == 3) then
        ! R_4^3 = sqrt(35)*mu*xi*(-0.75*eta**2 + 0.25*mu**2)
        generate_y_lm = 5.91607978309961604*mu*xi*(-0.75*eta**2 + 0.25*mu**2)
      else if (m == 4) then
        ! R_4^4 = sqrt(70)*(0.0625*eta**4 - 0.375*eta**2*mu**2 + 0.0625*mu**4)
        generate_y_lm = 0.522912516583797217*eta**4 - 3.1374750995027833*eta**2*mu**2 + 0.522912516583797217*mu**4
      end if
    else if (l == 5) then
      if (m == -5) then
        ! R_5^-5 = sqrt(7)*eta*(0.1875*eta**4 - 1.875*eta**2*mu**2 + 0.9375*mu**4)
        generate_y_lm = 2.64575131106459059*eta*(0.1875*eta**4 - 1.875*eta**2*mu**2 + 0.9375*mu**4)
      else if (m == -4) then
        ! R_5^-4 = 0.75*sqrt(70)*eta*mu*xi*(-eta**2 + mu**2)
        generate_y_lm = 6.27495019900556661*eta*mu*xi*(-eta**2 + mu**2)
      else if (m == -3) then
        ! R_5^-3 = -sqrt(35)*eta*(eta**2 - 3*mu**2)*(472.5*xi**2 - 52.5)/840
        generate_y_lm = -0.00704295212273763815*eta*(eta**2 - 3.0*mu**2)*(472.5*xi**2 - 52.5)
      else if (m == -2) then
        ! R_5^-2 = sqrt(210)*eta*mu*xi*(0.75*xi**2 - 0.25)
        generate_y_lm = 14.4913767461894386*eta*mu*xi*(0.75*xi**2 - 0.25)
      else if (m == -1) then
        ! R_5^-1 = sqrt(30)*eta*(1.3125*xi**4 - 0.875*xi**2 + 0.0625)
        generate_y_lm = 5.47722557505166113*eta*(1.3125*xi**4 - 0.875*xi**2 + 0.0625)
      else if (m == 0) then
        ! R_5^0 = sqrt(2)*xi*(7.875*xi**4 - 8.75*xi**2 + 1.875)
        generate_y_lm = 1.41421356237309505*xi*(7.875*xi**4 - 8.75*xi**2 + 1.875)
      else if (m == 1) then
        ! R_5^1 = sqrt(30)*mu*(1.3125*xi**4 - 0.875*xi**2 + 0.0625)
        generate_y_lm = 5.47722557505166113*mu*(1.3125*xi**4 - 0.875*xi**2 + 0.0625)
      else if (m == 2) then
        ! R_5^2 = sqrt(210)*xi*(-eta**2 + mu**2)*(157.5*xi**2 - 52.5)/420
        generate_y_lm = 0.0345032779671177109*xi*(-eta**2 + mu**2)*(157.5*xi**2 - 52.5)
      else if (m == 3) then
        ! R_5^3 = -sqrt(35)*mu*(3*eta**2 - mu**2)*(472.5*xi**2 - 52.5)/840
        generate_y_lm = -0.00704295212273763815*mu*(3.0*eta**2 - mu**2)*(472.5*xi**2 - 52.5)
      else if (m == 4) then
        ! R_5^4 = sqrt(70)*xi*(0.1875*eta**4 - 1.125*eta**2*mu**2 + 0.1875*mu**4)
        generate_y_lm = 8.36660026534075548*xi*(0.1875*eta**4 - 1.125*eta**2*mu**2 + 0.1875*mu**4)
      else if (m == 5) then
        ! R_5^5 = sqrt(7)*mu*(0.9375*eta**4 - 1.875*eta**2*mu**2 + 0.1875*mu**4)
        generate_y_lm = 2.64575131106459059*mu*(0.9375*eta**4 - 1.875*eta**2*mu**2 + 0.1875*mu**4)
      end if
    else if (l == 6) then
      if (m == -6) then
        ! R_6^-6 = sqrt(231)*eta*mu*(0.1875*eta**4 - 0.625*eta**2*mu**2 + 0.1875*mu**4)
        generate_y_lm = 15.1986841535706636*eta*mu*(0.1875*eta**4 - 0.625*eta**2*mu**2 + 0.1875*mu**4)
      else if (m == -5) then
        ! R_6^-5 = sqrt(77)*eta*xi*(0.1875*eta**4 - 1.875*eta**2*mu**2 + 0.9375*mu**4)
        generate_y_lm = 8.77496438739212206*eta*xi*(0.1875*eta**4 - 1.875*eta**2*mu**2 + 0.9375*mu**4)
      else if (m == -4) then
        ! R_6^-4 = -sqrt(14)*eta*mu*(eta**2 - mu**2)*(5197.5*xi**2 - 472.5)/1260
        generate_y_lm = -0.00296956935458249316*eta*mu*(eta**2 - mu**2)*(5197.5*xi**2 - 472.5)
      else if (m == -3) then
        ! R_6^-3 = -sqrt(105)*eta*xi*(eta**2 - 3*mu**2)*(1732.5*xi**2 - 472.5)/2520
        generate_y_lm = -0.00406625030395222158*eta*xi*(eta**2 - 3.0*mu**2)*(1732.5*xi**2 - 472.5)
      else if (m == -2) then
        ! R_6^-2 = sqrt(105)*eta*mu*(2.0625*xi**4 - 1.125*xi**2 + 0.0625)
        generate_y_lm = 10.2469507659595984*eta*mu*(2.0625*xi**4 - 1.125*xi**2 + 0.0625)
      else if (m == -1) then
        ! R_6^-1 = sqrt(42)*eta*xi*(2.0625*xi**4 - 1.875*xi**2 + 0.3125)
        generate_y_lm = 6.48074069840786023*eta*xi*(2.0625*xi**4 - 1.875*xi**2 + 0.3125)
      else if (m == 0) then
        ! R_6^0 = sqrt(2)*(14.4375*xi**6 - 19.6875*xi**4 + 6.5625*xi**2 - 0.3125)
        generate_y_lm = 20.4177083067615598*xi**6 - 27.8423295092203088*xi**4 + 9.28077650307343626*xi**2 - 0.441941738241592203
      else if (m == 1) then
        ! R_6^1 = sqrt(42)*mu*xi*(2.0625*xi**4 - 1.875*xi**2 + 0.3125)
        generate_y_lm = 6.48074069840786023*mu*xi*(2.0625*xi**4 - 1.875*xi**2 + 0.3125)
      else if (m == 2) then
        ! R_6^2 = sqrt(105)*(-eta**2 + mu**2)*(433.125*xi**4 - 236.25*xi**2 + 13.125)/420
        generate_y_lm = 0.0243975018237133295*(-eta**2 + mu**2)*(433.125*xi**4 - 236.25*xi**2 + 13.125)
      else if (m == 3) then
        ! R_6^3 = -sqrt(105)*mu*xi*(3*eta**2 - mu**2)*(1732.5*xi**2 - 472.5)/2520
        generate_y_lm = -0.00406625030395222158*mu*xi*(3.0*eta**2 - mu**2)*(1732.5*xi**2 - 472.5)
      else if (m == 4) then
        ! R_6^4 = sqrt(14)*(5197.5*xi**2 - 472.5)*(eta**2*(eta**2 - 3*mu**2) - mu**2*(3*eta**2 - mu**2))/5040
        generate_y_lm = 0.000742392338645623291*(5197.5*xi**2 - 472.5)*(eta**2*(eta**2 - 3.0*mu**2) - mu**2*(3.0*eta**2 - mu**2))
      else if (m == 5) then
        ! R_6^5 = sqrt(77)*mu*xi*(0.9375*eta**4 - 1.875*eta**2*mu**2 + 0.1875*mu**4)
        generate_y_lm = 8.77496438739212206*mu*xi*(0.9375*eta**4 - 1.875*eta**2*mu**2 + 0.1875*mu**4)
      else if (m == 6) then
        ! R_6^6 = sqrt(231)*(-0.03125*eta**6 + 0.46875*eta**4*mu**2 - 0.46875*eta**2*mu**4 + 0.03125*mu**6)
        generate_y_lm = -0.474958879799083238*eta**6 + 7.12438319698624858*eta**4*mu**2 &
                       - 7.12438319698624858*eta**2*mu**4 + 0.474958879799083238*mu**6
      end if
    else if (l == 7) then
      if (m == -7) then
        ! R_7^-7 = sqrt(858)*eta*(-0.015625*eta**6 + 0.328125*eta**4*mu**2 - 0.546875*eta**2*mu**4 + 0.109375*mu**6)
        generate_y_lm = 29.2916370317536196*eta*(-0.015625*eta**6 + 0.328125*eta**4*mu**2 - 0.546875*eta**2*mu**4 + 0.109375*mu**6)
      else if (m == -6) then
        ! R_7^-6 = sqrt(3003)*eta*mu*xi*(0.1875*eta**4 - 0.625*eta**2*mu**2 + 0.1875*mu**4)
        generate_y_lm = 54.7996350352810288*eta*mu*xi*(0.1875*eta**4 - 0.625*eta**2*mu**2 + 0.1875*mu**4)
      else if (m == -5) then
        ! R_7^-5 = sqrt(462)*eta*(67567.5*xi**2 - 5197.5)*(eta**2*(eta**2 - 3*mu**2) - 4*mu**2*(eta**2 - mu**2) - mu**2*(3*eta**2 - mu**2))/332640
        generate_y_lm = 0.0000646169590554493658*eta*(67567.5*xi**2 - 5197.5) &
                       * (eta**2*(eta**2 - 3.0*mu**2) - 4.0*mu**2*(eta**2 - mu**2) - mu**2*(3.0*eta**2 - mu**2))
      else if (m == -4) then
        ! R_7^-4 = -sqrt(462)*eta*mu*xi*(eta**2 - mu**2)*(22522.5*xi**2 - 5197.5)/13860
        generate_y_lm = -0.00155080701733078478*eta*mu*xi*(eta**2 - mu**2)*(22522.5*xi**2 - 5197.5)
      else if (m == -3) then
        ! R_7^-3 = -sqrt(42)*eta*(eta**2 - 3*mu**2)*(5630.625*xi**4 - 2598.75*xi**2 + 118.125)/2520
        generate_y_lm = -0.0025717224993681985*eta*(eta**2 - 3.0*mu**2)*(5630.625*xi**4 - 2598.75*xi**2 + 118.125)
      else if (m == -2) then
        ! R_7^-2 = sqrt(21)*eta*mu*xi*(8.9375*xi**4 - 6.875*xi**2 + 0.9375)
        generate_y_lm = 4.58257569495584001*eta*mu*xi*(8.9375*xi**4 - 6.875*xi**2 + 0.9375)
      else if (m == -1) then
        ! R_7^-1 = sqrt(14)*eta*(6.703125*xi**6 - 7.734375*xi**4 + 2.109375*xi**2 - 0.078125)
        generate_y_lm = 3.74165738677394139*eta*(6.703125*xi**6 - 7.734375*xi**4 + 2.109375*xi**2 - 0.078125)
      else if (m == 0) then
        ! R_7^0 = sqrt(2)*xi*(26.8125*xi**6 - 43.3125*xi**4 + 19.6875*xi**2 - 2.1875)
        generate_y_lm = 1.41421356237309505*xi*(26.8125*xi**6 - 43.3125*xi**4 + 19.6875*xi**2 - 2.1875)
      else if (m == 1) then
        ! R_7^1 = sqrt(14)*mu*(6.703125*xi**6 - 7.734375*xi**4 + 2.109375*xi**2 - 0.078125)
        generate_y_lm = 3.74165738677394139*mu*(6.703125*xi**6 - 7.734375*xi**4 + 2.109375*xi**2 - 0.078125)
      else if (m == 2) then
        ! R_7^2 = sqrt(21)*xi*(-eta**2 + mu**2)*(1126.125*xi**4 - 866.25*xi**2 + 118.125)/252
        generate_y_lm = 0.0181848241863326984*xi*(-eta**2 + mu**2)*(1126.125*xi**4 - 866.25*xi**2 + 118.125)
      else if (m == 3) then
        ! R_7^3 = -sqrt(42)*mu*(3*eta**2 - mu**2)*(5630.625*xi**4 - 2598.75*xi**2 + 118.125)/2520
        generate_y_lm = -0.0025717224993681985*mu*(3.0*eta**2 - mu**2)*(5630.625*xi**4 - 2598.75*xi**2 + 118.125)
      else if (m == 4) then
        ! R_7^4 = sqrt(462)*xi*(22522.5*xi**2 - 5197.5)*(eta**2*(eta**2 - 3*mu**2) - mu**2*(3*eta**2 - mu**2))/55440
        generate_y_lm = 0.000387701754332696195*xi*(22522.5*xi**2 - 5197.5) &
                       * (eta**2*(eta**2 - 3.0*mu**2) - mu**2*(3.0*eta**2 - mu**2))
      else if (m == 5) then
        ! R_7^5 = sqrt(462)*mu*(67567.5*xi**2 - 5197.5)*(eta**2*(eta**2 - 3*mu**2) + 4*eta**2*(eta**2 - mu**2) - mu**2*(3*eta**2 - mu**2))/332640
        generate_y_lm = 0.0000646169590554493658*mu*(67567.5*xi**2 - 5197.5) &
                       * (eta**2*(eta**2 - 3.0*mu**2) + 4.0*eta**2*(eta**2 - mu**2) - mu**2*(3.0*eta**2 - mu**2))
      else if (m == 6) then
        ! R_7^6 = sqrt(3003)*xi*(-0.03125*eta**6 + 0.46875*eta**4*mu**2 - 0.46875*eta**2*mu**4 + 0.03125*mu**6)
        generate_y_lm = 54.7996350352810288*xi*(-0.03125*eta**6 + 0.46875*eta**4*mu**2 - 0.46875*eta**2*mu**4 + 0.03125*mu**6)
      else if (m == 7) then
        ! R_7^7 = sqrt(858)*mu*(-0.109375*eta**6 + 0.546875*eta**4*mu**2 - 0.328125*eta**2*mu**4 + 0.015625*mu**6)
        generate_y_lm = 29.2916370317536196*mu*(-0.109375*eta**6 + 0.546875*eta**4*mu**2 - 0.328125*eta**2*mu**4 + 0.015625*mu**6)
      end if
    else
      print *, 'Order l=', l, ' not implemented'
      stop
    end if

  end function generate_y_lm

end module angle
