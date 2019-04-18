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
        a, & ! Angle index
        i, & ! Counting variable
        j, & ! Counting variable
        p
    double precision :: &
        phia(number_angles_azi), &
        wt_azi(number_angles_azi), &
        xi(2 * number_angles_pol), &
        wt_pol(2 * number_angles_pol)
    double precision, allocatable, dimension(:) :: &
        mu_vals, & !
        wt_vals    !
    integer, allocatable, dimension(:) :: &
        wt_map

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

      if (angle_order < 2 .or. angle_order > 24) then
        print *, 'angles are only defined for 2 <= angle_order <= 24'
        stop
      else if (modulo(angle_order,  2) /= 0) then
        print *, 'angles are only defined for even angle_order'
      end if

      ! Compute the total number of angles
      number_angles = angle_order * (angle_order + 2) / 8
      allocate(mu(number_angles), eta(number_angles), wt(number_angles), wt_map(number_angles))
      if (angle_order == 2) then
        allocate(mu_vals(1), wt_vals(1))
        mu_vals(1) = 0.577350269189625764509149
        wt_vals(1) = 1.000000000000000000000000
        wt_map = [0]
      else if (angle_order == 4) then
        allocate(mu_vals(2), wt_vals(1))
        mu_vals(1) = 0.350021174581540677777041
        mu_vals(2) = 0.868890300722201205229788
        wt_vals(1) = 0.333333333333333333333333
        wt_map = [0, 0, 0]
      else if (angle_order == 6) then
        allocate(mu_vals(3), wt_vals(2))
        mu_vals(1) = 0.266635401516704720331535
        mu_vals(2) = 0.681507726536546927403750
        mu_vals(3) = 0.926180935517489107558380
        wt_vals(1) = 0.176126130863383433783565
        wt_vals(2) = 0.157207202469949899549768
        wt_map = [0, 1, 0, 1, 1, 0]
      else if (angle_order == 8) then
        allocate(mu_vals(4), wt_vals(3))
        mu_vals(1) = 0.218217890235992381266097
        mu_vals(2) = 0.577350269189625764509149
        mu_vals(3) = 0.786795792469443145800830
        mu_vals(4) = 0.951189731211341853132399
        wt_vals(1) = 0.120987654320987654320988
        wt_vals(2) = 0.0907407407407407407407407
        wt_vals(3) = 0.0925925925925925925925926
        wt_map = [0, 1, 1, 0, 1, 2, 1, 1, 1, 0]
      else if (angle_order == 10) then
        allocate(mu_vals(5), wt_vals(4))
        mu_vals(1) = 0.189321326478010476671494
        mu_vals(2) = 0.508881755582618974382711
        mu_vals(3) = 0.694318887594384317279217
        mu_vals(4) = 0.839759962236684758403029
        mu_vals(5) = 0.963490981110468484701598
        wt_vals(1) = 0.0893031479843567214704325
        wt_vals(2) = 0.0725291517123655242296233
        wt_vals(3) = 0.0450437674364086390490892
        wt_vals(4) = 0.0539281144878369243545650
        wt_map = [0, 1, 2, 1, 0, 1, 3, 3, 1, 2, 3, 2, 1, 1, 0]
      else if (angle_order == 12) then
        allocate(mu_vals(6), wt_vals(5))
        mu_vals(1) = 0.167212652822713264084504
        mu_vals(2) = 0.459547634642594690016761
        mu_vals(3) = 0.628019096642130901034766
        mu_vals(4) = 0.760021014833664062877138
        mu_vals(5) = 0.872270543025721502340662
        mu_vals(6) = 0.971637719251358378302376
        wt_vals(1) = 0.0707625899700910439766549
        wt_vals(2) = 0.0558811015648888075828962
        wt_vals(3) = 0.0373376737588285824652402
        wt_vals(4) = 0.0502819010600571181385765
        wt_vals(5) = 0.0258512916557503911218290
        wt_map = [0, 1, 2, 2, 1, 0, 1, 3, 4, 3, 1, 2, 4, 4, 2, 2, 3, 2, 1, 1, 0]
      else if (angle_order == 14) then
        allocate(mu_vals(7), wt_vals(7))
        mu_vals(1) = 0.151985861461031912404799
        mu_vals(2) = 0.422156982304796966896263
        mu_vals(3) = 0.577350269189625764509149
        mu_vals(4) = 0.698892086775901338963210
        mu_vals(5) = 0.802226255231412057244328
        mu_vals(6) = 0.893691098874356784901111
        mu_vals(7) = 0.976627152925770351762946
        wt_vals(1) = 0.0579970408969969964063611
        wt_vals(2) = 0.0489007976368104874582568
        wt_vals(3) = 0.0227935342411872473257345
        wt_vals(4) = 0.0394132005950078294492985
        wt_vals(5) = 0.0380990861440121712365891
        wt_vals(6) = 0.0258394076418900119611012
        wt_vals(7) = 0.00826957997262252825269908
        wt_map = [0, 1, 2, 3, 2, 1, 0, 1, 4, 5, 5, 4, 1, 2, &
                  5, 6, 5, 2, 3, 5, 5, 3, 2, 4, 2, 1, 1, 0]
      else if (angle_order == 16) then
        allocate(mu_vals(8), wt_vals(8))
        mu_vals(1) = 0.138956875067780344591732
        mu_vals(2) = 0.392289261444811712294197
        mu_vals(3) = 0.537096561300879079878296
        mu_vals(4) = 0.650426450628771770509703
        mu_vals(5) = 0.746750573614681064580018
        mu_vals(6) = 0.831996556910044145168291
        mu_vals(7) = 0.909285500943725291652116
        mu_vals(8) = 0.980500879011739882135849
        wt_vals(1) = 0.0489872391580385335008367
        wt_vals(2) = 0.0413295978698440232405505
        wt_vals(3) = 0.0203032007393652080748070
        wt_vals(4) = 0.0265500757813498446015484
        wt_vals(5) = 0.0379074407956004002099321
        wt_vals(6) = 0.0135295047786756344371600
        wt_vals(7) = 0.0326369372026850701318409
        wt_vals(8) = 0.0103769578385399087825920
        wt_map = [0, 1, 2, 3, 3, 2, 1, 0, 1, 4, 5, 6, 5, 4, &
                  1, 2, 5, 7, 7, 5, 2, 3, 6, 7, 6, 3, 3, 5, &
                  5, 3, 2, 4, 2, 1, 1, 0]
      else if (angle_order == 18) then
        allocate(mu_vals(9), wt_vals(10))
        mu_vals(1)  = 0.129344504545924818514086
        mu_vals(2)  = 0.368043816053393605686086
        mu_vals(3)  = 0.504165151725164054411848
        mu_vals(4)  = 0.610662549934881101060239
        mu_vals(5)  = 0.701166884252161909657019
        mu_vals(6)  = 0.781256199495913171286914
        mu_vals(7)  = 0.853866206691488372341858
        mu_vals(8)  = 0.920768021061018932899055
        mu_vals(9)  = 0.983127661236087115272518
        wt_vals(1)  = 0.0422646448843821748535825
        wt_vals(2)  = 0.0376127473827281471532380
        wt_vals(3)  = 0.0122691351637405931037187
        wt_vals(4)  = 0.0324188352558815048715646
        wt_vals(5)  = 0.00664438614619073823264082
        wt_vals(6)  = 0.0312093838436551370068864
        wt_vals(7)  = 0.0160127252691940275641645
        wt_vals(8)  = 0.0200484595308572875885066
        wt_vals(9)  = 0.000111409402059638628382279
        wt_vals(10) = 0.0163797038522425240494567
        wt_map = [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 5, 6, 7, 7, &
                  6, 5, 1, 2, 6, 8, 9, 8, 6, 2, 3, 7, 9, 9, &
                  7, 3, 4, 7, 8, 7, 4, 3, 6, 6, 3, 2, 5, 2, &
                  1, 1, 0]
      else if (angle_order == 20) then
        allocate(mu_vals(10), wt_vals(12))
        mu_vals(1)  = 0.120603343036693597409418
        mu_vals(2)  = 0.347574292315847257336779
        mu_vals(3)  = 0.476519266143665680817278
        mu_vals(4)  = 0.577350269189625764509149
        mu_vals(5)  = 0.663020403653288019308783
        mu_vals(6)  = 0.738822561910371432904974
        mu_vals(7)  = 0.807540401661143067193530
        mu_vals(8)  = 0.870852583760463975580977
        mu_vals(9)  = 0.929863938955324566667817
        mu_vals(10) = 0.985347485558646574628509
        wt_vals(1)  = 0.0370210490604481342320295
        wt_vals(2)  = 0.0332842165376314841003910
        wt_vals(3)  = 0.0111738965965092519614021
        wt_vals(4)  = 0.0245177476959359285418987
        wt_vals(5)  = 0.0135924329650041789567081
        wt_vals(6)  = 0.0318029065936585971501960
        wt_vals(7)  = 0.00685492401402507781062634
        wt_vals(8)  = 0.0308105481755299327227893
        wt_vals(9)  = -0.000139484716502602877593527
        wt_vals(10) = 0.00544675187330776223879437
        wt_vals(11) = 0.00474564692642379971238396
        wt_vals(12) = 0.0277298541009064049325246
        wt_map = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 1, 5, 6, 7, &
                  8, 7, 6, 5, 1, 2, 6, 9, 10, 10, 9, 6, 2, 3, 7, &
                  10, 11, 10, 7, 3, 4, 8, 10, 10, 8, 4, 4, 7, 9, 7, 4, &
                  3, 6, 6, 3, 2, 5, 2, 1, 1, 0]
      else if (angle_order == 22) then
        allocate(mu_vals(11), wt_vals(14))
        mu_vals(1)  = 0.113888641383070838173488
        mu_vals(2)  = 0.330271760593086736334651
        mu_vals(3)  = 0.452977095507524183904005
        mu_vals(4)  = 0.548905330875560154226714
        mu_vals(5)  = 0.630401360620980621392149
        mu_vals(6)  = 0.702506006153654989703184
        mu_vals(7)  = 0.767869456282208576047898
        mu_vals(8)  = 0.828089557415325768804621
        mu_vals(9)  = 0.884217805921983001958912
        mu_vals(10) = 0.936989829997455780115072
        mu_vals(11) = 0.986944149751056870330152
        wt_vals(1)  = 0.0329277718552552308051381
        wt_vals(2)  = 0.0309569328165031538543025
        wt_vals(3)  = 0.00577105953220643022391829
        wt_vals(4)  = 0.0316834548379952775919418
        wt_vals(5)  = -0.00669350304140992494103696
        wt_vals(6)  = 0.0368381622687682466526634
        wt_vals(7)  = 0.0273139698006629537455404
        wt_vals(8)  = 0.0100962716435030437817055
        wt_vals(9)  = 0.0195181067555849392224199
        wt_vals(10) = 0.0117224275470949786864925
        wt_vals(11) = -0.00442773155233893239996431
        wt_vals(12) = 0.0156214785078803432781324
        wt_vals(13) = -0.0101774221315738297143270
        wt_vals(14) = 0.0135061258938431808485310
        wt_map = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 6, 7, 8, 9, &
                  9, 8, 7, 6, 1, 2, 7, 10, 11, 12, 11, 10, 7, 2, 3, &
                  8, 11, 13, 13, 11, 8, 3, 4, 9, 12, 13, 12, 9, 4, 5, &
                  9, 11, 11, 9, 5, 4, 8, 10, 8, 4, 3, 7, 7, 3, 2, 6, 2, 1, 1, 0]
      else if (angle_order == 24) then
        allocate(mu_vals(12), wt_vals(16))
        mu_vals(1)  = 0.107544208775147285552086
        mu_vals(2)  = 0.315151630853896874875332
        mu_vals(3)  = 0.432522073446742487657060
        mu_vals(4)  = 0.524242441631224399254880
        mu_vals(5)  = 0.602150256328323868809286
        mu_vals(6)  = 0.671073561381361944701265
        mu_vals(7)  = 0.733549261041044861004094
        mu_vals(8)  = 0.791106384731321324814121
        mu_vals(9)  = 0.844750913317919895113069
        mu_vals(10) = 0.895186516397704814461305
        mu_vals(11) = 0.942928254285052510917188
        mu_vals(12) = 0.988366574868785749937406
        wt_vals(1)  = 0.0295284942030736546025272
        wt_vals(2)  = 0.0281530651743695026834932
        wt_vals(3)  = 0.00519730128072174996473824
        wt_vals(4)  = 0.0259897467786242920448933
        wt_vals(5)  = 0.00146378160153344429844948
        wt_vals(6)  = 0.0166609651269037212368055
        wt_vals(7)  = 0.0281343344093849194875108
        wt_vals(8)  = 0.00214364311909247909952968
        wt_vals(9)  = 0.0331943417648083019611294
        wt_vals(10) = -0.0142483904822400753741381
        wt_vals(11) = 0.0416812529998231580614934
        wt_vals(12) = 0.00323522898964475022578598
        wt_vals(13) = 0.000813552611571786631179287
        wt_vals(14) = 0.00228403610697848813660369
        wt_vals(15) = 0.0338971925236628645848112
        wt_vals(16) = -0.00644725595698339499416262
        wt_map = [0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 1, 6, 7, 8, 9, 10, &
                  9, 8, 7, 6, 1, 2, 7, 11, 12, 13, 13, 12, 11, 7, 2, 3, 8, &
                  12, 14, 15, 14, 12, 8, 3, 4, 9, 13, 15, 15, 13, 9, 4, 5, 10, &
                  13, 14, 13, 10, 5, 5, 9, 12, 12, 9, 5, 4, 8, 11, 8, 4, 3, 7, &
                  7, 3, 2, 6, 2, 1, 1, 0]
      end if

      a = 1
      do i = 0, angle_order / 2 - 1
        do j = 0, angle_order / 2 - i - 1
          mu(a) = mu_vals(i + 1)
          eta(a) = mu_vals(j + 1)
          wt(a) = wt_vals(wt_map(a) + 1)
          a = a + 1
        end do
      end do

      deallocate(mu_vals, wt_vals)

      ! Verify that the angles are specified for 2D
!      if (number_angles_azi < 1) then
!        print *, 'Number of azimuthal angles must be at least 1'
!        stop
!      end if
!      if (number_angles_pol < 1) then
!        print *, 'Number of polar angles must be at least 1'
!        stop
!      end if
!
!      number_angles = number_angles_azi * number_angles_pol
!
!      do a = 0, number_angles_azi - 1
!        phia(a + 1) = PI * (2 * a + 1) / (4 * number_angles_azi)
!        wt_azi(a + 1) = PI / (2 * number_angles_azi)
!      end do
!
!      call generate_gl_parameters(2 * number_angles_pol, xi, wt_pol)
!
!      do a = 1, number_angles_pol
!        xi(2 * number_angles_pol - a + 1) = -xi(a)
!        wt_pol(2 * number_angles_pol - a + 1) = wt_pol(a)
!      end do
!
!      xi = -xi / 2 + 0.5
!
!      allocate(mu(number_angles), eta(number_angles), wt(number_angles))
!
!      do p = 1, number_angles_pol
!        do a = 1, number_angles_azi
!          wt(a + (p - 1) * number_angles_azi) = wt_pol(p) * wt_azi(a)
!          mu(a + (p - 1) * number_angles_azi) = cos(phia(a)) * sqrt(1 - xi(p) ** 2)
!          eta(a + (p - 1) * number_angles_azi) = sin(phia(a)) * sqrt(1 - xi(p) ** 2)
!        end do
!      end do
!
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
