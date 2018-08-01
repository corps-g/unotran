module sources
  ! ############################################################################
  ! Compute the sources for the solver
  ! ############################################################################

  implicit none

  contains

  subroutine compute_in_source(g)
    ! ##########################################################################
    ! Compute the sources into group g from gp (excluding from g <- g)
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_source

    ! Variable definitions
    integer :: &
      g  ! Group index

    ! Reset the sources
    mg_source = 0.0

    ! Add the external source
    call compute_external(g, mg_source)

    ! Add the fission source
    call compute_in_fission(g, mg_source)

    ! Add the scatter source
    call compute_in_scatter(g, mg_source)

  end subroutine compute_in_source

  subroutine compute_within_group_source(g, c, a, source)
    ! ##########################################################################
    ! Compute the sources into group g from g in cell c and angle a
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_source
    use control, only : use_DGM

    ! Variable definitions
    integer, intent(in) :: &
      g,   & ! Group index
      a,   & ! Angle index
      c      ! Cell index
    double precision, intent(out) :: &
      source ! source for group g, cell c, and angle a

    ! Get the sources into group g
    source = mg_source(a,c)

    ! Add the fission source
    call compute_within_group_fission(g, c, source)

    ! Add the scatter source
    call compute_within_group_scatter(g, c, a, source)

    ! Add the delta source if DGM and not higher moment
    if (use_DGM) then
      call compute_delta(g, c, a, source)
    end if


  end subroutine compute_within_group_source

  subroutine compute_external(g, source)
    ! ##########################################################################
    ! Compute the external and fixed sources into group g
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_constant_source
    use dgm, only : source_m, dgm_order
    use control, only: use_DGM

    ! Variable definitions
    integer, intent(in) :: &
      g        ! Group index
    double precision, intent(out), dimension(:,:) :: &
      source   ! Source for group g

    if (use_DGM) then
      source(:,:) = source(:,:) + source_m(:,:,g,dgm_order)
    else
      source(:,:) = source(:,:) + mg_constant_source
    end if

  end subroutine compute_external

  subroutine compute_in_scatter(g, source)
    ! ##########################################################################
    ! Compute the scatter source into group g from gp (excluding from g <- g)
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_mMap, mg_phi, mg_sig_s
    use angle, only : p_leg
    use control, only : number_cells, number_angles, number_legendre, number_groups
    use dgm, only : dgm_order, phi_m_zero

    ! Variable definitions
    integer, intent(in) :: &
      g        ! Group index
    integer :: &
      gp,    & ! Group prime index
      a,     & ! Angle index
      c,     & ! Cell index
      l,     & ! Legendre index
      m        ! Material index
    double precision :: &
      phi      ! Scalar flux container
    double precision, intent(out), dimension(:,:) :: &
      source   ! Source for group g

    do gp = 1, number_groups
      ! Ignore the within-group terms
      if (g == gp) then
        cycle
      end if

      do a = 1, 2 * number_angles
        do c = 1, number_cells
          m = mg_mMap(c)
          do l = 0, number_legendre
            phi = merge(phi_m_zero(l,c,gp), mg_phi(l,c,gp), dgm_order > 0)
            source(c,a) = source(c,a) + 0.5 * p_leg(l,a) * mg_sig_s(l,m,gp,g) * phi
          end do
        end do
      end do
    end do

  end subroutine compute_in_scatter

  subroutine compute_within_group_scatter(g, c, a, source)
    ! ##########################################################################
    ! Compute the scatter source into group g from g
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_mMap, mg_phi, mg_sig_s
    use angle, only : p_leg
    use control, only : number_legendre
    use dgm, only : dgm_order, phi_m_zero

    ! Variable definitions
    integer, intent(in) :: &
      g,     & ! Group index
      a,     & ! Angle index
      c        ! Cell index
    integer :: &
      l,     & ! Legendre index
      m        ! Material index
    double precision :: &
      phi      ! Scalar flux container
    double precision, intent(inout) :: &
      source   ! Source for group g, cell c, and angle a

    m = mg_mMap(c)
    do l = 0, number_legendre
      phi = merge(phi_m_zero(l,c,g), mg_phi(l,c,g), dgm_order > 0)
      source = source + 0.5 * p_leg(l,a) * mg_sig_s(l,m,g,g) * phi
    end do

  end subroutine compute_within_group_scatter

  subroutine compute_in_fission(g, source)
    ! ##########################################################################
    ! Compute the fission source into group g from gp (excluding from g <- g)
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_mMap, keff, mg_phi, mg_chi, mg_nu_sig_f
    use control, only : number_cells, number_groups
    use dgm, only : dgm_order, phi_m_zero

    ! Variable definitions
    integer, intent(in) :: &
      g        ! Group index
    integer :: &
      gp,    & ! Group prime index
      c,     & ! Cell index
      m        ! Material index
    double precision :: &
      phi      ! Scalar flux container
    double precision, intent(out), dimension(:,:) :: &
      source   ! Source for group g

    do gp = 1, number_groups
      if (g == gp) then
        cycle
      end if
      do c = 1, number_cells
        m = mg_mMap(c)
        phi = merge(phi_m_zero(0,c,gp), mg_phi(0,c,gp), dgm_order > 0)
        source(c,:) = source(c,:) + 0.5 * mg_chi(m,g) * mg_nu_sig_f(m,gp) * phi / keff
      end do
    end do

  end subroutine compute_in_fission

  subroutine compute_within_group_fission(g, c, source)
    ! ##########################################################################
    ! Compute the fission source into group g from g in cell c
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_mMap, keff, mg_phi, mg_chi, mg_nu_sig_f
    use dgm, only : dgm_order, phi_m_zero

    ! Variable definitions
    integer, intent(in) :: &
      g,     & ! Group index
      c        ! Cell index
    integer :: &
      m        ! Material index
    double precision :: &
      phi      ! Scalar flux container
    double precision, intent(inout) :: &
      source   ! Source for group g in cell c

    m = mg_mMap(c)
    phi = merge(phi_m_zero(0,c,g), mg_phi(0,c,g), dgm_order > 0)
    source = source + 0.5 * mg_chi(m,g) * mg_nu_sig_f(m,g) * phi / keff

  end subroutine compute_within_group_fission

  subroutine compute_delta(g, c, a, source)
    ! ##########################################################################
    ! Compute the delta source into group g from g in cell c and angle a
    ! ##########################################################################

    ! Use Statements
    use dgm, only : delta_m, psi_m_zero, dgm_order

    ! Variable definitions
    integer, intent(in) :: &
      g,     & ! Group index
      a,     & ! Angle index
      c        ! Cell index
    double precision, intent(inout) :: &
      source   ! Source for group g

    source = source - delta_m(a,c,g,dgm_order) * psi_m_zero(a,c,g)

  end subroutine compute_delta

end module sources
