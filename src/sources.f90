module sources
  ! ############################################################################
  ! Compute the sources for the solver
  ! ############################################################################

  implicit none

  contains

  subroutine compute_in_source
    ! ##########################################################################
    ! Compute the sources into group g from gp (excluding from g <- g)
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_source, update_fission_density
    use control, only : number_cells, number_angles, allow_fission, solver_type, &
                        number_groups
    use dgm, only : dgm_order

    ! Variable definitions
    integer :: &
      g, & ! Group index
      a, & ! Angle index
      c    ! Cell index

    ! Reset the sources
    mg_source = 0.0

    ! Update the fission density if needed
    if (allow_fission .or. solver_type == 'eigen') then
      if (solver_type == 'fixed' .or. dgm_order > 0) then
        call update_fission_density()
      end if
    end if

    ! Compute the source
    do c = 1, number_cells
      do a = 1, number_angles * 2
        do g = 1, number_groups
          ! Add the external source
          mg_source(g,a,c) = mg_source(g,a,c) + compute_external(g, c, a)

          ! Add the fission source
          if (allow_fission .or. solver_type == 'eigen') then
            mg_source(g,a,c) = mg_source(g,a,c) + compute_fission(g, c)
          end if

          ! Add the scatter source
          mg_source(g,a,c) = mg_source(g,a,c) + compute_in_scatter(g, c, a)
        end do
      end do
    end do

  end subroutine compute_in_source

  function compute_within_group_source(g, c, a) result(source)
    ! ##########################################################################
    ! Compute the sources into group g from g in cell c and angle a
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_source
    use control, only : use_DGM, allow_fission, solver_type

    ! Variable definitions
    integer, intent(in) :: &
      g,   & ! Group index
      a,   & ! Angle index
      c      ! Cell index
    double precision :: &
      source ! source for group g, cell c, and angle a

    ! Get the sources into group g
    source = mg_source(g,a,c)

    ! Add the scatter source
    source = source + compute_within_group_scatter(g, c, a)

    ! Add the delta source if DGM and not higher moment
    if (use_DGM) then
      source = source + compute_delta(g, c, a)
    end if

  end function compute_within_group_source

  function compute_external(g, c, a) result(source)
    ! ##########################################################################
    ! Compute the external and fixed sources into group g
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_constant_source
    use dgm, only : source_m, dgm_order
    use control, only: use_DGM

    ! Variable definitions
    integer, intent(in) :: &
      g,   & ! Group index
      a,   & ! Angle index
      c      ! Cell index
    double precision :: &
      source ! Source for group g

    if (use_DGM) then
      source = source_m(g,a,c,dgm_order)
    else
      source = mg_constant_source
    end if

  end function compute_external

  function compute_in_scatter(g, c, a) result(source)
    ! ##########################################################################
    ! Compute the scatter source into group g from gp (excluding from g <- g)
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_mMap, mg_phi, mg_sig_s
    use angle, only : p_leg
    use control, only : number_legendre, number_groups, use_DGM
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
    double precision :: &
      source   ! Source for group g, cell c, and angle a

    source = 0.0

    m = mg_mMap(c)

    do gp = 1, number_groups
      ! Ignore the within-group terms
      if (g == gp) then
        cycle
      end if

      do l = 0, number_legendre
        if (use_DGM .and. dgm_order > 0) then
          phi = phi_m_zero(l,gp,c)
        else
          phi = mg_phi(l,gp,c)
        end if
        source = source + 0.5 * p_leg(l,a) * mg_sig_s(l,gp,g,m) * phi
      end do
    end do

  end function compute_in_scatter

  function compute_within_group_scatter(g, c, a) result(source)
    ! ##########################################################################
    ! Compute the scatter source into group g from g
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_mMap, mg_phi, mg_sig_s
    use angle, only : p_leg
    use control, only : number_legendre, use_DGM
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
    double precision :: &
      source   ! Source for group g, cell c, and angle a

    source = 0.0
    m = mg_mMap(c)
    do l = 0, number_legendre
      if (use_DGM .and. dgm_order > 0) then
        phi = phi_m_zero(l,g,c)
      else
        phi = mg_phi(l,g,c)
      end if
      source = source + 0.5 * p_leg(l,a) * mg_sig_s(l,g,g,m) * phi
    end do

  end function compute_within_group_scatter

  function compute_fission(g, c) result(source)
    ! ##########################################################################
    ! Compute the fission source into group g from gp (excluding from g <- g)
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_mMap, keff, mg_chi, mg_density
    use control, only : number_groups

    ! Variable definitions
    integer, intent(in) :: &
      g,     & ! Group index
      c        ! Cell index
    integer :: &
      gp,    & ! Group prime index
      m        ! Material index
    double precision :: &
      source   ! Source for group g

    source = 0.5 * mg_chi(g,mg_mMap(c)) * mg_density(c) / keff

  end function compute_fission

  function compute_delta(g, c, a) result(source)
    ! ##########################################################################
    ! Compute the delta source into group g from g in cell c and angle a
    ! ##########################################################################

    ! Use Statements
    use dgm, only : delta_m, psi_m_zero, dgm_order
    use state, only : mg_mMap

    ! Variable definitions
    integer, intent(in) :: &
      g,     & ! Group index
      a,     & ! Angle index
      c        ! Cell index
    double precision :: &
      source   ! Source for group g


    source = -delta_m(g,a,mg_mMap(c),dgm_order) * psi_m_zero(g,a,c)

  end function compute_delta

end module sources
