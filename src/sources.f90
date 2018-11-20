module sources
  ! ############################################################################
  ! Compute the sources for the solver
  ! ############################################################################

  implicit none

  contains

  subroutine compute_source()
    ! ##########################################################################
    ! Compute the sources into group g from gp
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_source, update_fission_density
    use control, only : number_cells, allow_fission, solver_type, number_groups
    use dgm, only : dgm_order

    ! Variable definitions
    integer :: &
      g, & ! Group index
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
      do g = 1, number_groups
        ! Add the external source
        mg_source(g,c) = mg_source(g,c) + compute_external(g)

        ! Add the fission source
        if (allow_fission .or. solver_type == 'eigen') then
          mg_source(g,c) = mg_source(g,c) + compute_fission(g, c)
        end if
      end do
    end do

  end subroutine compute_source

  function add_transport_sources(g, c, a) result(source)
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
    double precision :: &
      source ! source for group g, cell c, and angle a

    ! Get the sources into group g
    source = mg_source(g, c)

    ! Add the scatter source
    source = source + compute_scatter(g, c, a)

    ! Add the delta source if DGM
    if (use_DGM) then
      source = source + compute_delta(g, c, a)
    end if

  end function add_transport_sources

  function compute_external(g) result(source)
    ! ##########################################################################
    ! Compute the external and fixed sources into group g
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_constant_source
    use dgm, only : source_m, dgm_order
    use control, only: use_DGM

    ! Variable definitions
    integer, intent(in) :: &
      g      ! Group index
    double precision :: &
      source ! Source for group g

    if (use_DGM) then
      source = source_m(g,dgm_order)
    else
      source = mg_constant_source
    end if

  end function compute_external

  function compute_scatter(g, c, a) result(source)
    ! ##########################################################################
    ! Compute the scatter source into group g from gp
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_mMap, mg_phi, mg_sig_s
    use angle, only : p_leg
    use control, only : scatter_legendre_order, number_groups, use_DGM
    use dgm, only : dgm_order, phi_m_zero

    ! Variable definitions
    integer, intent(in) :: &
      g        ! Group index
    integer :: &
      a,     & ! Angle index
      c,     & ! Cell index
      ord      ! short name for scatter_legendre_order
    double precision, dimension(0:scatter_legendre_order, number_groups) :: &
      sigphi      ! Scalar flux container
    double precision :: &
      source   ! Source for group g, cell c, and angle a

    ord = scatter_legendre_order

    if (use_DGM .and. dgm_order > 0) then
      sigphi(:,:) = phi_m_zero(0:ord,:,c) * mg_sig_s(:,:,g,mg_mMap(c))
    else
      sigphi(:,:) = mg_phi(0:ord,:,c) * mg_sig_s(:,:,g,mg_mMap(c))
    end if

    source = 0.5 * sum(matmul(p_leg(:ord,a), sigphi))

  end function compute_scatter

  function compute_fission(g, c) result(source)
    ! ##########################################################################
    ! Compute the fission source into group g from gp (excluding from g <- g)
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_mMap, keff, mg_chi, mg_density

    ! Variable definitions
    integer, intent(in) :: &
      g,     & ! Group index
      c        ! Cell index
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
