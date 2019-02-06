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
    use state, only : mg_source, update_fission_density, sigphi, mg_sig_s, &
                      mg_mMap, mg_phi, keff, mg_chi, mg_density, mg_constant_source
    use control, only : number_cells, allow_fission, solver_type, number_groups, &
                        scatter_legendre_order, use_DGM
    use dgm, only : dgm_order, phi_m, source_m

    ! Variable definitions
    integer :: &
      g, & ! Group index
      c, & ! Cell index
      m,        & ! Material index
      ord  ! short name for scatter_legendre_order
    logical :: &
      dgm_switch    !

    ! Reset the sources
    mg_source = 0.0

    ! Update the fission density if needed
    if (allow_fission .or. solver_type == 'eigen') then
      if (solver_type == 'fixed' .or. dgm_order > 0) then
        call update_fission_density()
      end if
    end if

    ord = scatter_legendre_order
    dgm_switch = use_DGM .and. dgm_order > 0

    sigphi = 0.0

    ! Compute the source
    do c = 1, number_cells
      m = mg_mMap(c)

      ! Add the external source
      if (use_DGM) then
        mg_source(:,c) = mg_source(:,c) + source_m(:,dgm_order)
      else
        mg_source(:,c) = mg_source(:,c) + mg_constant_source
      end if

      ! Add the fission source
      if (allow_fission .or. solver_type == 'eigen') then
        mg_source(:,c) = mg_source(:,c) + 0.5 * mg_chi(:,m) * mg_density(c) / keff
      end if

      if (dgm_switch) then
        do g = 1, number_groups
          sigphi(:,g,c) = sum(phi_m(0, 0:ord,:,c) * mg_sig_s(:,:,g,m), 2)
        end do
      else
        do g = 1, number_groups
          sigphi(:,g,c) = sum(mg_phi(0:ord,:,c) * mg_sig_s(:,:,g,m), 2)
        end do
      end if
    end do

  end subroutine compute_source

  function add_transport_sources(g, c, a) result(source)
    ! ##########################################################################
    ! Compute the sources into group g from g in cell c and angle a
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_source, sigphi, mg_mMap
    use control, only : use_DGM, scatter_legendre_order
    use angle, only : p_leg
    use dgm, only : delta_m, psi_m, dgm_order

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
    source = source + 0.5 * dot_product(p_leg(:scatter_legendre_order,a), sigphi(:scatter_legendre_order,g,c))

    ! Add the delta source if DGM
    if (use_DGM) then
      source = source - delta_m(g, a, mg_mMap(c), dgm_order) * psi_m(0, g, a, c)
    end if

  end function add_transport_sources

end module sources
