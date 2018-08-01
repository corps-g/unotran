module source
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

  subroutine compute_within_group_source(g, source)
    ! ##########################################################################
    ! Compute the sources into group g from g
    ! ##########################################################################

    ! Use Statements
    use state, only : mg_source

    ! Variable definitions
    integer :: &
      g  ! Group index

    ! Get the sources into group g
    source = mg_source(g)

    ! Add the fission source
    call compute_within_group_fission(g, source)

    ! Add the scatter source
    call compute_within_group_scatter(g, source)

    ! Add the delta source if DGM and not higher moment
    if (use_DGM .and. .not. higher_dgm_flag) then
      call compute_delta(g, source)
    end if


  end subroutine compute_within_group_source

  subroutine compute_external(g, source)
    ! ##########################################################################
    ! Compute the external and fixed sources into group g
    ! ##########################################################################

    ! Use Statements
    use control, only : source_value

    source = source + 0.5 * source_value

  end subroutine compute_external

  subroutine compute_in_scatter(g)
    ! ##########################################################################
    ! Compute the scatter source into group g from gp (excluding from g <- g)
    ! ##########################################################################

    ! Add the in-scattering source for each Legendre moment
    do gp = 1, number_groups
      ! Ignore the within-group terms
      if (g == gp) then
        cycle
      end if

      do a = 1, 2 * number_angles
        do c = 1, number_cells
          mat = mg_mMap(c)
          do l = 0, number_legendre
            source(c,a) = source(c,a) + 0.5 * p_leg(l, a) * mg_sig_s(l, mat, gp, g) * phi(l,c,gp)
          end do
        end do
      end do
    end do

  end subroutine compute_in_scatter

  subroutine compute_within_group_scatter(g)
    ! ##########################################################################
    ! Compute the scatter source into group g from g
    ! ##########################################################################

    ! Include the within-group scattering into source

    do c = 1, number_cells
      do a = 1, 2 * number_angles
        m = mg_mMap(c)
        do l = 0, number_legendre
          source(a,c) = source(a,c) + 0.5 * p_leg(l,a) * mg_sig_s(l,m,g,g) * phi_g(l,c)
        end do
      end do
    end do

  end subroutine compute_within_group_scatter

  subroutine compute_in_fission(g, source)
    ! ##########################################################################
    ! Compute the fission source into group g from gp (excluding from g <- g)
    ! ##########################################################################

    do gp = 1, number_groups
      if (g == gp) then
        cycle
      end if
      do c = 1, number_cells
        m = mg_mMap(c)
        source(c,g) = source(c,g) + 0.5 * mg_chi(m, g) * mg_nu_sig_f(m,gp) * phi(0,c,gp) / keff
      end do
    end do

  end subroutine compute_in_fission

  subroutine compute_within_group_fission(g)
    ! ##########################################################################
    ! Compute the fission source into group g from g
    ! ##########################################################################

    do c = 1, number_cells
      m = mg_mMap(c)
      source(c,g) = source(c,g) + 0.5 * mg_chi(m,g) * mg_nu_sig_f(m,g) * phi(0,c,g) / keff
    end do

  end subroutine compute_within_group_fission

  subroutine compute_delta(g, source)
    ! ##########################################################################
    ! Compute the delta source into group g from g
    ! ##########################################################################

    source(:,:) = source(:,:) - delta_m(:,:,g,0) * psi(:,:,g)

  end subroutine compute_delta

end module source
