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
                      mg_mMap, mg_phi, keff, mg_chi, mg_density, mg_constant_source, &
                      scaling
    use control, only : number_cells, allow_fission, solver_type, number_groups, &
                        scatter_leg_order, use_DGM
    use dgm, only : dgm_order, phi_m, source_m

    ! Variable definitions
    integer :: &
      g, & ! Group index
      c, & ! Cell index
      m,        & ! Material index
      l,  & ! Legendre index
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

    ord = scatter_leg_order
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
        mg_source(:,c) = mg_source(:,c) + scaling * mg_chi(:,m) * mg_density(c) / keff
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

    do l = 0, scatter_leg_order
      sigphi(l,:,:) = sigphi(l,:,:) * (2 * l + 1) * scaling
    end do

  end subroutine compute_source

end module sources
