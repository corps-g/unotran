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
                        scatter_leg_order, use_DGM, spatial_dimension
    use dgm, only : dgm_order, phi_m, source_m

    ! Variable definitions
    integer :: &
      g,   & ! Group index
      c,   & ! Cell index
      mat, & ! Material index
      l,   & ! Legendre index
      m,   & ! Moment index
      ll,  & ! Total moment index
      ord    ! short name for scatter_legendre_order
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

    dgm_switch = use_DGM .and. dgm_order > 0

    sigphi = 0.0
    ord = scatter_leg_order

    ! Compute the source
    do c = 1, number_cells
      mat = mg_mMap(c)

      ! Add the external source
      if (use_DGM) then
        mg_source(:,c) = mg_source(:,c) + source_m(:,dgm_order)
      else
        mg_source(:,c) = mg_source(:,c) + mg_constant_source
      end if

      ! Add the fission source
      if (allow_fission .or. solver_type == 'eigen') then
        mg_source(:,c) = mg_source(:,c) + scaling * mg_chi(:,mat) * mg_density(c) / keff
      end if

      ! Compute the scattering matrix
      if (spatial_dimension == 1) then
        if (dgm_switch) then
          do g = 1, number_groups
            sigphi(:,g,c) = sum(phi_m(0, 0:ord,:,c) * mg_sig_s(:,:,g,mat), 2)
          end do  ! End g loop
        else
          do g = 1, number_groups
            sigphi(:,g,c) = sum(mg_phi(0:ord,:,c) * mg_sig_s(:,:,g,mat), 2)
          end do  ! End g loop
        end if
        do l = 0, scatter_leg_order
          sigphi(l,:,c) = sigphi(l,:,c) * (2 * l + 1) * scaling
        end do
      else
        if (dgm_switch) then
          do g = 1, number_groups
            ll = 0
            do l = 0, scatter_leg_order
              do m = -l, l
                sigphi(ll,g,c) = dot_product(phi_m(0,ll,:,c), mg_sig_s(l,:,g,mat)) * scaling
                ll = ll + 1
              end do  ! End m loop
            end do  ! End l loop
          end do  ! End g loop
        else
          do g = 1, number_groups
            ll = 0
            do l = 0, scatter_leg_order
              do m = -l, l
                sigphi(ll,g,c) = dot_product(mg_phi(ll,:,c), mg_sig_s(l,:,g,mat)) * scaling
                ll = ll + 1
              end do  ! End m loop
            end do  ! End l loop
          end do  ! End g loop
        end if
      end if
    end do  ! End c loop



  end subroutine compute_source

end module sources
